"""ExecutionCore demo: one simulated trading day of order flow.

Runs market/limit/stop/stop-limit/IOC/FOK orders plus TWAP and VWAP parent
orders against a deterministic synthetic intraday feed, then writes hub
results JSON. Run with::

    cd executioncore/ && python3 -m executioncore.demo
"""

from __future__ import annotations

import json
import math
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from executioncore.algos import TWAP, VWAP, AlgoExecutor, run_session
from executioncore.analytics import algo_comparison, executor_shortfall, fill_rate, slippage_bps
from executioncore.broker import PaperBroker
from executioncore.costs import PerShareCommission, VolumeImpactSlippage
from executioncore.market import synthetic_intraday_feed
from executioncore.oms import OrderManager
from executioncore.orders import Order, OrderSide, OrderStatus, OrderType, TimeInForce
from executioncore.risk import MaxNotional, MaxOrderSize, PriceBand

SYMBOL = "SYNTH"
SEED = 7
HUB_DATA_DIR = Path(__file__).resolve().parents[2] / "hub" / "data"


def _clean(value: float | None, digits: int = 4) -> float | None:
    """Round and convert to plain float; NaN/inf become None (never NaN in JSON)."""
    if value is None:
        return None
    value = float(value)
    if math.isnan(value) or math.isinf(value):
        return None
    return round(value, digits)


def build_scenario() -> tuple[PaperBroker, OrderManager, dict[int, list[Order]], list[AlgoExecutor]]:
    feed = synthetic_intraday_feed(symbol=SYMBOL, n_bars=390, start_price=100.0, seed=SEED)
    broker = PaperBroker(
        feed,
        slippage=VolumeImpactSlippage(impact_bps=25.0),
        commission=PerShareCommission(rate=0.005, minimum=1.0),
        latency_bars=1,
        participation_cap=0.10,
    )
    oms = OrderManager(
        broker,
        cash=5_000_000.0,
        validators=(MaxOrderSize(30_000), MaxNotional(3_000_000), PriceBand(0.05)),
    )

    closes = feed.closes(SYMBOL)

    def mark_before(bar: int) -> float:
        return float(closes[bar - 1]) if bar > 0 else feed.bar(SYMBOL, 0).open

    submissions: dict[int, list[Order]] = {
        5: [Order(SYMBOL, OrderSide.BUY, 1_200)],
        12: [Order(SYMBOL, OrderSide.SELL, 900)],
        20: [
            Order(
                SYMBOL,
                OrderSide.BUY,
                800,
                OrderType.LIMIT,
                limit_price=round(mark_before(20) * 0.9995, 2),
            )
        ],
        40: [
            Order(
                SYMBOL,
                OrderSide.SELL,
                1_500,
                OrderType.LIMIT,
                limit_price=round(mark_before(40) * 1.0015, 2),
                time_in_force=TimeInForce.GTC,
            )
        ],
        60: [
            # Deep out-of-the-money DAY limit: expires at the close.
            Order(
                SYMBOL,
                OrderSide.BUY,
                600,
                OrderType.LIMIT,
                limit_price=round(mark_before(60) * 0.97, 2),
            )
        ],
        80: [
            Order(
                SYMBOL,
                OrderSide.SELL,
                1_000,
                OrderType.STOP,
                stop_price=round(mark_before(80) * 0.995, 2),
            )
        ],
        100: [
            Order(
                SYMBOL,
                OrderSide.BUY,
                700,
                OrderType.STOP_LIMIT,
                stop_price=round(mark_before(100) * 1.003, 2),
                limit_price=round(mark_before(100) * 1.006, 2),
            )
        ],
        130: [
            # Far larger than one bar's participation cap: partial then cancel.
            Order(SYMBOL, OrderSide.BUY, 5_000, time_in_force=TimeInForce.IOC)
        ],
        150: [Order(SYMBOL, OrderSide.SELL, 6_000, time_in_force=TimeInForce.FOK)],
        200: [
            # Fat finger: limit 20% above the market -> PriceBand rejection.
            Order(
                SYMBOL,
                OrderSide.BUY,
                500,
                OrderType.LIMIT,
                limit_price=round(mark_before(200) * 1.20, 2),
            )
        ],
        220: [Order(SYMBOL, OrderSide.BUY, 60_000)],  # MaxOrderSize rejection
        300: [Order(SYMBOL, OrderSide.SELL, 1_000)],
    }

    twap_parent = Order(SYMBOL, OrderSide.BUY, 6_000)
    vwap_parent = Order(SYMBOL, OrderSide.BUY, 6_000)
    executors = [
        # Decision stamped 5 bars before the execution window opens.
        AlgoExecutor(oms, twap_parent, TWAP(), start_index=250, n_slices=20, decision_index=245),
        AlgoExecutor(oms, vwap_parent, VWAP(), start_index=250, n_slices=20, decision_index=245),
    ]
    return broker, oms, submissions, executors


def run_demo() -> dict:
    broker, oms, submissions, executors = build_scenario()
    feed = broker.feed
    run_session(broker, submissions=submissions, executors=executors, oms=oms)

    final_price = feed.bar(SYMBOL, feed.n_bars - 1).close
    all_orders = list(oms.orders.values())
    flat_orders = [o for o in all_orders if o.parent_id is None]
    rejected = [o for o in all_orders if o.status == OrderStatus.REJECTED]

    slippages = [s for s in (slippage_bps(o) for o in all_orders) if s is not None]
    fills = [f for o in all_orders for f in o.fills]
    overall_fill_rate = fill_rate(all_orders)
    twap_report = executor_shortfall(executors[0], final_price)
    vwap_report = executor_shortfall(executors[1], final_price)
    snapshot = oms.snapshot()

    summary_metrics = [
        {"label": "Orders submitted", "value": str(len(all_orders))},
        {"label": "Fill rate", "value": f"{overall_fill_rate * 100:.1f}", "unit": "%"},
        {"label": "Avg slippage vs arrival", "value": f"{np.mean(slippages):.2f}", "unit": "bps"},
        {"label": "TWAP shortfall", "value": f"{twap_report.total_bps:.2f}", "unit": "bps"},
        {"label": "VWAP shortfall", "value": f"{vwap_report.total_bps:.2f}", "unit": "bps"},
        {"label": "Total commission", "value": f"{snapshot.total_commission:.2f}", "unit": "$"},
        {"label": "Risk rejections", "value": str(len(rejected))},
        {"label": "Session PnL", "value": f"{snapshot.equity - oms.starting_cash:,.0f}", "unit": "$"},
    ]

    # Chart 1: intraday price.
    closes = feed.closes(SYMBOL)
    price_chart = {
        "id": "intraday_price",
        "title": "Synthetic intraday price (1-min bars)",
        "type": "line",
        "x_label": "Bar (minute)",
        "y_label": "Price ($)",
        "x": list(range(feed.n_bars)),
        "series": [{"name": f"{SYMBOL} close", "data": [_clean(c) for c in closes]}],
    }

    # Chart 2: cumulative execution cost vs arrival, per algo slice.
    cost_series = []
    slice_count = max(len(ex.children) for ex in executors)
    for ex in executors:
        arrival = ex.arrival_price
        sign = ex.parent.side.sign
        costs = []
        cumulative = 0.0
        for child in ex.children:
            for f in child.fills:
                cumulative += sign * (f.price - arrival) * f.quantity + f.commission
            costs.append(round(cumulative, 2))
        costs += [costs[-1]] * (slice_count - len(costs))
        cost_series.append({"name": ex.algo.name, "data": costs})
    cost_chart = {
        "id": "algo_cum_cost",
        "title": "Cumulative execution cost vs arrival (TWAP vs VWAP)",
        "type": "line",
        "x_label": "Child slice #",
        "y_label": "Cumulative cost ($)",
        "x": list(range(1, slice_count + 1)),
        "series": cost_series,
    }

    # Chart 3: slippage distribution across all fills.
    fill_slips = [
        f.side.sign
        * (f.price - broker.get_order(f.order_id).arrival_price)
        / broker.get_order(f.order_id).arrival_price
        * 1e4
        for f in fills
    ]
    edges = np.histogram_bin_edges(fill_slips, bins=10)
    counts, _ = np.histogram(fill_slips, bins=edges)
    slip_chart = {
        "id": "slippage_dist",
        "title": "Per-fill slippage vs arrival price",
        "type": "bar",
        "x_label": "Slippage (bps)",
        "y_label": "Fill count",
        "x": [f"{edges[i]:.1f} to {edges[i + 1]:.1f}" for i in range(len(counts))],
        "series": [{"name": "Fills", "data": [int(c) for c in counts]}],
    }

    # Table 1: algo comparison.
    comparison = algo_comparison(executors, feed, final_price=final_price)
    algo_table = {
        "id": "algo_comparison",
        "title": "Execution algo comparison",
        "columns": [
            "Algo",
            "Side",
            "Qty",
            "Filled",
            "Children",
            "Avg fill px",
            "Bench VWAP",
            "Slip vs arrival (bps)",
            "Slip vs VWAP (bps)",
            "Shortfall (bps)",
            "Commission ($)",
        ],
        "rows": [
            [
                row["algo"],
                row["side"],
                _clean(row["quantity"], 0),
                _clean(row["filled_quantity"], 0),
                row["n_children"],
                _clean(row["avg_fill_price"]),
                _clean(row["benchmark_vwap"]),
                _clean(row["slippage_vs_arrival_bps"], 2),
                _clean(row["slippage_vs_vwap_bps"], 2),
                _clean(row["shortfall_bps"], 2),
                _clean(row["commission"], 2),
            ]
            for row in comparison
        ],
    }

    # Table 2: order summary for the hand-submitted (non-child) orders.
    order_table = {
        "id": "order_summary",
        "title": "Parent / standalone order summary",
        "columns": ["Order", "Type", "TIF", "Side", "Qty", "Filled", "Avg px", "Slip (bps)", "Status"],
        "rows": [
            [
                o.order_id,
                o.order_type.value,
                o.time_in_force.value,
                o.side.value,
                _clean(o.quantity, 0),
                _clean(o.filled_quantity, 0),
                _clean(o.avg_fill_price) if o.filled_quantity else None,
                _clean(slippage_bps(o), 2),
                o.status.value,
            ]
            for o in flat_orders
        ],
    }

    # Table 3: blotter excerpt (audit trail).
    blotter_rows = [
        [
            e.bar_index,
            e.order_id,
            e.event,
            e.side,
            _clean(e.quantity, 0),
            _clean(e.price),
            e.detail[:60],
        ]
        for e in oms.blotter[:20]
    ]
    blotter_table = {
        "id": "blotter",
        "title": "Blotter excerpt (first 20 events)",
        "columns": ["Bar", "Order", "Event", "Side", "Qty", "Price", "Detail"],
        "rows": blotter_rows,
    }

    is_table = {
        "id": "shortfall_decomposition",
        "title": "Implementation shortfall decomposition (Perold)",
        "columns": ["Algo", "Delay ($)", "Trading ($)", "Opportunity ($)", "Commission ($)", "Total ($)", "Total (bps)"],
        "rows": [
            [
                name,
                _clean(r.delay_cost, 2),
                _clean(r.trading_cost, 2),
                _clean(r.opportunity_cost, 2),
                _clean(r.commission, 2),
                _clean(r.total_cost, 2),
                _clean(r.total_bps, 2),
            ]
            for name, r in (("TWAP", twap_report), ("VWAP", vwap_report))
        ],
    }

    return {
        "schema_version": 1,
        "product": "executioncore",
        "title": "ExecutionCore",
        "tagline": "Order management and execution simulation with TWAP/VWAP and shortfall analytics",
        "version": "0.1.0",
        "generated_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "status": "ok",
        "summary_metrics": summary_metrics,
        "charts": [price_chart, cost_chart, slip_chart],
        "tables": [algo_table, is_table, order_table, blotter_table],
        "notes": (
            "Offline MVP: live broker adapters (Alpaca/IBKR/CCXT) from the design doc are "
            "replaced by a deterministic PaperBroker behind a Broker ABC. One synthetic "
            "session (390 1-min bars, seed 7), volume-impact slippage, per-share commission, "
            "1-bar latency, 10% volume participation cap."
        ),
    }


def main() -> None:
    payload = run_demo()
    HUB_DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = HUB_DATA_DIR / "executioncore.json"
    with open(out_path, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"ExecutionCore demo complete -> {out_path}")
    for metric in payload["summary_metrics"]:
        unit = metric.get("unit", "")
        print(f"  {metric['label']}: {metric['value']}{unit}")


if __name__ == "__main__":
    main()
