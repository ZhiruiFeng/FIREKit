"""Execution analytics: fill rate, slippage, Perold implementation shortfall."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from executioncore.algos import AlgoExecutor
from executioncore.market import PriceFeed
from executioncore.orders import Order, OrderSide, OrderStatus


def fill_rate(orders: Iterable[Order]) -> float:
    """Filled quantity / submitted quantity over non-rejected orders."""
    submitted = 0.0
    filled = 0.0
    for order in orders:
        if order.status == OrderStatus.REJECTED:
            continue
        submitted += order.quantity
        filled += order.filled_quantity
    if submitted == 0:
        return 0.0
    return filled / submitted


def slippage_bps(order: Order) -> float | None:
    """Signed slippage of the average fill vs the arrival price, in bps.

    Positive = adverse (paid more on a buy / received less on a sell).
    """
    if order.filled_quantity <= 0 or not order.arrival_price:
        return None
    return order.side.sign * (order.avg_fill_price - order.arrival_price) / order.arrival_price * 1e4


@dataclass
class ShortfallReport:
    """Perold implementation shortfall decomposed into its classic components.

    All dollar figures are costs (positive = money lost vs the paper
    portfolio executed instantly at the decision price).
    """

    delay_cost: float
    trading_cost: float
    opportunity_cost: float
    commission: float
    total_cost: float
    total_bps: float
    filled_quantity: float
    unfilled_quantity: float
    decision_price: float
    arrival_price: float
    avg_fill_price: float | None
    final_price: float


def implementation_shortfall(
    side: OrderSide,
    quantity: float,
    filled_quantity: float,
    avg_fill_price: float | None,
    decision_price: float,
    arrival_price: float,
    final_price: float,
    commission: float = 0.0,
) -> ShortfallReport:
    """Decompose execution cost vs the decision price (Perold 1988).

    - delay cost: price drift between decision and arrival, on the filled qty
    - trading cost: execution price vs arrival, on the filled qty
    - opportunity cost: price drift over the horizon, on the unfilled qty
    """
    s = side.sign
    unfilled = quantity - filled_quantity
    if filled_quantity > 0 and avg_fill_price is not None:
        delay = s * (arrival_price - decision_price) * filled_quantity
        trading = s * (avg_fill_price - arrival_price) * filled_quantity
    else:
        delay = 0.0
        trading = 0.0
    opportunity = s * (final_price - decision_price) * unfilled
    total = delay + trading + opportunity + commission
    notional = decision_price * quantity
    return ShortfallReport(
        delay_cost=delay,
        trading_cost=trading,
        opportunity_cost=opportunity,
        commission=commission,
        total_cost=total,
        total_bps=total / notional * 1e4 if notional else 0.0,
        filled_quantity=filled_quantity,
        unfilled_quantity=unfilled,
        decision_price=decision_price,
        arrival_price=arrival_price,
        avg_fill_price=avg_fill_price if filled_quantity > 0 else None,
        final_price=final_price,
    )


def order_shortfall(order: Order, final_price: float) -> ShortfallReport:
    """Implementation shortfall for a single order."""
    decision = order.decision_price if order.decision_price is not None else order.arrival_price
    arrival = order.arrival_price if order.arrival_price is not None else decision
    if decision is None:
        raise ValueError("order has no decision or arrival price")
    return implementation_shortfall(
        side=order.side,
        quantity=order.quantity,
        filled_quantity=order.filled_quantity,
        avg_fill_price=order.avg_fill_price if order.filled_quantity else None,
        decision_price=decision,
        arrival_price=arrival,
        final_price=final_price,
        commission=order.commission,
    )


def executor_shortfall(executor: AlgoExecutor, final_price: float) -> ShortfallReport:
    """Implementation shortfall for an algo parent, rolled up from children."""
    parent = executor.parent
    decision = parent.decision_price
    arrival = executor.arrival_price if executor.arrival_price is not None else decision
    return implementation_shortfall(
        side=parent.side,
        quantity=parent.quantity,
        filled_quantity=executor.filled_quantity,
        avg_fill_price=executor.avg_fill_price if executor.filled_quantity else None,
        decision_price=decision,
        arrival_price=arrival,
        final_price=final_price,
        commission=executor.total_commission,
    )


def algo_comparison(
    executors: Iterable[AlgoExecutor], feed: PriceFeed, final_price: float | None = None
) -> list[dict[str, float | str]]:
    """Per-algo summary: fills, benchmark VWAP, slippage and shortfall in bps."""
    rows: list[dict[str, float | str]] = []
    for ex in executors:
        symbol = ex.parent.symbol
        if final_price is None:
            fp = feed.bar(symbol, feed.n_bars - 1).close
        else:
            fp = final_price
        start, end = ex.window
        bench_vwap = feed.vwap(symbol, start, end)
        report = executor_shortfall(ex, fp)
        avg = ex.avg_fill_price
        s = ex.parent.side.sign
        vs_arrival = (
            s * (avg - report.arrival_price) / report.arrival_price * 1e4 if avg else None
        )
        vs_vwap = s * (avg - bench_vwap) / bench_vwap * 1e4 if avg else None
        rows.append(
            {
                "algo": ex.algo.name,
                "side": ex.parent.side.value,
                "quantity": ex.parent.quantity,
                "filled_quantity": ex.filled_quantity,
                "n_children": len(ex.children),
                "avg_fill_price": avg,
                "benchmark_vwap": bench_vwap,
                "slippage_vs_arrival_bps": vs_arrival,
                "slippage_vs_vwap_bps": vs_vwap,
                "shortfall_bps": report.total_bps,
                "commission": ex.total_commission,
            }
        )
    return rows
