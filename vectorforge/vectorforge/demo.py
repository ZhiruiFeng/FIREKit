"""
VectorForge hub demo.

Runs a cross-sectional momentum portfolio backtest on a synthetic universe and
writes results for the FIREKit hub (see hub/SCHEMA.md).

Usage: cd vectorforge && python3 -m vectorforge.demo
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from vectorforge import VectorizedBacktester, __version__
from vectorforge.portfolio import (
    CrossSectionalSignal,
    MissingDataPolicy,
    PortfolioData,
    PortfolioMetrics,
    RebalanceFrequency,
    Rebalancer,
)

HUB_DATA_DIR = Path(__file__).resolve().parents[2] / "hub" / "data"

N_SYMBOLS = 40
N_YEARS = 4
SEED = 7
INITIAL_CAPITAL = 1_000_000
SECTORS = ["Technology", "Financials", "Healthcare", "Energy", "Consumer"]


def _make_universe(seed: int = SEED) -> tuple[PortfolioData, dict[str, str]]:
    """Build a seeded GBM universe with per-symbol drift so momentum has signal."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2021-01-04", periods=N_YEARS * 252)
    frames: dict[str, pd.DataFrame] = {}
    sector_map: dict[str, str] = {}
    for i in range(N_SYMBOLS):
        symbol = f"SYM{i:02d}"
        drift = rng.normal(0.0003, 0.0004)
        vol = rng.uniform(0.01, 0.025)
        rets = rng.normal(drift, vol, len(dates))
        close = 100 * np.cumprod(1 + rets)
        frames[symbol] = pd.DataFrame(
            {
                "open": close * (1 + rng.normal(0, 0.001, len(dates))),
                "high": close * (1 + np.abs(rng.normal(0, 0.004, len(dates)))),
                "low": close * (1 - np.abs(rng.normal(0, 0.004, len(dates)))),
                "close": close,
                "volume": rng.integers(1e5, 5e6, len(dates)).astype(float),
            },
            index=dates,
        )
        sector_map[symbol] = SECTORS[i % len(SECTORS)]
    data = PortfolioData.from_dict(frames).align(policy=MissingDataPolicy.FORWARD_FILL)
    return data.validate(), sector_map


def _downsample(values: list, max_points: int = 500) -> list:
    if len(values) <= max_points:
        return values
    step = -(-len(values) // max_points)  # ceil division
    return values[::step]


def _round(x: float, digits: int = 4) -> float:
    return float(np.round(float(x), digits))


def run_demo() -> dict:
    data, sector_map = _make_universe()

    momentum = CrossSectionalSignal.momentum(lookback=126, skip_recent=10)
    weights = momentum.generate(data).top_percentile(20).to_weights(method="equal")
    rebalancer = Rebalancer.calendar(
        frequency=RebalanceFrequency.MONTHLY, turnover_limit=0.30
    )

    engine = VectorizedBacktester()
    result = engine.run_portfolio(
        strategy=weights,
        data=data,
        rebalancer=rebalancer,
        initial_capital=INITIAL_CAPITAL,
    )

    # Equal-weight buy-and-hold benchmark on the same universe
    asset_rets = np.nan_to_num(data.returns(), nan=0.0)
    bench_rets = asset_rets.mean(axis=0)
    bench_equity = INITIAL_CAPITAL * np.cumprod(1 + bench_rets)

    metrics = PortfolioMetrics.from_backtest_result(
        result, sector_map=sector_map, risk_free_rate=0.04
    )
    concentration = metrics.concentration_metrics()
    diversification = metrics.diversification_metrics()
    attribution = metrics.return_attribution()

    equity = result.equity_curve
    dates = [d.strftime("%Y-%m-%d") for d in equity.index]
    strat_vals = [_round(v, 2) for v in equity.tolist()]
    bench_vals = [_round(v, 2) for v in bench_equity.tolist()]
    # Equity curve has n_dates-1 points (post-first-return); align benchmark
    n = min(len(strat_vals), len(bench_vals), len(dates))

    running_max = np.maximum.accumulate(equity.to_numpy())
    drawdown = (equity.to_numpy() / running_max - 1) * 100

    top_contributors = sorted(attribution, key=lambda a: a.contribution, reverse=True)[:8]

    return {
        "schema_version": 1,
        "product": "vectorforge",
        "title": "VectorForge",
        "tagline": "High-performance backtesting engine — multi-asset portfolio mode",
        "version": __version__,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "status": "ok",
        "summary_metrics": [
            {"label": "Total return", "value": f"{result.total_return:.1%}"},
            {"label": "Sharpe ratio", "value": f"{result.sharpe_ratio:.2f}"},
            {"label": "Max drawdown", "value": f"{result.max_drawdown:.1%}"},
            {"label": "Universe", "value": f"{data.n_symbols} symbols"},
            {"label": "Rebalances", "value": str(len(result.rebalance_dates))},
            {
                "label": "Avg turnover",
                "value": f"{result.turnover_history.mean():.1%}"
                if len(result.turnover_history)
                else "0%",
            },
            {"label": "HHI", "value": f"{concentration.herfindahl_index:.3f}"},
            {
                "label": "Diversification ratio",
                "value": f"{diversification.diversification_ratio:.2f}",
            },
        ],
        "charts": [
            {
                "id": "equity",
                "title": "Momentum portfolio vs equal-weight benchmark",
                "type": "line",
                "x_label": "Date",
                "y_label": "Equity ($)",
                "x": _downsample(dates[:n]),
                "series": [
                    {"name": "Top-20% momentum", "data": _downsample(strat_vals[:n])},
                    {"name": "Equal-weight universe", "data": _downsample(bench_vals[:n])},
                ],
            },
            {
                "id": "drawdown",
                "title": "Strategy drawdown",
                "type": "line",
                "x_label": "Date",
                "y_label": "Drawdown (%)",
                "x": _downsample(dates[:n]),
                "series": [
                    {
                        "name": "Drawdown",
                        "data": _downsample([_round(v, 2) for v in drawdown[:n].tolist()]),
                    }
                ],
            },
            {
                "id": "attribution",
                "title": "Top contributors to portfolio return",
                "type": "bar",
                "x_label": "Symbol",
                "y_label": "Contribution (%)",
                "x": [a.symbol for a in top_contributors],
                "series": [
                    {
                        "name": "Contribution",
                        "data": [_round(a.contribution * 100, 2) for a in top_contributors],
                    }
                ],
            },
        ],
        "tables": [
            {
                "id": "metrics",
                "title": "Portfolio analytics",
                "columns": ["Metric", "Value"],
                "rows": [
                    ["Total return", f"{result.total_return:.2%}"],
                    ["Sharpe ratio", f"{result.sharpe_ratio:.2f}"],
                    ["Max drawdown", f"{result.max_drawdown:.2%}"],
                    ["HHI (concentration)", f"{concentration.herfindahl_index:.4f}"],
                    ["Effective N", f"{concentration.effective_n:.1f}"],
                    ["Top-5 weight", f"{concentration.top_5_weight:.1%}"],
                    [
                        "Diversification ratio",
                        f"{diversification.diversification_ratio:.2f}",
                    ],
                    ["Avg pairwise correlation", f"{diversification.avg_correlation:.2f}"],
                ],
            },
            {
                "id": "contributors",
                "title": "Return attribution (top 8)",
                "columns": ["Symbol", "Sector", "Contribution"],
                "rows": [
                    [a.symbol, sector_map[a.symbol], f"{a.contribution:.2%}"]
                    for a in top_contributors
                ],
            },
        ],
        "notes": (
            f"Cross-sectional momentum (126d lookback, 10d skip), top quintile equal-"
            f"weighted, monthly rebalancing with a 30% turnover limit, on a seeded "
            f"{N_SYMBOLS}-symbol x {N_YEARS}-year synthetic GBM universe. "
            f"Backtested with VectorizedBacktester.run_portfolio (v0.3.0 multi-asset mode)."
        ),
    }


def main() -> None:
    payload = run_demo()
    HUB_DATA_DIR.mkdir(parents=True, exist_ok=True)
    out = HUB_DATA_DIR / "vectorforge.json"
    with out.open("w") as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
