"""PortfolioEngine demo: writes hub/data/portfolioengine.json (hub schema v1).

Run with: cd portfolioengine && python3 -m portfolioengine.demo
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from portfolioengine import (
    Constraints,
    EqualWeight,
    HierarchicalRiskParity,
    InverseVolatility,
    MaxSharpe,
    MinimumVariance,
    RiskParity,
    __version__,
    annualize_cov,
    efficient_frontier,
    ledoit_wolf_cov,
    make_universe,
    portfolio_volatility,
    run_backtest,
)

HUB_DATA = Path(__file__).resolve().parents[2] / "hub" / "data"


def fnum(x: float, digits: int = 4) -> float | None:
    x = float(x)
    if not np.isfinite(x):
        return None
    return round(x, digits)


def downsample(series: pd.Series, max_points: int = 500) -> pd.Series:
    if len(series) <= max_points:
        return series
    step = int(np.ceil(len(series) / max_points))
    sampled = series.iloc[::step]
    if sampled.index[-1] != series.index[-1]:
        sampled = pd.concat([sampled, series.iloc[[-1]]])
    return sampled


def build_payload() -> dict:
    universe = make_universe(n_assets=20, n_days=1512, n_sectors=4, seed=7)
    returns = universe.returns
    constraints = Constraints(long_only=True, max_weight=0.15)

    # --- Efficient frontier (full-sample Ledoit-Wolf estimates) -------------
    mu_ann = returns.to_numpy().mean(axis=0) * 252.0
    cov_ann = annualize_cov(ledoit_wolf_cov(returns))
    frontier = efficient_frontier(mu_ann, cov_ann, n_points=25, constraints=constraints)

    ms_opt = MaxSharpe(constraints)
    w_ms = ms_opt.allocate(mu_ann, cov_ann)
    ms_vol = portfolio_volatility(w_ms, cov_ann)
    ms_ret = float(w_ms @ mu_ann)

    xs = [fnum(100 * p.volatility, 3) for p in frontier]
    frontier_rets: list[float | None] = [fnum(100 * p.expected_return, 3) for p in frontier]
    # Insert the max-Sharpe point into the shared x grid.
    insert_at = int(np.searchsorted([p.volatility for p in frontier], ms_vol))
    xs.insert(insert_at, fnum(100 * ms_vol, 3))
    frontier_rets.insert(insert_at, None)
    ms_series: list[float | None] = [None] * len(xs)
    ms_series[insert_at] = fnum(100 * ms_ret, 3)
    frontier_chart = {
        "id": "efficient_frontier",
        "title": "Efficient frontier (long-only, max weight 15%)",
        "type": "scatter",
        "x_label": "Annualized volatility (%)",
        "y_label": "Annualized return (%)",
        "x": xs,
        "series": [
            {"name": "Frontier", "data": frontier_rets},
            {"name": "Max Sharpe", "data": ms_series},
        ],
    }

    # --- Backtest comparison -------------------------------------------------
    optimizers = {
        "EqualWeight": EqualWeight(constraints),
        "InverseVol": InverseVolatility(constraints),
        "MinVariance": MinimumVariance(constraints),
        "MaxSharpe": MaxSharpe(constraints),
        "RiskParity": RiskParity(constraints),
        "HRP": HierarchicalRiskParity(constraints),
    }
    results = run_backtest(
        returns,
        optimizers,
        lookback=252,
        rebalance_every=21,
        cov_estimator=ledoit_wolf_cov,
        cost_bps=5.0,
    )

    comparison_table = {
        "id": "optimizer_comparison",
        "title": "Optimizer comparison (rolling 252d estimation, monthly rebalance, 5bps costs)",
        "columns": [
            "Method",
            "Ann. return (%)",
            "Ann. vol (%)",
            "Sharpe",
            "Max DD (%)",
            "Avg turnover (%)",
        ],
        "rows": [
            [
                name,
                fnum(100 * r.annual_return, 2),
                fnum(100 * r.annual_volatility, 2),
                fnum(r.sharpe, 2),
                fnum(100 * r.max_drawdown, 2),
                fnum(100 * r.avg_turnover, 2),
            ]
            for name, r in results.items()
        ],
    }

    # --- Equity curves (<= 5 series) ----------------------------------------
    equity_methods = ["EqualWeight", "MinVariance", "MaxSharpe", "RiskParity", "HRP"]
    base = downsample(results[equity_methods[0]].equity)
    equity_chart = {
        "id": "equity_curves",
        "title": "Backtest equity curves (growth of $1)",
        "type": "line",
        "x_label": "Date",
        "y_label": "Equity ($)",
        "x": [d.strftime("%Y-%m-%d") for d in base.index],
        "series": [
            {
                "name": name,
                "data": [fnum(v) for v in results[name].equity.loc[base.index]],
            }
            for name in equity_methods
        ],
    }

    # --- Final weight allocations (<= 5 series) ------------------------------
    weight_methods = ["InverseVol", "MinVariance", "MaxSharpe", "RiskParity", "HRP"]
    assets = universe.assets
    weights_chart = {
        "id": "weights",
        "title": "Latest rebalance weights by optimizer",
        "type": "bar",
        "x_label": "Asset",
        "y_label": "Weight (%)",
        "x": assets,
        "series": [
            {
                "name": name,
                "data": [
                    fnum(100 * results[name].weights.iloc[-1][a], 2) for a in assets
                ],
            }
            for name in weight_methods
        ],
    }

    # --- Summary metrics ------------------------------------------------------
    best = max(results.items(), key=lambda kv: kv[1].sharpe)
    summary = [
        {"label": "Universe", "value": "20 assets / 4 sectors"},
        {"label": "Best Sharpe", "value": f"{best[0]} ({best[1].sharpe:.2f})"},
        {"label": "EqualWeight Sharpe", "value": f"{results['EqualWeight'].sharpe:.2f}"},
        {"label": "RiskParity Sharpe", "value": f"{results['RiskParity'].sharpe:.2f}"},
        {"label": "HRP Sharpe", "value": f"{results['HRP'].sharpe:.2f}"},
        {
            "label": "MinVar ann. vol",
            "value": f"{100 * results['MinVariance'].annual_volatility:.1f}",
            "unit": "%",
        },
        {
            "label": "Max Sharpe (frontier)",
            "value": f"{ms_ret / ms_vol:.2f}",
        },
        {"label": "Frontier points", "value": str(len(frontier))},
    ]

    return {
        "schema_version": 1,
        "product": "portfolioengine",
        "title": "PortfolioEngine",
        "tagline": "Asset allocation and portfolio optimization",
        "version": __version__,
        "generated_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "status": "ok",
        "summary_metrics": summary,
        "charts": [frontier_chart, equity_chart, weights_chart],
        "tables": [comparison_table],
        "notes": (
            "Synthetic 20-asset universe (seed 7) with 4-sector block correlation "
            "(0.55 intra / 0.10 inter), ~6 years daily. Backtest: 252d rolling "
            "Ledoit-Wolf estimation, 21d rebalancing, 5bps one-way costs, long-only "
            "with a 15% per-asset cap. Frontier uses full-sample annualized estimates."
        ),
    }


def main() -> None:
    payload = build_payload()
    HUB_DATA.mkdir(parents=True, exist_ok=True)
    out = HUB_DATA / "portfolioengine.json"
    with out.open("w") as fh:
        json.dump(payload, fh, indent=2)
        fh.write("\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
