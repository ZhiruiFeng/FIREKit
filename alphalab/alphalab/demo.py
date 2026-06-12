"""AlphaLab demo pipeline.

Run with ``cd alphalab && python3 -m alphalab.demo``. Writes
``hub/data/alphalab.json`` following ``hub/SCHEMA.md``: evaluates the full
built-in factor zoo on a 50-symbol, 3-year synthetic panel and reports IC
rankings, the best factor's IC time series, quantile portfolio returns and
factor-correlation insights.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from alphalab import __version__
from alphalab.panel import make_synthetic_panel
from alphalab.zoo import FactorZoo

HUB_DATA_DIR = Path(__file__).resolve().parents[2] / "hub" / "data"

SEED = 20260612
N_SYMBOLS = 50
N_DAYS = 756  # ~3 years of business days
HORIZON = 5


def _round(value: float, digits: int = 4) -> float | None:
    """JSON-safe float (NaN -> None, no -0.0)."""
    import math

    value = float(value)
    if not math.isfinite(value):
        return None
    out = round(value, digits)
    return 0.0 if out == 0 else out


def run_pipeline() -> dict:
    panel = make_synthetic_panel(n_symbols=N_SYMBOLS, n_days=N_DAYS, seed=SEED)
    zoo = FactorZoo.default()
    report = zoo.evaluate(panel, horizon=HORIZON, n_quantiles=5)

    best = report.best
    top10 = report.evaluations[:10]
    significant = sum(
        1 for ev in report.evaluations if abs(ev.rank_ic.t_stat) > 2.0
    )

    # Chart 1: best factor's rank-IC time series, smoothed and downsampled.
    smoothed = best.rank_ic_timeseries.rolling(21).mean().dropna()
    if len(smoothed) > 500:
        step = -(-len(smoothed) // 500)  # ceil division
        smoothed = smoothed.iloc[::step]

    # Chart 2: quantile portfolio mean forward returns for the best factor (bps).
    qmeans = best.quantile_mean_returns

    rows = [
        [
            ev.name,
            ev.category,
            _round(ev.ic.mean),
            _round(ev.rank_ic.mean),
            _round(ev.rank_ic.ir, 3),
            _round(ev.rank_ic.t_stat, 2),
            _round(ev.spread_mean * 1e4, 1),  # bps per holding period
            _round(ev.turnover, 3),
        ]
        for ev in top10
    ]

    corr_rows = [
        [a, b, _round(rho, 3)] for a, b, rho in report.top_correlated_pairs(8)
    ]
    abs_corr = report.correlation.abs()
    n = len(abs_corr)
    mean_abs_corr = (abs_corr.sum().sum() - n) / (n * (n - 1)) if n > 1 else 0.0

    return {
        "schema_version": 1,
        "product": "alphalab",
        "title": "AlphaLab",
        "tagline": "Factor mining workbench: zoo evaluation, IC analytics, quantile spreads",
        "version": __version__,
        "generated_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "status": "ok",
        "summary_metrics": [
            {"label": "Factors evaluated", "value": str(len(report.evaluations))},
            {"label": "Universe", "value": f"{N_SYMBOLS} symbols x {N_DAYS} days"},
            {"label": "Forward horizon", "value": f"{HORIZON}", "unit": "days"},
            {"label": "Best factor", "value": best.name},
            {"label": "Best rank IC", "value": f"{best.rank_ic.mean:.4f}"},
            {"label": "Best rank-IC IR", "value": f"{best.rank_ic.ir:.3f}"},
            {"label": "Significant factors (|t|>2)", "value": str(significant)},
            {"label": "Mean |factor correlation|", "value": f"{mean_abs_corr:.3f}"},
        ],
        "charts": [
            {
                "id": "best_factor_ic",
                "title": f"Rank IC of {best.name} (21-day rolling mean, {HORIZON}d horizon)",
                "type": "line",
                "x_label": "Date",
                "y_label": "Rank IC",
                "x": [d.strftime("%Y-%m-%d") for d in smoothed.index],
                "series": [
                    {"name": "Rank IC (21d avg)", "data": [_round(v) for v in smoothed]}
                ],
            },
            {
                "id": "quantile_returns",
                "title": f"Mean {HORIZON}-day forward return by {best.name} quantile",
                "type": "bar",
                "x_label": "Factor quantile (Q1 = lowest)",
                "y_label": "Mean forward return (bps)",
                "x": list(qmeans.index),
                "series": [
                    {
                        "name": "Forward return",
                        "data": [_round(v * 1e4, 2) for v in qmeans],
                    }
                ],
            },
        ],
        "tables": [
            {
                "id": "top_factors",
                "title": f"Top 10 factors by |rank-IC IR| ({HORIZON}-day horizon)",
                "columns": [
                    "Factor", "Category", "IC", "Rank IC", "Rank-IC IR",
                    "t-stat", "Q5-Q1 spread (bps)", "Turnover",
                ],
                "rows": rows,
            },
            {
                "id": "correlated_pairs",
                "title": "Most correlated factor pairs (redundancy candidates)",
                "columns": ["Factor A", "Factor B", "Rank corr"],
                "rows": corr_rows,
            },
        ],
        "notes": (
            f"Synthetic {N_SYMBOLS}-symbol panel (seed {SEED}) with embedded momentum, "
            "reversal and market-beta structure. The full 20-factor zoo (momentum, reversal, "
            "volatility, volume/liquidity, technical, beta and 5 Alpha101 formulaic alphas) is "
            f"evaluated against {HORIZON}-day forward returns: per-date IC and rank IC, "
            "IC IR/t-stats, Q1-Q5 quantile portfolios, top-quintile turnover and a "
            "cross-factor rank-correlation matrix."
        ),
    }


def main() -> Path:
    payload = run_pipeline()
    HUB_DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = HUB_DATA_DIR / "alphalab.json"
    with out_path.open("w") as fh:
        json.dump(payload, fh, indent=2)
        fh.write("\n")
    print(f"wrote {out_path}")
    for metric in payload["summary_metrics"]:
        unit = metric.get("unit", "")
        print(f"  {metric['label']}: {metric['value']}{unit}")
    return out_path


if __name__ == "__main__":
    main()
