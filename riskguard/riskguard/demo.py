"""RiskGuard demo: writes hub/data/riskguard.json (FIREKit hub schema v1).

Run with: cd riskguard && python3 -m riskguard.demo
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from riskguard import (
    DrawdownCircuitBreaker,
    ExposureLimits,
    LimitEngine,
    VolatilityTargeter,
    __version__,
    annualized_volatility,
    equity_from_returns,
    expected_shortfall,
    fractional_kelly,
    kelly_binary,
    kelly_from_moments,
    max_drawdown,
    sharpe_ratio,
    value_at_risk,
)

HUB_DATA = Path(__file__).resolve().parents[2] / "hub" / "data"


def make_strategy_returns(seed: int = 42) -> pd.Series:
    """Synthetic daily strategy returns with a built-in crash and recovery."""
    rng = np.random.default_rng(seed)
    n = 1260  # ~5 trading years
    dates = pd.bdate_range("2021-01-04", periods=n)

    mu = np.full(n, 0.0009)
    vol = np.full(n, 0.010)
    # Crash regime: sharp losses with elevated volatility.
    crash = slice(580, 620)
    mu[crash] = -0.0075
    vol[crash] = 0.028
    # Choppy aftermath, then grind back up.
    after = slice(620, 745)
    mu[after] = 0.0014
    vol[after] = 0.018

    returns = mu + vol * rng.standard_normal(n)
    return pd.Series(returns, index=dates, name="strategy")


def downsample(series: pd.Series, max_points: int = 500) -> pd.Series:
    if len(series) <= max_points:
        return series
    step = int(np.ceil(len(series) / max_points))
    sampled = series.iloc[::step]
    if sampled.index[-1] != series.index[-1]:
        sampled = pd.concat([sampled, series.iloc[[-1]]])
    return sampled


def fnum(x: float, digits: int = 4) -> float | None:
    """JSON-safe rounded float (NaN/inf -> null)."""
    x = float(x)
    if not np.isfinite(x):
        return None
    return round(x, digits)


def build_payload() -> dict:
    returns = make_strategy_returns()

    # --- Protection pipeline: vol targeting, then circuit breaker ----------
    targeter = VolatilityTargeter(target_vol=0.10, window=20, max_leverage=2.0)
    vt = targeter.apply(returns)
    breaker = DrawdownCircuitBreaker(max_drawdown=0.12, reentry_drawdown=0.04)
    protected = breaker.apply(vt.scaled_returns)

    eq_raw = equity_from_returns(returns, 100_000.0)
    eq_vt = equity_from_returns(vt.scaled_returns, 100_000.0)
    eq_prot = 100_000.0 * protected.protected_equity

    # --- Charts -------------------------------------------------------------
    s_raw = downsample(eq_raw)
    s_vt = eq_vt.loc[s_raw.index]
    s_prot = eq_prot.loc[s_raw.index]
    equity_chart = {
        "id": "equity_protection",
        "title": "Equity: raw vs vol-targeted vs vol-target + circuit breaker",
        "type": "line",
        "x_label": "Date",
        "y_label": "Equity ($)",
        "x": [d.strftime("%Y-%m-%d") for d in s_raw.index],
        "series": [
            {"name": "Raw strategy", "data": [fnum(v, 2) for v in s_raw]},
            {"name": "Vol-targeted (10%)", "data": [fnum(v, 2) for v in s_vt]},
            {"name": "Vol-target + breaker", "data": [fnum(v, 2) for v in s_prot]},
        ],
    }

    exp_combined = (vt.exposure * protected.exposure).clip(lower=0.0)
    s_exp = downsample(exp_combined)
    exposure_chart = {
        "id": "exposure",
        "title": "Effective exposure multiplier (vol target x circuit breaker)",
        "type": "line",
        "x_label": "Date",
        "y_label": "Exposure (x)",
        "x": [d.strftime("%Y-%m-%d") for d in s_exp.index],
        "series": [
            {"name": "Exposure", "data": [fnum(v) for v in s_exp]},
        ],
    }

    # --- VaR / CVaR table ----------------------------------------------------
    method_labels = {
        "historical": "Historical",
        "gaussian": "Gaussian",
        "cornish_fisher": "Cornish-Fisher",
    }
    var_rows = []
    for method, label in method_labels.items():
        for conf in (0.95, 0.99):
            var_rows.append(
                [
                    label,
                    f"{conf:.0%}",
                    fnum(100 * value_at_risk(returns, conf, method), 3),
                    fnum(100 * expected_shortfall(returns, conf, method), 3),
                ]
            )
    var_table = {
        "id": "var_cvar",
        "title": "Daily VaR / CVaR of raw strategy returns",
        "columns": ["Method", "Confidence", "VaR (%)", "CVaR (%)"],
        "rows": var_rows,
    }

    # --- Kelly sizing table ----------------------------------------------------
    kelly_rows = []
    for p, b in ((0.55, 1.5), (0.60, 2.0), (0.45, 3.0)):
        full = kelly_binary(p, b)
        kelly_rows.append(
            [
                f"Win rate {p:.0%}, payoff {b:.1f}x",
                fnum(full),
                fnum(fractional_kelly(full, 0.5)),
                fnum(fractional_kelly(full, 0.25)),
            ]
        )
    mu_d = float(returns.mean())
    var_d = float(returns.var(ddof=1))
    full_m = kelly_from_moments(mu_d, var_d)
    kelly_rows.append(
        [
            "Strategy moments (mu/sigma^2, daily)",
            fnum(full_m),
            fnum(fractional_kelly(full_m, 0.5)),
            fnum(fractional_kelly(full_m, 0.25)),
        ]
    )
    kelly_table = {
        "id": "kelly",
        "title": "Kelly position sizing (fraction of capital)",
        "columns": ["Input", "Full Kelly", "Half Kelly", "Quarter Kelly"],
        "rows": kelly_rows,
    }

    # --- Exposure limits example ----------------------------------------------
    proposed = {
        "AAPL": 0.18,
        "MSFT": 0.15,
        "NVDA": 0.22,
        "XOM": 0.08,
        "CVX": 0.06,
        "JPM": 0.12,
        "GS": -0.30,
    }
    sectors = {
        "AAPL": "tech",
        "MSFT": "tech",
        "NVDA": "tech",
        "XOM": "energy",
        "CVX": "energy",
        "JPM": "financials",
        "GS": "financials",
    }
    engine = LimitEngine(
        ExposureLimits(max_position=0.15, max_gross=1.0, max_net=0.60, max_sector=0.35)
    )
    limit_result = engine.apply(proposed, sectors=sectors)
    violations_table = {
        "id": "limit_violations",
        "title": "Exposure limit engine: proposed book vs limits",
        "columns": ["Rule", "Scope", "Exposure", "Limit", "Action"],
        "rows": [
            [v.rule, v.scope, fnum(v.value), fnum(v.limit), v.action]
            for v in limit_result.violations
        ],
    }

    # --- Summary metrics ----------------------------------------------------
    summary = [
        {"label": "Max DD (raw)", "value": f"{100 * max_drawdown(eq_raw):.1f}", "unit": "%"},
        {
            "label": "Max DD (protected)",
            "value": f"{100 * max_drawdown(eq_prot):.1f}",
            "unit": "%",
        },
        {
            "label": "Ann. vol raw",
            "value": f"{100 * annualized_volatility(returns):.1f}",
            "unit": "%",
        },
        {
            "label": "Ann. vol protected",
            "value": f"{100 * annualized_volatility(protected.filtered_returns):.1f}",
            "unit": "%",
        },
        {"label": "Sharpe raw", "value": f"{sharpe_ratio(returns):.2f}"},
        {
            "label": "Sharpe protected",
            "value": f"{sharpe_ratio(protected.filtered_returns):.2f}",
        },
        {
            "label": "VaR 95% (hist.)",
            "value": f"{100 * value_at_risk(returns, 0.95, 'historical'):.2f}",
            "unit": "%",
        },
        {"label": "Breaker triggers", "value": str(protected.n_triggers)},
    ]

    return {
        "schema_version": 1,
        "product": "riskguard",
        "title": "RiskGuard",
        "tagline": "Position sizing, drawdown protection and tail-risk metrics",
        "version": __version__,
        "generated_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "status": "ok",
        "summary_metrics": summary,
        "charts": [equity_chart, exposure_chart],
        "tables": [var_table, kelly_table, violations_table],
        "notes": (
            "Synthetic daily strategy (seed 42) with an engineered crash regime. "
            "Protection stack: 10% vol target (20d window, lagged) followed by a "
            "12% max-drawdown circuit breaker re-entering at 4% drawdown. "
            "VaR/CVaR reported as positive daily loss percentages."
        ),
    }


def main() -> None:
    payload = build_payload()
    HUB_DATA.mkdir(parents=True, exist_ok=True)
    out = HUB_DATA / "riskguard.json"
    with out.open("w") as fh:
        json.dump(payload, fh, indent=2)
        fh.write("\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
