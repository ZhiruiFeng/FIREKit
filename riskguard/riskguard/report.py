"""RiskReport: one-call risk assessment for a returns + weights input."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from riskguard.drawdown import equity_from_returns, max_drawdown
from riskguard.kelly import kelly_multi_asset
from riskguard.limits import ExposureLimits, LimitEngine, LimitResult
from riskguard.metrics import (
    VAR_METHODS,
    annualized_volatility,
    expected_shortfall,
    sharpe_ratio,
    value_at_risk,
)
from riskguard.voltarget import realized_vol


@dataclass(frozen=True)
class RiskReport:
    """Aggregated risk assessment for a portfolio."""

    annual_return: float
    annual_volatility: float
    sharpe: float
    max_drawdown: float
    var: dict[str, dict[str, float]]  # method -> {"0.95": ..., "0.99": ...}
    cvar: dict[str, dict[str, float]]
    current_realized_vol: float
    vol_target_exposure: float | None
    kelly_weights: dict[str, float]
    limit_result: LimitResult
    portfolio_returns: pd.Series = field(repr=False)

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly dictionary (drops the raw return series)."""
        return {
            "annual_return": self.annual_return,
            "annual_volatility": self.annual_volatility,
            "sharpe": self.sharpe,
            "max_drawdown": self.max_drawdown,
            "var": self.var,
            "cvar": self.cvar,
            "current_realized_vol": self.current_realized_vol,
            "vol_target_exposure": self.vol_target_exposure,
            "kelly_weights": self.kelly_weights,
            "limit_violations": [
                {
                    "rule": v.rule,
                    "scope": v.scope,
                    "value": v.value,
                    "limit": v.limit,
                    "action": v.action,
                }
                for v in self.limit_result.violations
            ],
            "adjusted_weights": self.limit_result.weights,
        }


def build_risk_report(
    asset_returns: pd.DataFrame,
    weights: Mapping[str, float],
    *,
    confidences: tuple[float, ...] = (0.95, 0.99),
    var_methods: tuple[str, ...] = VAR_METHODS,
    vol_window: int = 20,
    target_vol: float | None = None,
    kelly_fraction: float = 0.5,
    kelly_cap: float = 1.0,
    limits: ExposureLimits | None = None,
    sectors: Mapping[str, str] | None = None,
    periods_per_year: int = 252,
) -> RiskReport:
    """Assemble a :class:`RiskReport` from asset returns and proposed weights.

    Parameters
    ----------
    asset_returns:
        Wide DataFrame of per-period simple returns (columns = assets).
    weights:
        Proposed portfolio weights keyed by column name. Assets present in
        ``asset_returns`` but missing from ``weights`` get weight 0.
    target_vol:
        Optional annualized vol target; when given, the report includes the
        exposure multiplier implied by current realized vol.
    kelly_fraction / kelly_cap:
        Fractional multi-asset Kelly suggestion computed from sample moments,
        with per-asset |weight| clipped at ``kelly_cap``.
    """
    missing = [a for a in weights if a not in asset_returns.columns]
    if missing:
        raise ValueError(f"weights refer to unknown assets: {missing}")

    cols = list(asset_returns.columns)
    w = np.array([float(weights.get(c, 0.0)) for c in cols])
    rets = asset_returns[cols].to_numpy(dtype=float)
    if np.isnan(rets).any():
        raise ValueError("asset_returns contains NaN values")
    port = pd.Series(rets @ w, index=asset_returns.index, name="portfolio")

    var_table: dict[str, dict[str, float]] = {}
    cvar_table: dict[str, dict[str, float]] = {}
    for method in var_methods:
        var_table[method] = {
            f"{c:.2f}": value_at_risk(port, c, method) for c in confidences
        }
        cvar_table[method] = {
            f"{c:.2f}": expected_shortfall(port, c, method) for c in confidences
        }

    rolling = realized_vol(port, vol_window, periods_per_year).dropna()
    current_vol = float(rolling.iloc[-1]) if len(rolling) else float("nan")
    exposure = None
    if target_vol is not None and current_vol > 0:
        exposure = float(target_vol / current_vol)

    mu = rets.mean(axis=0)
    cov = np.cov(rets, rowvar=False, ddof=1)
    kelly_w = kelly_multi_asset(mu, cov, fraction=kelly_fraction, max_abs_weight=kelly_cap)
    kelly_weights = {c: float(k) for c, k in zip(cols, kelly_w, strict=True)}

    engine = LimitEngine(limits or ExposureLimits())
    limit_result = engine.apply({c: float(weights.get(c, 0.0)) for c in cols}, sectors=sectors)

    ann_ret = float(port.mean() * periods_per_year)
    return RiskReport(
        annual_return=ann_ret,
        annual_volatility=annualized_volatility(port, periods_per_year),
        sharpe=sharpe_ratio(port, periods_per_year=periods_per_year),
        max_drawdown=max_drawdown(equity_from_returns(port)),
        var=var_table,
        cvar=cvar_table,
        current_realized_vol=current_vol,
        vol_target_exposure=exposure,
        kelly_weights=kelly_weights,
        limit_result=limit_result,
        portfolio_returns=port,
    )
