"""RiskGuard: position sizing and risk management toolkit.

Public API:

- Kelly sizing: :func:`kelly_binary`, :func:`kelly_from_moments`,
  :func:`fractional_kelly`, :func:`kelly_multi_asset`
- Volatility targeting: :class:`VolatilityTargeter`, :func:`realized_vol`
- Drawdown protection: :class:`DrawdownCircuitBreaker`, :func:`drawdown_curve`,
  :func:`max_drawdown`, :func:`equity_from_returns`
- Risk metrics: :func:`value_at_risk`, :func:`expected_shortfall` and the
  per-method functions, plus rolling variants
- Exposure limits: :class:`ExposureLimits`, :class:`LimitEngine`
- :func:`build_risk_report` for a one-call :class:`RiskReport`
"""

from riskguard.drawdown import (
    DrawdownCircuitBreaker,
    DrawdownProtectionResult,
    drawdown_curve,
    equity_from_returns,
    max_drawdown,
)
from riskguard.kelly import (
    fractional_kelly,
    kelly_binary,
    kelly_from_moments,
    kelly_multi_asset,
)
from riskguard.limits import ExposureLimits, LimitEngine, LimitResult, Violation
from riskguard.metrics import (
    VAR_METHODS,
    annualized_volatility,
    cvar_cornish_fisher,
    cvar_gaussian,
    cvar_historical,
    expected_shortfall,
    rolling_cvar,
    rolling_var,
    sharpe_ratio,
    value_at_risk,
    var_cornish_fisher,
    var_gaussian,
    var_historical,
)
from riskguard.report import RiskReport, build_risk_report
from riskguard.voltarget import VolatilityTargeter, VolTargetResult, realized_vol

__version__ = "0.1.0"

__all__ = [
    "VAR_METHODS",
    "DrawdownCircuitBreaker",
    "DrawdownProtectionResult",
    "ExposureLimits",
    "LimitEngine",
    "LimitResult",
    "RiskReport",
    "Violation",
    "VolTargetResult",
    "VolatilityTargeter",
    "__version__",
    "annualized_volatility",
    "build_risk_report",
    "cvar_cornish_fisher",
    "cvar_gaussian",
    "cvar_historical",
    "drawdown_curve",
    "equity_from_returns",
    "expected_shortfall",
    "fractional_kelly",
    "kelly_binary",
    "kelly_from_moments",
    "kelly_multi_asset",
    "max_drawdown",
    "realized_vol",
    "rolling_cvar",
    "rolling_var",
    "sharpe_ratio",
    "value_at_risk",
    "var_cornish_fisher",
    "var_gaussian",
    "var_historical",
]
