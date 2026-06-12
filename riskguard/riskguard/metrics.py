"""Risk metrics: VaR and CVaR (Expected Shortfall), plus rolling versions.

Conventions
-----------
- VaR and CVaR are reported as *positive loss fractions*: VaR 0.02 at 95%
  means "the 5% worst-case one-period loss is at least 2%".
- ``confidence`` is the confidence level (e.g. 0.95), so the tail
  probability is ``1 - confidence``.
- Methods: ``"historical"`` (empirical quantile), ``"gaussian"``
  (parametric normal), ``"cornish_fisher"`` (normal adjusted for sample
  skewness and excess kurtosis).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from scipy import stats

VAR_METHODS = ("historical", "gaussian", "cornish_fisher")


def _as_array(returns: pd.Series | np.ndarray) -> np.ndarray:
    arr = np.asarray(returns, dtype=float).ravel()
    arr = arr[~np.isnan(arr)]
    if arr.size < 2:
        raise ValueError("need at least 2 non-NaN return observations")
    return arr


def _check_confidence(confidence: float) -> None:
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")


def var_historical(returns: pd.Series | np.ndarray, confidence: float = 0.95) -> float:
    """Historical VaR: negative of the (1 - confidence) empirical quantile."""
    _check_confidence(confidence)
    arr = _as_array(returns)
    return float(-np.quantile(arr, 1.0 - confidence))


def cvar_historical(returns: pd.Series | np.ndarray, confidence: float = 0.95) -> float:
    """Historical CVaR / Expected Shortfall: mean loss beyond the VaR quantile."""
    _check_confidence(confidence)
    arr = _as_array(returns)
    threshold = np.quantile(arr, 1.0 - confidence)
    tail = arr[arr <= threshold]
    if tail.size == 0:
        return float(-threshold)
    return float(-tail.mean())


def var_gaussian(returns: pd.Series | np.ndarray, confidence: float = 0.95) -> float:
    """Parametric Gaussian VaR from sample mean and std (ddof=1)."""
    _check_confidence(confidence)
    arr = _as_array(returns)
    mu = arr.mean()
    sigma = arr.std(ddof=1)
    z = stats.norm.ppf(1.0 - confidence)
    return float(-(mu + z * sigma))


def cvar_gaussian(returns: pd.Series | np.ndarray, confidence: float = 0.95) -> float:
    """Parametric Gaussian Expected Shortfall.

    ES = -mu + sigma * phi(z_alpha) / alpha, with alpha = 1 - confidence.
    """
    _check_confidence(confidence)
    arr = _as_array(returns)
    mu = arr.mean()
    sigma = arr.std(ddof=1)
    alpha = 1.0 - confidence
    z = stats.norm.ppf(alpha)
    return float(-mu + sigma * stats.norm.pdf(z) / alpha)


def _cornish_fisher_z(z: float, skew: float, excess_kurtosis: float) -> float:
    """Cornish-Fisher quantile adjustment of a standard-normal quantile."""
    return (
        z
        + (z**2 - 1.0) * skew / 6.0
        + (z**3 - 3.0 * z) * excess_kurtosis / 24.0
        - (2.0 * z**3 - 5.0 * z) * skew**2 / 36.0
    )


def var_cornish_fisher(returns: pd.Series | np.ndarray, confidence: float = 0.95) -> float:
    """Cornish-Fisher (modified) VaR adjusting for skewness and kurtosis."""
    _check_confidence(confidence)
    arr = _as_array(returns)
    mu = arr.mean()
    sigma = arr.std(ddof=1)
    skew = float(stats.skew(arr))
    kurt = float(stats.kurtosis(arr))  # excess kurtosis
    z = float(stats.norm.ppf(1.0 - confidence))
    z_cf = _cornish_fisher_z(z, skew, kurt)
    return float(-(mu + z_cf * sigma))


def cvar_cornish_fisher(
    returns: pd.Series | np.ndarray,
    confidence: float = 0.95,
    n_grid: int = 200,
) -> float:
    """Cornish-Fisher Expected Shortfall.

    Uses ES_c = (1 / (1 - c)) * integral_c^1 VaR_u du, evaluated with a
    midpoint rule over the tail confidences.
    """
    _check_confidence(confidence)
    arr = _as_array(returns)
    mu = arr.mean()
    sigma = arr.std(ddof=1)
    skew = float(stats.skew(arr))
    kurt = float(stats.kurtosis(arr))
    grid = confidence + (1.0 - confidence) * (np.arange(n_grid) + 0.5) / n_grid
    z = stats.norm.ppf(1.0 - grid)
    z_cf = _cornish_fisher_z(z, skew, kurt)
    return float(np.mean(-(mu + z_cf * sigma)))


_VAR_FUNCS: dict[str, Callable[..., float]] = {
    "historical": var_historical,
    "gaussian": var_gaussian,
    "cornish_fisher": var_cornish_fisher,
}
_CVAR_FUNCS: dict[str, Callable[..., float]] = {
    "historical": cvar_historical,
    "gaussian": cvar_gaussian,
    "cornish_fisher": cvar_cornish_fisher,
}


def value_at_risk(
    returns: pd.Series | np.ndarray,
    confidence: float = 0.95,
    method: str = "historical",
) -> float:
    """Dispatch VaR by method name (see :data:`VAR_METHODS`)."""
    if method not in _VAR_FUNCS:
        raise ValueError(f"unknown method {method!r}, expected one of {VAR_METHODS}")
    return _VAR_FUNCS[method](returns, confidence)


def expected_shortfall(
    returns: pd.Series | np.ndarray,
    confidence: float = 0.95,
    method: str = "historical",
) -> float:
    """Dispatch CVaR / Expected Shortfall by method name."""
    if method not in _CVAR_FUNCS:
        raise ValueError(f"unknown method {method!r}, expected one of {VAR_METHODS}")
    return _CVAR_FUNCS[method](returns, confidence)


def rolling_var(
    returns: pd.Series,
    window: int = 252,
    confidence: float = 0.95,
    method: str = "historical",
) -> pd.Series:
    """Rolling-window VaR series (NaN until the window fills)."""
    if window < 10:
        raise ValueError(f"window must be >= 10, got {window}")
    func = _VAR_FUNCS.get(method)
    if func is None:
        raise ValueError(f"unknown method {method!r}, expected one of {VAR_METHODS}")
    return returns.rolling(window).apply(lambda x: func(x, confidence), raw=True)


def rolling_cvar(
    returns: pd.Series,
    window: int = 252,
    confidence: float = 0.95,
    method: str = "historical",
) -> pd.Series:
    """Rolling-window CVaR series (NaN until the window fills)."""
    if window < 10:
        raise ValueError(f"window must be >= 10, got {window}")
    func = _CVAR_FUNCS.get(method)
    if func is None:
        raise ValueError(f"unknown method {method!r}, expected one of {VAR_METHODS}")
    return returns.rolling(window).apply(lambda x: func(x, confidence), raw=True)


def annualized_volatility(returns: pd.Series | np.ndarray, periods_per_year: int = 252) -> float:
    """Annualized volatility from per-period returns (ddof=1)."""
    arr = _as_array(returns)
    return float(arr.std(ddof=1) * np.sqrt(periods_per_year))


def sharpe_ratio(
    returns: pd.Series | np.ndarray,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """Annualized Sharpe ratio; ``risk_free_rate`` is annualized."""
    arr = _as_array(returns)
    excess = arr - risk_free_rate / periods_per_year
    sd = excess.std(ddof=1)
    if sd == 0.0:
        return 0.0
    return float(excess.mean() / sd * np.sqrt(periods_per_year))
