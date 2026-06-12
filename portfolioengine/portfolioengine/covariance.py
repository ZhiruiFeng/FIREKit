"""Covariance estimators: sample, Ledoit-Wolf shrinkage, EWMA."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf


def _to_matrix(returns: pd.DataFrame | np.ndarray) -> np.ndarray:
    arr = returns.to_numpy(dtype=float) if isinstance(returns, pd.DataFrame) else np.asarray(
        returns, dtype=float
    )
    if arr.ndim != 2:
        raise ValueError(f"returns must be 2-D (T x N), got shape {arr.shape}")
    if arr.shape[0] < 2:
        raise ValueError("need at least 2 observations")
    if np.isnan(arr).any():
        raise ValueError("returns contain NaN values")
    return arr


def sample_cov(returns: pd.DataFrame | np.ndarray) -> np.ndarray:
    """Unbiased sample covariance (ddof=1), per-period units."""
    arr = _to_matrix(returns)
    return np.cov(arr, rowvar=False, ddof=1)


def ledoit_wolf_cov(returns: pd.DataFrame | np.ndarray) -> np.ndarray:
    """Ledoit-Wolf shrinkage covariance (scikit-learn implementation)."""
    arr = _to_matrix(returns)
    lw = LedoitWolf(assume_centered=False)
    lw.fit(arr)
    return np.asarray(lw.covariance_)


def ewma_cov(returns: pd.DataFrame | np.ndarray, lam: float = 0.94) -> np.ndarray:
    """Exponentially weighted covariance (RiskMetrics-style).

    Weight on observation k periods back is proportional to ``lam**k``
    (most recent observation gets the largest weight). Returns are demeaned
    with the equally-weighted sample mean.
    """
    if not 0.0 < lam < 1.0:
        raise ValueError(f"lam must be in (0, 1), got {lam}")
    arr = _to_matrix(returns)
    t = arr.shape[0]
    demeaned = arr - arr.mean(axis=0)
    # Oldest -> newest: weight lam^(t-1-i), normalized to sum to 1.
    w = lam ** np.arange(t - 1, -1, -1, dtype=float)
    w /= w.sum()
    return (demeaned * w[:, None]).T @ demeaned


def to_correlation(cov: np.ndarray) -> np.ndarray:
    """Convert a covariance matrix to a correlation matrix."""
    sd = np.sqrt(np.diag(cov))
    if np.any(sd <= 0.0):
        raise ValueError("covariance has non-positive variances")
    corr = cov / np.outer(sd, sd)
    np.fill_diagonal(corr, 1.0)
    return np.clip(corr, -1.0, 1.0)


def annualize_cov(cov: np.ndarray, periods_per_year: int = 252) -> np.ndarray:
    """Scale a per-period covariance matrix to annual units."""
    return np.asarray(cov, dtype=float) * periods_per_year
