"""Kelly criterion position sizing.

Implements the classic binary-outcome Kelly fraction, the continuous
(return-moment) approximation, fractional Kelly scaling, and the
multi-asset Kelly portfolio via the inverse covariance matrix.

Conventions
-----------
- All fractions are expressed relative to total capital (1.0 == 100%).
- ``payoff_ratio`` (b) is average win divided by average loss, both positive.
"""

from __future__ import annotations

import numpy as np


def kelly_binary(win_rate: float, payoff_ratio: float) -> float:
    """Full Kelly fraction for a binary bet.

    f* = (p * b - q) / b  where p = win probability, q = 1 - p,
    b = payoff ratio (avg win / avg loss).

    Returns the raw Kelly fraction, which may be negative when the bet has
    negative edge (callers typically floor at zero).
    """
    if not 0.0 <= win_rate <= 1.0:
        raise ValueError(f"win_rate must be in [0, 1], got {win_rate}")
    if payoff_ratio <= 0.0:
        raise ValueError(f"payoff_ratio must be positive, got {payoff_ratio}")
    p = win_rate
    q = 1.0 - p
    return (p * payoff_ratio - q) / payoff_ratio


def kelly_from_moments(mean_return: float, variance: float) -> float:
    """Continuous Kelly approximation from return moments.

    For small returns the growth-optimal fraction is f* = mu / sigma^2.
    ``mean_return`` is the expected excess return per period and ``variance``
    the per-period return variance (same time scale).
    """
    if variance <= 0.0:
        raise ValueError(f"variance must be positive, got {variance}")
    return mean_return / variance


def fractional_kelly(kelly_fraction: float, fraction: float = 0.5) -> float:
    """Scale a full-Kelly fraction by ``fraction`` (e.g. 0.5 = half Kelly)."""
    if fraction < 0.0:
        raise ValueError(f"fraction must be non-negative, got {fraction}")
    return kelly_fraction * fraction


def kelly_multi_asset(
    expected_returns: np.ndarray,
    cov: np.ndarray,
    fraction: float = 1.0,
    max_abs_weight: float | None = None,
) -> np.ndarray:
    """Multi-asset Kelly weights: f = fraction * cov^{-1} mu.

    Parameters
    ----------
    expected_returns:
        Expected excess return per asset (per period).
    cov:
        Per-period return covariance matrix (must be square, symmetric,
        positive definite up to numerical tolerance).
    fraction:
        Fractional Kelly multiplier applied to the raw solution.
    max_abs_weight:
        Optional sanity clip: each weight is clipped to
        ``[-max_abs_weight, +max_abs_weight]`` after scaling.

    Returns
    -------
    np.ndarray of Kelly weights (may exceed 1.0 in aggregate -- leverage).
    """
    mu = np.asarray(expected_returns, dtype=float).ravel()
    sigma = np.asarray(cov, dtype=float)
    if sigma.ndim != 2 or sigma.shape[0] != sigma.shape[1]:
        raise ValueError(f"cov must be a square matrix, got shape {sigma.shape}")
    if sigma.shape[0] != mu.shape[0]:
        raise ValueError(
            f"dimension mismatch: {mu.shape[0]} returns vs {sigma.shape[0]}x{sigma.shape[1]} cov"
        )
    if fraction < 0.0:
        raise ValueError(f"fraction must be non-negative, got {fraction}")

    try:
        raw = np.linalg.solve(sigma, mu)
    except np.linalg.LinAlgError:
        # Singular covariance: fall back to pseudo-inverse.
        raw = np.linalg.pinv(sigma) @ mu

    weights = fraction * raw
    if max_abs_weight is not None:
        if max_abs_weight <= 0.0:
            raise ValueError(f"max_abs_weight must be positive, got {max_abs_weight}")
        weights = np.clip(weights, -max_abs_weight, max_abs_weight)
    return weights
