"""Synthetic market generator with learnable regime-switching structure."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MOMENTUM = 0
MEAN_REVERSION = 1


@dataclass
class RegimeSeries:
    """A synthetic price path plus the hidden regime that generated it."""

    prices: np.ndarray  # length n + 1
    returns: np.ndarray  # length n, returns[t] = prices[t+1]/prices[t] - 1
    regimes: np.ndarray  # length n, 0 = momentum, 1 = mean-reversion


def regime_switching_series(
    n: int = 3000,
    seed: int = 42,
    phi_momentum: float = 0.65,
    phi_mean_reversion: float = -0.65,
    switch_prob: float = 0.005,
    sigma: float = 0.008,
    start_price: float = 100.0,
) -> RegimeSeries:
    """Generate an AR(1) return series whose autocorrelation flips by regime.

    In the momentum regime returns follow ``r_t = +phi * r_{t-1} + eps``; in
    the mean-reversion regime ``r_t = -phi * r_{t-1} + eps``. The hidden
    regime is persistent (it switches with probability ``switch_prob`` per
    step), so the *sign agreement of recent consecutive returns* reveals the
    current regime and the series carries genuinely learnable structure:
    follow the last return's sign in momentum, fade it in mean reversion.
    """
    if n < 2:
        raise ValueError("n must be >= 2")
    rng = np.random.default_rng(seed)
    eps = rng.normal(0.0, sigma, size=n)
    switches = rng.uniform(size=n) < switch_prob

    regimes = np.empty(n, dtype=np.int64)
    regime = MOMENTUM
    returns = np.empty(n)
    prev_r = 0.0
    for t in range(n):
        if switches[t]:
            regime = 1 - regime
        regimes[t] = regime
        phi = phi_momentum if regime == MOMENTUM else phi_mean_reversion
        r = phi * prev_r + eps[t]
        returns[t] = r
        prev_r = r

    prices = start_price * np.exp(np.concatenate([[0.0], np.cumsum(np.log1p(returns))]))
    return RegimeSeries(prices=prices, returns=returns, regimes=regimes)
