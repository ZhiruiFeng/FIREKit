"""Synthetic market data generation for SignalML demos and tests.

Generates a multi-symbol daily price/volume panel whose returns contain a
*known, learnable* structure (short-term momentum plus one-day reversal) so
that models trained on the derived features have a real signal to recover.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Coefficients of the injected return structure. Returns at day t are
#   r_t = strength * (MOM_COEF * mean(r_{t-5..t-1}) + REV_COEF * r_{t-1}) + noise
MOM_COEF = 0.35
REV_COEF = -0.15


def make_symbols(n: int) -> list[str]:
    """Create deterministic placeholder tickers SYM000, SYM001, ..."""
    return [f"SYM{i:03d}" for i in range(n)]


def generate_panel(
    n_symbols: int = 50,
    n_days: int = 600,
    seed: int = 7,
    signal_strength: float = 1.0,
    daily_vol: float = 0.02,
    start: str = "2022-01-03",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate a synthetic (close, volume) panel.

    Args:
        n_symbols: number of symbols (columns).
        n_days: number of business days (rows).
        seed: RNG seed; identical seeds give identical panels.
        signal_strength: scales the injected momentum/reversal signal.
            0.0 yields pure noise (useful for null tests).
        daily_vol: stdev of the idiosyncratic daily return noise.
        start: first business day.

    Returns:
        Tuple of wide DataFrames ``(close, volume)`` indexed by business-day
        DatetimeIndex named "date" with symbol columns named "symbol".
    """
    if n_symbols < 1 or n_days < 1:
        raise ValueError("n_symbols and n_days must be positive")

    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=n_days, name="date")
    symbols = make_symbols(n_symbols)

    eps = rng.normal(0.0, daily_vol, size=(n_days, n_symbols))
    rets = np.zeros((n_days, n_symbols))
    for t in range(n_days):
        mom = rets[max(0, t - 5) : t].mean(axis=0) if t >= 1 else np.zeros(n_symbols)
        rev = rets[t - 1] if t >= 1 else np.zeros(n_symbols)
        rets[t] = signal_strength * (MOM_COEF * mom + REV_COEF * rev) + eps[t]

    close = pd.DataFrame(
        100.0 * np.cumprod(1.0 + rets, axis=0),
        index=dates,
        columns=pd.Index(symbols, name="symbol"),
    )

    base = rng.lognormal(mean=13.0, sigma=0.4, size=n_symbols)
    vol_noise = rng.normal(0.0, 0.3, size=(n_days, n_symbols))
    # Volume responds to absolute returns (turnover spikes on big moves).
    volume = pd.DataFrame(
        base * np.exp(vol_noise + 8.0 * np.abs(rets)),
        index=dates,
        columns=pd.Index(symbols, name="symbol"),
    )
    return close, volume
