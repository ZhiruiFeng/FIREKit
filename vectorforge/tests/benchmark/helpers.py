"""Shared helpers for benchmark tests.

Builds synthetic geometric-Brownian-motion (GBM) universes directly as
NumPy arrays via the public ``PortfolioData`` constructor. We deliberately
bypass ``PortfolioData.from_dict`` here: the success criteria (SC-001,
SC-004, SC-008) target the *backtest* and *metrics* paths, not the
per-row Python ingestion loop in ``from_dict``.

All generators use a fixed seed for reproducibility.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from vectorforge.portfolio.data import PortfolioData

# 10 years of daily (business-day) data
TEN_YEARS_DAYS = 2520


def make_portfolio_data(
    n_symbols: int,
    n_days: int,
    seed: int = 42,
) -> PortfolioData:
    """
    Create a synthetic GBM multi-asset universe.

    Prices follow geometric Brownian motion with per-symbol drift and
    volatility drawn from fixed-seed uniform distributions. OHLC fields are
    derived deterministically from the close so the array is fully valid
    (no NaNs, all dates tradeable).

    Args:
        n_symbols: Number of assets in the universe
        n_days: Number of business days
        seed: Random seed (fixed for reproducibility)

    Returns:
        Aligned, fully-valid PortfolioData of shape (n_symbols, n_days, 5)
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2015-01-02", periods=n_days)

    drift = rng.uniform(-0.0002, 0.0006, (n_symbols, 1))
    vol = rng.uniform(0.01, 0.03, (n_symbols, 1))
    returns = rng.normal(drift, vol, (n_symbols, n_days))
    close = 100.0 * np.exp(np.cumsum(returns, axis=1))

    prices = np.empty((n_symbols, n_days, 5), dtype=np.float32)
    prices[:, :, 0] = close * 0.995  # open
    prices[:, :, 1] = close * 1.01  # high
    prices[:, :, 2] = close * 0.99  # low
    prices[:, :, 3] = close  # close
    prices[:, :, 4] = 1e6  # volume

    symbols = [f"SYM{i:04d}" for i in range(n_symbols)]
    return PortfolioData(symbols=symbols, dates=dates, prices=prices)


def make_sector_map(symbols: list[str], n_sectors: int = 10) -> dict[str, str]:
    """Assign symbols round-robin to ``n_sectors`` synthetic sectors."""
    return {s: f"Sector{i % n_sectors:02d}" for i, s in enumerate(symbols)}
