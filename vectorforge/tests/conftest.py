"""
Pytest configuration and fixtures for VectorForge tests.
"""

import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timedelta


@pytest.fixture
def sample_ohlcv_data():
    """Generate sample OHLCV data for testing."""
    np.random.seed(42)
    n_days = 252  # 1 year of trading days

    # Generate random walk price data
    returns = np.random.normal(0.0005, 0.02, n_days)
    prices = 100 * np.cumprod(1 + returns)

    # Create dates (skip weekends)
    dates = pd.date_range(start="2023-01-01", periods=n_days, freq="B")

    # Generate OHLCV
    data = pd.DataFrame(index=dates)
    data["close"] = prices
    data["open"] = data["close"].shift(1).fillna(100)
    data["high"] = np.maximum(data["open"], data["close"]) * (1 + np.random.uniform(0, 0.02, n_days))
    data["low"] = np.minimum(data["open"], data["close"]) * (1 - np.random.uniform(0, 0.02, n_days))
    data["volume"] = np.random.uniform(1e6, 5e6, n_days).astype(int)

    data.attrs["symbol"] = "TEST"
    return data


@pytest.fixture
def small_ohlcv_data():
    """Generate small OHLCV data for quick tests."""
    dates = pd.date_range(start="2023-01-01", periods=50, freq="B")

    data = pd.DataFrame(index=dates)
    data["open"] = np.linspace(100, 110, 50)
    data["close"] = np.linspace(100, 110, 50) + np.random.uniform(-1, 1, 50)
    data["high"] = np.maximum(data["open"], data["close"]) * 1.01
    data["low"] = np.minimum(data["open"], data["close"]) * 0.99
    data["volume"] = np.full(50, 1_000_000)

    data.attrs["symbol"] = "TEST"
    return data


@pytest.fixture
def trending_data():
    """Generate strongly trending data for testing."""
    n_days = 100
    dates = pd.date_range(start="2023-01-01", periods=n_days, freq="B")

    # Strong uptrend
    prices = 100 * (1.001 ** np.arange(n_days))

    data = pd.DataFrame(index=dates)
    data["close"] = prices
    data["open"] = prices * 0.999
    data["high"] = prices * 1.005
    data["low"] = prices * 0.995
    data["volume"] = np.full(n_days, 1_000_000)

    data.attrs["symbol"] = "TREND"
    return data


@pytest.fixture
def config():
    """Create default VectorForge configuration."""
    from vectorforge.config import VectorForgeConfig
    return VectorForgeConfig.default()


@pytest.fixture
def vectorized_backtester(config):
    """Create VectorizedBacktester instance."""
    from vectorforge import VectorizedBacktester
    return VectorizedBacktester(config)


@pytest.fixture
def event_driven_backtester(config):
    """Create EventDrivenBacktester instance."""
    from vectorforge import EventDrivenBacktester
    return EventDrivenBacktester(config)


@pytest.fixture
def momentum_strategy():
    """Create MomentumStrategy instance."""
    from vectorforge.strategy.base import MomentumStrategy
    return MomentumStrategy(lookback=20)


@pytest.fixture
def ma_crossover_strategy():
    """Create MovingAverageCrossover instance."""
    from vectorforge.strategy.base import MovingAverageCrossover
    return MovingAverageCrossover(fast_period=10, slow_period=30)
