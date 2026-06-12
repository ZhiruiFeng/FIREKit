"""Shared fixtures for SignalML tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from signalml.data import generate_panel
from signalml.features import build_dataset


@pytest.fixture(scope="session")
def small_panel() -> tuple[pd.DataFrame, pd.DataFrame]:
    """A small but signal-bearing synthetic panel (20 symbols x 220 days)."""
    return generate_panel(n_symbols=20, n_days=220, seed=11)


@pytest.fixture(scope="session")
def small_dataset(small_panel) -> tuple[pd.DataFrame, pd.Series]:
    close, volume = small_panel
    return build_dataset(close, volume, horizon=5)


@pytest.fixture()
def toy_xy() -> tuple[pd.DataFrame, pd.Series]:
    """A tiny deterministic dataset with a perfect linear relationship."""
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2024-01-01", periods=30, name="date")
    symbols = [f"S{i}" for i in range(8)]
    index = pd.MultiIndex.from_product([dates, symbols], names=["date", "symbol"])
    X = pd.DataFrame(
        rng.normal(size=(len(index), 3)), index=index, columns=["f0", "f1", "f2"]
    )
    y = pd.Series(2.0 * X["f0"] - 1.0 * X["f1"], index=index, name="label")
    return X, y
