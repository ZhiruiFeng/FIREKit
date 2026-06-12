"""Tests for synthetic panel generation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from signalml.data import generate_panel, make_symbols


def test_panel_shapes_and_index():
    close, volume = generate_panel(n_symbols=7, n_days=50, seed=1)
    assert close.shape == (50, 7)
    assert volume.shape == (50, 7)
    assert isinstance(close.index, pd.DatetimeIndex)
    assert close.index.name == "date"
    assert close.columns.name == "symbol"
    assert list(close.columns) == make_symbols(7)
    assert close.index.equals(volume.index)


def test_panel_deterministic_for_same_seed():
    a, va = generate_panel(n_symbols=5, n_days=40, seed=42)
    b, vb = generate_panel(n_symbols=5, n_days=40, seed=42)
    pd.testing.assert_frame_equal(a, b)
    pd.testing.assert_frame_equal(va, vb)


def test_panel_differs_for_different_seed():
    a, _ = generate_panel(n_symbols=5, n_days=40, seed=1)
    b, _ = generate_panel(n_symbols=5, n_days=40, seed=2)
    assert not np.allclose(a.to_numpy(), b.to_numpy())


def test_prices_and_volume_positive():
    close, volume = generate_panel(n_symbols=10, n_days=300, seed=3)
    assert (close.to_numpy() > 0).all()
    assert (volume.to_numpy() > 0).all()


def test_injected_signal_is_learnable():
    """Trailing 5-day mean return should positively predict the next return."""
    close, _ = generate_panel(n_symbols=30, n_days=400, seed=5, signal_strength=1.0)
    rets = close.pct_change()
    mom = rets.rolling(5).mean()
    fwd = rets.shift(-1)
    corr = mom.stack().corr(fwd.stack())
    assert corr > 0.02


def test_zero_strength_has_no_signal():
    close, _ = generate_panel(n_symbols=30, n_days=400, seed=5, signal_strength=0.0)
    rets = close.pct_change()
    mom = rets.rolling(5).mean()
    fwd = rets.shift(-1)
    corr = mom.stack().corr(fwd.stack())
    assert abs(corr) < 0.03


def test_invalid_sizes_raise():
    with pytest.raises(ValueError):
        generate_panel(n_symbols=0, n_days=10)
    with pytest.raises(ValueError):
        generate_panel(n_symbols=5, n_days=0)
