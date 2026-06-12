"""Tests for the feature pipeline and label construction."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from signalml.features import (
    build_dataset,
    build_features,
    build_labels,
    compute_rsi,
)


def test_feature_index_structure(small_panel):
    close, volume = small_panel
    X = build_features(close, volume)
    assert isinstance(X.index, pd.MultiIndex)
    assert X.index.names == ["date", "symbol"]
    assert not X.isna().any().any()
    expected = {
        "ret_1", "ret_5", "ret_10", "ret_21", "vol_21", "rsi_14",
        "ma_gap_10", "ma_gap_21", "volume_z_21", "volume_ratio_5_21",
    }
    assert set(X.columns) == expected


def test_features_without_volume(small_panel):
    close, _ = small_panel
    X = build_features(close, volume=None)
    assert "volume_z_21" not in X.columns
    assert "volume_ratio_5_21" not in X.columns


def test_no_lookahead_in_features(small_panel):
    """Features at date t must be identical if all future data is removed."""
    close, volume = small_panel
    full = build_features(close, volume, normalize=False)
    cutoff = close.index[150]
    trunc = build_features(close.loc[:cutoff], volume.loc[:cutoff], normalize=False)
    common = trunc.index
    pd.testing.assert_frame_equal(full.loc[common], trunc)


def test_label_is_forward_return_exact(small_panel):
    close, _ = small_panel
    h = 5
    labels = build_labels(close, horizon=h)
    t = close.index[100]
    sym = close.columns[3]
    expected = close.loc[close.index[105], sym] / close.loc[t, sym] - 1.0
    assert labels.loc[(t, sym)] == pytest.approx(expected, rel=1e-12)


def test_labels_drop_last_horizon_dates(small_panel):
    close, _ = small_panel
    labels = build_labels(close, horizon=5)
    label_dates = labels.index.get_level_values("date").unique()
    assert close.index[-5] not in label_dates
    assert close.index[-6] in label_dates


def test_label_invalid_horizon_raises(small_panel):
    close, _ = small_panel
    with pytest.raises(ValueError):
        build_labels(close, horizon=0)


def test_normalization_zero_mean_unit_std(small_panel):
    close, volume = small_panel
    X = build_features(close, volume, normalize=True)
    by_date = X.groupby(level="date")
    means = by_date.mean()
    stds = by_date.std()
    assert np.allclose(means.to_numpy(), 0.0, atol=1e-8)
    assert np.allclose(stds.to_numpy(), 1.0, atol=1e-2)


def test_rsi_bounds(small_panel):
    close, _ = small_panel
    rsi = compute_rsi(close, 14).dropna()
    assert (rsi.to_numpy() >= 0).all()
    assert (rsi.to_numpy() <= 100).all()


def test_dataset_alignment(small_panel):
    close, volume = small_panel
    X, y = build_dataset(close, volume, horizon=5)
    assert X.index.equals(y.index)
    assert not X.isna().any().any()
    assert not y.isna().any()
    # Dataset must end at least `horizon` dates before the panel ends.
    assert X.index.get_level_values("date").max() <= close.index[-6]
