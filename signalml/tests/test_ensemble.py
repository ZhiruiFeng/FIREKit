"""Tests for ensemble combination."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from signalml.ensemble import EnsembleCombiner, ic_weights


def _preds(n_dates: int = 4, n_symbols: int = 6, seed: int = 0) -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-01", periods=n_dates, name="date")
    symbols = [f"S{i}" for i in range(n_symbols)]
    idx = pd.MultiIndex.from_product([dates, symbols], names=["date", "symbol"])
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {"m1": rng.normal(size=len(idx)), "m2": rng.normal(size=len(idx))}, index=idx
    )


def test_zscore_mean_equal_weights_exact():
    preds = _preds()
    combined = EnsembleCombiner("zscore_mean").combine(preds)
    grouped = preds.groupby(level="date")
    z = (preds - grouped.transform("mean")) / (grouped.transform("std") + 1e-9)
    expected = 0.5 * z["m1"] + 0.5 * z["m2"]
    assert np.allclose(combined.to_numpy(), expected.to_numpy())


def test_single_model_weight_recovers_that_model_zscore():
    preds = _preds()
    combined = EnsembleCombiner("zscore_mean", weights={"m1": 1.0}).combine(preds)
    grouped = preds.groupby(level="date")
    z1 = (preds["m1"] - grouped["m1"].transform("mean")) / (
        grouped["m1"].transform("std") + 1e-9
    )
    assert np.allclose(combined.to_numpy(), z1.to_numpy())


def test_rank_mean_bounds_and_centering():
    preds = _preds(n_dates=3, n_symbols=10)
    combined = EnsembleCombiner("rank_mean").combine(preds)
    assert combined.min() >= -0.5
    assert combined.max() <= 0.5
    # ranks per date are symmetric around 0.55/2... mean of pct ranks is (n+1)/2n
    per_date_mean = combined.groupby(level="date").mean()
    assert np.allclose(per_date_mean.to_numpy(), 0.05, atol=1e-9)  # (11/20) - 0.5


def test_rank_mean_monotone_in_agreement():
    """When both models agree on ordering, ensemble preserves it."""
    dates = pd.bdate_range("2024-01-01", periods=1, name="date")
    idx = pd.MultiIndex.from_product([dates, ["A", "B", "C"]], names=["date", "symbol"])
    preds = pd.DataFrame({"m1": [1.0, 2.0, 3.0], "m2": [10.0, 20.0, 30.0]}, index=idx)
    combined = EnsembleCombiner("rank_mean").combine(preds)
    assert combined.loc[(dates[0], "A")] < combined.loc[(dates[0], "B")] < combined.loc[
        (dates[0], "C")
    ]


def test_weights_normalized():
    preds = _preds()
    c1 = EnsembleCombiner("zscore_mean", weights={"m1": 2.0, "m2": 2.0}).combine(preds)
    c2 = EnsembleCombiner("zscore_mean", weights={"m1": 1.0, "m2": 1.0}).combine(preds)
    assert np.allclose(c1.to_numpy(), c2.to_numpy())


def test_invalid_method_raises():
    with pytest.raises(ValueError):
        EnsembleCombiner("stacking")


def test_unknown_weight_key_raises():
    preds = _preds()
    with pytest.raises(KeyError):
        EnsembleCombiner("zscore_mean", weights={"nope": 1.0}).combine(preds)


def test_negative_or_zero_weights_raise():
    with pytest.raises(ValueError):
        EnsembleCombiner(weights={"m1": -1.0})
    with pytest.raises(ValueError):
        EnsembleCombiner(weights={"m1": 0.0})
    with pytest.raises(ValueError):
        EnsembleCombiner(weights={})


def test_ic_weights_favor_better_model():
    preds = _preds(n_dates=10, n_symbols=20, seed=1)
    actual = preds["m1"] * 1.0 + 0.1 * np.random.default_rng(2).normal(size=len(preds))
    actual = pd.Series(actual, index=preds.index)
    weights = ic_weights(preds, actual)
    assert weights["m1"] > weights["m2"]
    assert sum(weights.values()) == pytest.approx(1.0)
    assert all(w >= 0 for w in weights.values())


def test_ic_weights_fallback_equal_when_all_nonpositive():
    preds = _preds(n_dates=10, n_symbols=20, seed=3)
    actual = pd.Series(-preds["m1"] - preds["m2"], index=preds.index)
    weights = ic_weights(preds, actual)
    assert weights == {"m1": 0.5, "m2": 0.5}
