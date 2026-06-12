"""Tests for evaluation metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from signalml.evaluation import (
    aggregate_importances,
    cumulative_spread,
    decile_spread,
    hit_rate,
    information_coefficient,
    per_date_ic,
    rank_ic,
    summarize,
)


def _index(n_dates: int, n_symbols: int) -> pd.MultiIndex:
    dates = pd.bdate_range("2024-01-01", periods=n_dates, name="date")
    symbols = [f"S{i}" for i in range(n_symbols)]
    return pd.MultiIndex.from_product([dates, symbols], names=["date", "symbol"])


def test_perfect_ic_is_one():
    idx = _index(5, 10)
    rng = np.random.default_rng(0)
    actual = pd.Series(rng.normal(size=len(idx)), index=idx)
    assert information_coefficient(actual, actual) == pytest.approx(1.0)
    assert rank_ic(actual, actual) == pytest.approx(1.0)


def test_inverted_ic_is_minus_one():
    idx = _index(5, 10)
    rng = np.random.default_rng(1)
    actual = pd.Series(rng.normal(size=len(idx)), index=idx)
    assert information_coefficient(-actual, actual) == pytest.approx(-1.0)


def test_rank_ic_ignores_monotone_transform():
    idx = _index(4, 12)
    rng = np.random.default_rng(2)
    actual = pd.Series(rng.normal(size=len(idx)), index=idx)
    pred = actual**3  # monotone, nonlinear
    assert rank_ic(pred, actual) == pytest.approx(1.0)
    assert information_coefficient(pred, actual) < 1.0


def test_per_date_ic_has_one_value_per_date():
    idx = _index(7, 10)
    rng = np.random.default_rng(3)
    pred = pd.Series(rng.normal(size=len(idx)), index=idx)
    actual = pd.Series(rng.normal(size=len(idx)), index=idx)
    ic = per_date_ic(pred, actual)
    assert len(ic) == 7


def test_hit_rate_exact():
    idx = _index(1, 4)
    pred = pd.Series([1.0, -1.0, 1.0, -1.0], index=idx)
    actual = pd.Series([0.5, -0.5, -0.5, -0.5], index=idx)  # 3 of 4 agree
    assert hit_rate(pred, actual) == pytest.approx(0.75)


def test_decile_spread_exact_on_ordered_data():
    """10 symbols, 1 date: pred rank == actual value -> spread = top - bottom."""
    idx = _index(1, 10)
    values = np.arange(10, dtype=float)  # 0..9
    pred = pd.Series(values, index=idx)
    actual = pd.Series(values / 100.0, index=idx)
    spread = decile_spread(pred, actual, quantiles=10)
    assert len(spread) == 1
    assert spread.iloc[0] == pytest.approx(0.09 - 0.00)


def test_quintile_spread_exact():
    idx = _index(1, 10)
    values = np.arange(10, dtype=float)
    pred = pd.Series(values, index=idx)
    actual = pd.Series(values, index=idx)
    spread = decile_spread(pred, actual, quantiles=5)
    # top quintile {8,9} mean 8.5, bottom {0,1} mean 0.5
    assert spread.iloc[0] == pytest.approx(8.0)


def test_cumulative_spread_is_cumsum():
    s = pd.Series([0.1, -0.05, 0.2], index=pd.bdate_range("2024-01-01", periods=3, name="date"))
    cs = cumulative_spread(s)
    assert cs.iloc[-1] == pytest.approx(0.25)


def test_summarize_keys():
    idx = _index(6, 10)
    rng = np.random.default_rng(4)
    pred = pd.Series(rng.normal(size=len(idx)), index=idx)
    actual = pd.Series(rng.normal(size=len(idx)), index=idx)
    metrics = summarize(pred, actual)
    assert set(metrics) == {"ic", "rank_ic", "hit_rate", "mean_decile_spread", "n_obs"}
    assert metrics["n_obs"] == 60


def test_invalid_method_raises():
    idx = _index(2, 5)
    s = pd.Series(np.arange(10, dtype=float), index=idx)
    with pytest.raises(ValueError):
        per_date_ic(s, s, method="kendall")


def test_empty_overlap_raises():
    idx1 = _index(2, 3)
    idx2 = _index(2, 3)
    pred = pd.Series(np.ones(6), index=idx1)
    actual = pd.Series(np.full(6, np.nan), index=idx2)
    with pytest.raises(ValueError):
        information_coefficient(pred, actual)


def test_aggregate_importances_normalized_and_sorted():
    frame = pd.DataFrame({"a": [0.5, 0.7], "b": [0.2, 0.2], "c": [0.3, 0.1]})
    agg = aggregate_importances(frame)
    assert agg.sum() == pytest.approx(1.0)
    assert list(agg.index) == ["a", "b", "c"]
    assert agg["a"] == pytest.approx(0.6)
