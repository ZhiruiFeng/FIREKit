"""Tests for the walk-forward engine."""

from __future__ import annotations

import pandas as pd
import pytest

from signalml.models import RidgeSignalModel
from signalml.walkforward import WalkForwardEngine


def _engine(**kwargs) -> WalkForwardEngine:
    defaults = {
        "model_factories": {"ridge": RidgeSignalModel},
        "train_size": 60,
        "test_size": 10,
        "gap": 5,
    }
    defaults.update(kwargs)
    return WalkForwardEngine(**defaults)


def test_windows_respect_purged_gap(small_dataset):
    X, _ = small_dataset
    dates = pd.DatetimeIndex(X.index.get_level_values("date").unique()).sort_values()
    windows = _engine().make_windows(dates)
    assert len(windows) >= 2
    for w in windows:
        n_gap = ((dates > w.train_end) & (dates < w.test_start)).sum()
        assert n_gap == 5
        assert w.train_end < w.test_start


def test_windows_step_by_test_size(small_dataset):
    X, _ = small_dataset
    dates = pd.DatetimeIndex(X.index.get_level_values("date").unique()).sort_values()
    windows = _engine().make_windows(dates)
    for prev, nxt in zip(windows, windows[1:], strict=False):
        # consecutive test windows are contiguous and non-overlapping
        assert nxt.test_start > prev.test_end
        gap_between = ((dates > prev.test_end) & (dates < nxt.test_start)).sum()
        assert gap_between == 0


def test_rolling_vs_expanding_train_start(small_dataset):
    X, _ = small_dataset
    dates = pd.DatetimeIndex(X.index.get_level_values("date").unique()).sort_values()
    rolling = _engine(expanding=False).make_windows(dates)
    expanding = _engine(expanding=True).make_windows(dates)
    assert all(w.train_start == dates[0] for w in expanding)
    assert rolling[1].train_start > rolling[0].train_start
    assert expanding[-1].n_train > rolling[-1].n_train


def test_predictions_cover_only_test_dates(small_dataset):
    X, y = small_dataset
    result = _engine().run(X, y)
    pred_dates = set(result.predictions.index.get_level_values("date").unique())
    test_dates: set = set()
    all_dates = pd.DatetimeIndex(X.index.get_level_values("date").unique()).sort_values()
    for w in result.windows:
        test_dates.update(all_dates[(all_dates >= w.test_start) & (all_dates <= w.test_end)])
    assert pred_dates == test_dates
    # no prediction falls inside any training or gap region of its window
    for w in result.windows:
        assert not any(d <= w.train_end for d in pred_dates if d >= w.test_start)


def test_result_alignment_and_columns(small_dataset):
    X, y = small_dataset
    factories = {"ridge": RidgeSignalModel, "ridge2": lambda: RidgeSignalModel(alpha=100.0)}
    result = _engine(model_factories=factories).run(X, y)
    assert list(result.predictions.columns) == ["ridge", "ridge2"]
    assert result.predictions.index.equals(result.actuals.index)
    assert not result.predictions.isna().any().any()


def test_window_metrics_shape(small_dataset):
    X, y = small_dataset
    factories = {"ridge": RidgeSignalModel, "ridge2": lambda: RidgeSignalModel(alpha=100.0)}
    result = _engine(model_factories=factories).run(X, y)
    n = len(result.windows)
    assert len(result.window_metrics) == 2 * n
    assert set(result.window_metrics["model"]) == {"ridge", "ridge2"}
    assert {"ic", "rank_ic", "hit_rate"} <= set(result.window_metrics.columns)


def test_importances_one_row_per_window(small_dataset):
    X, y = small_dataset
    result = _engine().run(X, y)
    imp = result.importances["ridge"]
    assert len(imp) == len(result.windows)
    assert list(imp.columns) == list(X.columns)


def test_oos_ic_positive_on_injected_signal(small_dataset):
    """The synthetic panel has real signal; OOS ridge IC should be positive."""
    X, y = small_dataset
    result = _engine().run(X, y)
    assert result.window_metrics["ic"].mean() > 0.02


def test_too_few_dates_raises(small_dataset):
    X, y = small_dataset
    engine = _engine(train_size=5000)
    with pytest.raises(ValueError, match="not enough dates"):
        engine.run(X, y)


def test_empty_factories_raise():
    with pytest.raises(ValueError):
        WalkForwardEngine(model_factories={})


def test_misaligned_xy_raises(small_dataset):
    X, y = small_dataset
    with pytest.raises(ValueError):
        _engine().run(X, y.iloc[:-10])
