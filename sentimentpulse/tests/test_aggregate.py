"""Tests for sentiment aggregation: daily index, shocks, breadth."""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from sentimentpulse.aggregate import (
    daily_scores,
    detect_shocks,
    scores_frame,
    sentiment_breadth,
    sentiment_index,
    sentiment_momentum,
)
from sentimentpulse.news import NewsItem
from sentimentpulse.scorer import SentimentResult


def _result(score: float) -> SentimentResult:
    return SentimentResult(score=score, uncertainty=0.0, litigious=0.0)


def _item(i: int, ts: datetime, symbol: str) -> NewsItem:
    return NewsItem(id=f"n{i}", timestamp=ts, headline=f"headline {i}", symbol=symbol)


def test_scores_frame_alignment():
    items = [
        _item(0, datetime(2024, 1, 2, 9), "AAA"),
        _item(1, datetime(2024, 1, 2, 15), "AAA"),
    ]
    frame = scores_frame(items, [_result(1.0), _result(0.0)])
    assert list(frame.columns) == ["timestamp", "date", "symbol", "score", "uncertainty"]
    assert len(frame) == 2
    with pytest.raises(ValueError):
        scores_frame(items, [_result(1.0)])


def test_daily_scores_means_per_symbol_day():
    items = [
        _item(0, datetime(2024, 1, 2, 9), "AAA"),
        _item(1, datetime(2024, 1, 2, 15), "AAA"),
        _item(2, datetime(2024, 1, 3, 10), "BBB"),
    ]
    frame = scores_frame(items, [_result(1.0), _result(0.0), _result(-1.0)])
    daily = daily_scores(frame)
    assert daily.at[pd.Timestamp("2024-01-02"), "AAA"] == pytest.approx(0.5)
    assert daily.at[pd.Timestamp("2024-01-03"), "BBB"] == pytest.approx(-1.0)
    assert np.isnan(daily.at[pd.Timestamp("2024-01-03"), "AAA"])


def test_daily_scores_reindexed_to_calendar():
    items = [_item(0, datetime(2024, 1, 2, 9), "AAA")]
    frame = scores_frame(items, [_result(1.0)])
    dates = pd.bdate_range("2024-01-01", periods=5)
    daily = daily_scores(frame, dates=dates)
    assert len(daily) == 5
    assert daily.notna().sum().sum() == 1


def test_sentiment_index_is_decay_weighted():
    """A recent observation must outweigh an equally sized older one."""
    dates = pd.bdate_range("2024-01-01", periods=10)
    daily = pd.DataFrame(index=dates, columns=["AAA"], dtype=float)
    daily.iloc[0, 0] = 1.0
    daily.iloc[9, 0] = -1.0
    index = sentiment_index(daily, halflife=2.0)
    # last value dominated by the recent -1, but pulled up slightly by old +1
    assert index.iloc[-1, 0] < -0.9
    assert index.iloc[-1, 0] > -1.0
    # before the second observation, the index holds the first value
    assert index.iloc[5, 0] == pytest.approx(1.0)


def test_sentiment_index_invalid_halflife():
    daily = pd.DataFrame({"AAA": [0.1]}, index=pd.bdate_range("2024-01-01", periods=1))
    with pytest.raises(ValueError):
        sentiment_index(daily, halflife=0.0)


def test_momentum_is_diff():
    dates = pd.bdate_range("2024-01-01", periods=6)
    idx = pd.DataFrame({"AAA": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]}, index=dates)
    mom = sentiment_momentum(idx, lookback=2)
    assert mom.iloc[-1, 0] == pytest.approx(0.2)


def test_detect_shocks_finds_injected_spike():
    dates = pd.bdate_range("2024-01-01", periods=40)
    rng = np.random.default_rng(0)
    daily = pd.DataFrame(
        {"AAA": rng.normal(0.0, 0.05, 40), "BBB": rng.normal(0.0, 0.05, 40)},
        index=dates,
    )
    daily.iloc[30, 0] = 0.95  # huge positive spike for AAA
    shocks = detect_shocks(daily, window=20, z_threshold=2.0)
    assert ("AAA" in set(shocks["symbol"])) and len(shocks) >= 1
    spike = shocks[(shocks["symbol"] == "AAA") & (shocks["date"] == dates[30])]
    assert len(spike) == 1
    assert spike.iloc[0]["direction"] == "positive"
    assert spike.iloc[0]["zscore"] > 2.0


def test_detect_shocks_constant_series_has_none():
    dates = pd.bdate_range("2024-01-01", periods=40)
    daily = pd.DataFrame({"AAA": np.full(40, 0.2)}, index=dates)
    shocks = detect_shocks(daily, window=20, z_threshold=2.0)
    assert shocks.empty


def test_detect_shocks_negative_direction():
    dates = pd.bdate_range("2024-01-01", periods=40)
    daily = pd.DataFrame({"AAA": np.zeros(40)}, index=dates)
    daily.iloc[::2, 0] = 0.05  # some benign variation
    daily.iloc[35, 0] = -0.9
    shocks = detect_shocks(daily, window=20, z_threshold=2.0)
    drop = shocks[shocks["date"] == dates[35]]
    assert len(drop) == 1
    assert drop.iloc[0]["direction"] == "negative"


def test_breadth_exact():
    dates = pd.bdate_range("2024-01-01", periods=2)
    idx = pd.DataFrame(
        {"A": [0.5, -0.1], "B": [0.2, 0.3], "C": [-0.4, np.nan], "D": [0.0, 0.4]},
        index=dates,
    )
    breadth = sentiment_breadth(idx)
    assert breadth.iloc[0] == pytest.approx(2 / 4)  # A, B positive; D == 0 not counted
    assert breadth.iloc[1] == pytest.approx(2 / 3)  # B, D of 3 valid


def test_empty_frame_raises():
    with pytest.raises(ValueError):
        daily_scores(pd.DataFrame())
