"""Tests for accuracy evaluation and the event study."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sentimentpulse.aggregate import daily_scores, detect_shocks, scores_frame
from sentimentpulse.evaluation import event_study, forward_returns, scorer_accuracy
from sentimentpulse.providers import LexiconProvider
from sentimentpulse.synthetic import generate_news, generate_prices, generate_universe


def test_scorer_accuracy_perfect():
    out = scorer_accuracy([0.9, -0.9, 0.0], [1, -1, 0])
    assert out["accuracy"] == 1.0
    assert out["n"] == 3
    assert out["per_class"] == {"negative": 1.0, "neutral": 1.0, "positive": 1.0}


def test_scorer_accuracy_confusion_counts():
    out = scorer_accuracy([0.9, 0.9, -0.9, 0.0], [1, -1, -1, 0])
    assert out["accuracy"] == pytest.approx(3 / 4)
    assert out["confusion"]["negative"]["positive"] == 1
    assert out["confusion"]["negative"]["negative"] == 1
    total = sum(sum(row.values()) for row in out["confusion"].values())
    assert total == 4


def test_scorer_accuracy_validates_inputs():
    with pytest.raises(ValueError):
        scorer_accuracy([0.1], [1, 0])
    with pytest.raises(ValueError):
        scorer_accuracy([], [])
    with pytest.raises(ValueError):
        scorer_accuracy([0.1], [2])


def test_lexicon_accuracy_on_synthetic_news_is_high():
    symbols, alias_map = generate_universe(10)
    items, truths = generate_news(symbols, alias_map, n_items=600, n_days=200, seed=13)
    provider = LexiconProvider()
    scores = [provider.score_text(item.text).score for item in items]
    out = scorer_accuracy(scores, truths)
    assert out["accuracy"] > 0.8
    assert out["per_class"]["positive"] > 0.7
    assert out["per_class"]["negative"] > 0.7


def test_forward_returns_exact():
    dates = pd.bdate_range("2024-01-01", periods=4)
    close = pd.DataFrame({"A": [100.0, 110.0, 121.0, 121.0]}, index=dates)
    fwd = forward_returns(close, 1)
    assert fwd.iloc[0, 0] == pytest.approx(0.10)
    assert np.isnan(fwd.iloc[-1, 0])
    with pytest.raises(ValueError):
        forward_returns(close, 0)


def test_event_study_recovers_constructed_effect():
    """Hand-built prices: +5% the day after positive events, -5% after negative."""
    dates = pd.bdate_range("2024-01-01", periods=12)
    rets = np.zeros(12)
    rets[5] = 0.05  # day after event at dates[4]
    rets[9] = -0.05  # day after event at dates[8]
    close = pd.DataFrame({"A": 100 * np.cumprod(1 + rets)}, index=dates)
    events = pd.DataFrame(
        {
            "date": [dates[4], dates[8]],
            "symbol": ["A", "A"],
            "direction": ["positive", "negative"],
        }
    )
    study = event_study(events, close, horizons=(1,))
    assert study.at["positive", "fwd_1d"] == pytest.approx(0.05)
    assert study.at["negative", "fwd_1d"] == pytest.approx(-0.05)
    assert study.at["positive", "n_events"] == 1


def test_event_study_requires_columns():
    close = pd.DataFrame({"A": [1.0, 2.0]}, index=pd.bdate_range("2024-01-01", periods=2))
    with pytest.raises(ValueError):
        event_study(pd.DataFrame({"date": []}), close)


def test_full_pipeline_recovers_injected_signal():
    """End-to-end: scored synthetic news -> shocks -> event study shows that
    positive shocks precede higher returns than negative shocks."""
    symbols, alias_map = generate_universe(12)
    items, truths = generate_news(symbols, alias_map, n_items=1800, n_days=300, seed=17)
    dates = pd.bdate_range("2024-01-01", periods=300)
    close = generate_prices(symbols, dates, items, truths, seed=18, impact=0.008)

    provider = LexiconProvider()
    results = provider.score_batch([item.text for item in items])
    frame = scores_frame(items, results)
    daily = daily_scores(frame, dates=dates)
    shocks = detect_shocks(daily, window=20, z_threshold=1.5)
    assert len(shocks) > 10

    study = event_study(shocks, close, horizons=(1, 3))
    assert study.at["positive", "fwd_1d"] > study.at["negative", "fwd_1d"]
    assert study.at["positive", "fwd_3d"] > study.at["negative", "fwd_3d"]
