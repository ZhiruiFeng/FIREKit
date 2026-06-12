"""Tests for the synthetic news/price generators."""

from __future__ import annotations

import pandas as pd
import pytest

from sentimentpulse.news import tag_symbols
from sentimentpulse.synthetic import (
    company_name,
    generate_news,
    generate_prices,
    generate_universe,
)


@pytest.fixture(scope="module")
def universe():
    return generate_universe(10)


@pytest.fixture(scope="module")
def news(universe):
    symbols, alias_map = universe
    return generate_news(symbols, alias_map, n_items=400, n_days=120, seed=5)


def test_universe_unique_symbols(universe):
    symbols, alias_map = universe
    assert len(symbols) == len(set(symbols)) == 10
    assert set(alias_map.aliases) == set(symbols)


def test_universe_invalid_size():
    with pytest.raises(ValueError):
        generate_universe(0)


def test_news_count_and_alignment(news):
    items, truths = news
    assert len(items) == len(truths) == 400
    assert all(t in (-1, 0, 1) for t in truths)


def test_news_deterministic(universe):
    symbols, alias_map = universe
    a_items, a_truths = generate_news(symbols, alias_map, n_items=50, seed=9)
    b_items, b_truths = generate_news(symbols, alias_map, n_items=50, seed=9)
    assert a_truths == b_truths
    assert [i.headline for i in a_items] == [i.headline for i in b_items]
    assert [i.timestamp for i in a_items] == [i.timestamp for i in b_items]


def test_news_covers_all_classes(news):
    _, truths = news
    counts = {t: truths.count(t) for t in (-1, 0, 1)}
    assert all(c > 20 for c in counts.values())
    # roughly 40/40/20 mix
    assert counts[1] > counts[0]
    assert counts[-1] > counts[0]


def test_news_sorted_and_in_range(news):
    items, _ = news
    stamps = [i.timestamp for i in items]
    assert stamps == sorted(stamps)
    dates = pd.bdate_range("2024-01-01", periods=120)
    assert min(stamps) >= dates[0].to_pydatetime()
    assert max(stamps) <= (dates[-1] + pd.Timedelta(days=1)).to_pydatetime()


def test_headlines_mention_their_symbol(universe, news):
    symbols, alias_map = universe
    items, _ = news
    for item in items[:50]:
        assert item.symbol in tag_symbols(item.headline, alias_map)
        assert company_name(item.symbol, alias_map) in item.headline


def test_prices_shape_positive_deterministic(universe, news):
    symbols, alias_map = universe
    items, truths = news
    dates = pd.bdate_range("2024-01-01", periods=120)
    a = generate_prices(symbols, dates, items, truths, seed=3)
    b = generate_prices(symbols, dates, items, truths, seed=3)
    assert a.shape == (120, 10)
    assert (a.to_numpy() > 0).all()
    pd.testing.assert_frame_equal(a, b)


def test_prices_have_injected_sentiment_impact(universe):
    """Prior-day net news polarity must positively predict next-day returns."""
    symbols, alias_map = universe
    items, truths = generate_news(symbols, alias_map, n_items=1500, n_days=250, seed=4)
    dates = pd.bdate_range("2024-01-01", periods=250)
    close = generate_prices(symbols, dates, items, truths, seed=5, impact=0.006)
    rets = close.pct_change()

    net = pd.DataFrame(0.0, index=dates, columns=symbols)
    for item, truth in zip(items, truths, strict=True):
        day = pd.Timestamp(item.timestamp).normalize()
        if day in net.index:
            net.at[day, item.symbol] += truth

    lagged = net.shift(1).stack()
    fwd = rets.stack()
    aligned = pd.concat([lagged, fwd], axis=1, keys=["net", "ret"]).dropna()
    corr = aligned["net"].corr(aligned["ret"])
    assert corr > 0.1


def test_zero_impact_kills_predictive_power(universe):
    symbols, alias_map = universe
    items, truths = generate_news(symbols, alias_map, n_items=800, n_days=200, seed=6)
    dates = pd.bdate_range("2024-01-01", periods=200)
    close = generate_prices(symbols, dates, items, truths, seed=7, impact=0.0)
    rets = close.pct_change()
    net = pd.DataFrame(0.0, index=dates, columns=symbols)
    for item, truth in zip(items, truths, strict=True):
        day = pd.Timestamp(item.timestamp).normalize()
        if day in net.index:
            net.at[day, item.symbol] += truth
    aligned = pd.concat(
        [net.shift(1).stack(), rets.stack()], axis=1, keys=["net", "ret"]
    ).dropna()
    assert abs(aligned["net"].corr(aligned["ret"])) < 0.05
