"""Tests for the news pipeline: NewsItem, tagging, dedup."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest
from pydantic import ValidationError

from sentimentpulse.news import AliasMap, NewsItem, deduplicate, tag_items, tag_symbols


def _item(i: int, headline: str, ts: datetime, symbol: str | None = "AAA") -> NewsItem:
    return NewsItem(id=f"n{i}", timestamp=ts, headline=headline, symbol=symbol)


@pytest.fixture()
def alias_map() -> AliasMap:
    return AliasMap(
        aliases={
            "APDY": ["Apex Dynamics", "Apex"],
            "BLSY": ["Blue Systems"],
        }
    )


def test_news_item_requires_headline():
    with pytest.raises(ValidationError):
        NewsItem(id="x", timestamp=datetime(2024, 1, 1), headline="   ")


def test_news_item_text_combines_headline_and_body():
    item = NewsItem(id="x", timestamp=datetime(2024, 1, 1), headline="H", body="B")
    assert item.text == "H\nB"
    assert _item(0, "Only headline", datetime(2024, 1, 1)).text == "Only headline"


def test_tag_symbols_case_insensitive(alias_map):
    assert tag_symbols("APEX DYNAMICS posts results", alias_map) == ["APDY"]
    assert tag_symbols("ticker apdy mentioned", alias_map) == ["APDY"]


def test_tag_symbols_word_boundary(alias_map):
    # "Apexes" must not match the alias "Apex"
    assert tag_symbols("Apexes are unrelated", alias_map) == []


def test_tag_symbols_multiple(alias_map):
    syms = tag_symbols("Apex Dynamics to merge with Blue Systems", alias_map)
    assert syms == ["APDY", "BLSY"]


def test_tag_items_assigns_and_duplicates(alias_map):
    ts = datetime(2024, 3, 1, 9)
    items = [
        _item(0, "Apex Dynamics and Blue Systems sign deal", ts, symbol=None),
        _item(1, "Nothing relevant here", ts, symbol=None),
        _item(2, "Already tagged", ts, symbol="ZZZ"),
    ]
    tagged = tag_items(items, alias_map)
    assert [(t.id, t.symbol) for t in tagged] == [
        ("n0", "APDY"),
        ("n0", "BLSY"),
        ("n2", "ZZZ"),
    ]


def test_deduplicate_drops_near_duplicates_keeps_earliest():
    t0 = datetime(2024, 5, 1, 9)
    items = [
        _item(0, "Acme beats estimates!", t0),
        _item(1, "ACME beats estimates", t0 + timedelta(hours=3)),  # dup (case/punct)
        _item(2, "Acme beats estimates", t0 + timedelta(hours=20)),  # dup within window
    ]
    kept = deduplicate(items, window=timedelta(days=1))
    assert [k.id for k in kept] == ["n0"]


def test_deduplicate_keeps_same_headline_outside_window():
    t0 = datetime(2024, 5, 1, 9)
    items = [
        _item(0, "Acme beats estimates", t0),
        _item(1, "Acme beats estimates", t0 + timedelta(days=3)),
    ]
    kept = deduplicate(items, window=timedelta(days=1))
    assert [k.id for k in kept] == ["n0", "n1"]


def test_deduplicate_distinguishes_symbols():
    t0 = datetime(2024, 5, 1, 9)
    items = [
        _item(0, "Beats estimates", t0, symbol="AAA"),
        _item(1, "Beats estimates", t0, symbol="BBB"),
    ]
    kept = deduplicate(items)
    assert len(kept) == 2


def test_deduplicate_sorts_by_timestamp():
    t0 = datetime(2024, 5, 1, 9)
    items = [
        _item(1, "Second headline", t0 + timedelta(hours=1)),
        _item(0, "First headline", t0),
    ]
    kept = deduplicate(items)
    assert [k.id for k in kept] == ["n0", "n1"]
