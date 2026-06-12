"""Shared fixtures: hand-built deterministic price feeds."""

from __future__ import annotations

import pytest

from executioncore.market import Bar, PriceFeed

SYMBOL = "TEST"


def make_feed(specs: list[tuple[float, float, float, float, float]], symbol: str = SYMBOL) -> PriceFeed:
    """Build a feed from (open, high, low, close, volume) tuples."""
    bars = [Bar(index=i, open=o, high=h, low=lo, close=c, volume=v) for i, (o, h, lo, c, v) in enumerate(specs)]
    return PriceFeed({symbol: bars})


@pytest.fixture
def flat_feed() -> PriceFeed:
    """Six flat bars at 100 with plenty of volume."""
    return make_feed([(100.0, 100.5, 99.5, 100.0, 10_000.0)] * 6)


@pytest.fixture
def trend_feed() -> PriceFeed:
    """A deterministic path that first falls then rallies."""
    return make_feed(
        [
            (100.0, 100.6, 99.8, 100.2, 10_000.0),  # 0
            (100.2, 100.4, 99.0, 99.2, 10_000.0),  # 1: dips to 99.0
            (99.2, 99.5, 98.5, 99.0, 10_000.0),  # 2: lower
            (99.0, 101.0, 98.9, 100.8, 10_000.0),  # 3: rallies through 101
            (100.8, 102.0, 100.5, 101.5, 10_000.0),  # 4
            (101.5, 102.5, 101.0, 102.0, 10_000.0),  # 5
        ]
    )
