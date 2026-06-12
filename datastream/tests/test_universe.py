"""Tests for point-in-time universe membership."""

import pandas as pd
import pytest

from datastream.universe import Membership, PointInTimeUniverse


@pytest.fixture()
def universe() -> PointInTimeUniverse:
    u = PointInTimeUniverse()
    u.add("AAA", "2021-01-04")  # member the whole time
    u.add("BBB", "2021-01-04", "2022-06-01")  # delisted
    u.add("CCC", "2022-01-03")  # late joiner
    return u


def test_members_as_of_exact(universe: PointInTimeUniverse) -> None:
    assert universe.members_as_of("2021-06-01") == ["AAA", "BBB"]
    assert universe.members_as_of("2022-03-01") == ["AAA", "BBB", "CCC"]
    assert universe.members_as_of("2022-07-01") == ["AAA", "CCC"]
    assert universe.members_as_of("2020-12-31") == []


def test_end_date_is_exclusive(universe: PointInTimeUniverse) -> None:
    assert "BBB" in universe.members_as_of("2022-05-31")
    assert "BBB" not in universe.members_as_of("2022-06-01")


def test_start_date_is_inclusive(universe: PointInTimeUniverse) -> None:
    assert "CCC" not in universe.members_as_of("2021-12-31")
    assert "CCC" in universe.members_as_of("2022-01-03")


def test_all_symbols_includes_dead_ones(universe: PointInTimeUniverse) -> None:
    assert universe.all_symbols() == ["AAA", "BBB", "CCC"]


def test_membership_mask_values(universe: PointInTimeUniverse) -> None:
    dates = ["2021-06-01", "2022-03-01", "2022-07-01"]
    mask = universe.membership_mask(dates)
    assert mask.shape == (3, 3)
    assert mask.loc["2021-06-01"].tolist() == [True, True, False]
    assert mask.loc["2022-03-01"].tolist() == [True, True, True]
    assert mask.loc["2022-07-01"].tolist() == [True, False, True]


def test_member_counts(universe: PointInTimeUniverse) -> None:
    counts = universe.member_counts(["2021-06-01", "2022-03-01", "2022-07-01"])
    assert counts.tolist() == [2, 3, 2]


def test_filter_frame_drops_non_member_rows(universe: PointInTimeUniverse) -> None:
    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2021-06-01", "2022-07-01", "2021-06-01", "2021-06-01"]
            ),
            "symbol": ["BBB", "BBB", "CCC", "AAA"],
            "open": [1.0] * 4,
            "high": [1.0] * 4,
            "low": [1.0] * 4,
            "close": [1.0] * 4,
            "volume": [1.0] * 4,
        }
    )
    out = universe.filter_frame(df)
    kept = set(zip(out["symbol"], out["timestamp"].dt.strftime("%Y-%m-%d"), strict=True))
    assert kept == {("BBB", "2021-06-01"), ("AAA", "2021-06-01")}


def test_from_records_roundtrip() -> None:
    u = PointInTimeUniverse.from_records(
        [
            {"symbol": "x", "start": "2021-01-04"},
            {"symbol": "y", "start": "2021-01-04", "end": "2021-02-01"},
        ]
    )
    assert u.all_symbols() == ["X", "Y"]
    assert u.members_as_of("2021-02-01") == ["X"]


def test_add_validates_dates_and_normalizes_symbol() -> None:
    u = PointInTimeUniverse()
    m = u.add(" abc ", "2021-01-04")
    assert m == Membership("ABC", pd.Timestamp("2021-01-04"), None)
    with pytest.raises(ValueError, match="must be after"):
        u.add("BAD", "2021-02-01", "2021-01-01")


def test_membership_active_on() -> None:
    m = Membership("Z", pd.Timestamp("2021-01-04"), pd.Timestamp("2021-02-01"))
    assert not m.active_on(pd.Timestamp("2021-01-03"))
    assert m.active_on(pd.Timestamp("2021-01-04"))
    assert m.active_on(pd.Timestamp("2021-01-31"))
    assert not m.active_on(pd.Timestamp("2021-02-01"))
