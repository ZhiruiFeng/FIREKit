"""Tests for SyntheticSource and FileSource."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from datastream.schema import CANONICAL_COLUMNS
from datastream.sources import FileSource, SyntheticSource


@pytest.fixture(scope="module")
def small_source() -> SyntheticSource:
    return SyntheticSource(n_symbols=4, start="2022-01-03", end="2022-06-30", seed=7)


def test_synthetic_symbols_default_naming() -> None:
    src = SyntheticSource(n_symbols=3, start="2022-01-03", end="2022-06-30")
    assert src.list_symbols() == ["SYM001", "SYM002", "SYM003"]


def test_synthetic_deterministic(small_source: SyntheticSource) -> None:
    other = SyntheticSource(n_symbols=4, start="2022-01-03", end="2022-06-30", seed=7)
    a = small_source.fetch()
    b = other.fetch()
    pd.testing.assert_frame_equal(a, b)


def test_synthetic_seed_changes_data(small_source: SyntheticSource) -> None:
    other = SyntheticSource(n_symbols=4, start="2022-01-03", end="2022-06-30", seed=8)
    a = small_source.fetch(["SYM001"])
    b = other.fetch(["SYM001"])
    assert not np.allclose(a["close"], b["close"])


def test_synthetic_shape_and_calendar(small_source: SyntheticSource) -> None:
    df = small_source.fetch()
    n_days = len(pd.bdate_range("2022-01-03", "2022-06-30"))
    assert len(df) == 4 * n_days
    assert set(df.columns) >= set(CANONICAL_COLUMNS)
    one = df[df["symbol"] == "SYM002"]
    assert one["timestamp"].is_monotonic_increasing
    assert one["timestamp"].iloc[0] == pd.Timestamp("2022-01-03")


def test_synthetic_clean_data_is_ohlc_consistent(small_source: SyntheticSource) -> None:
    df = small_source.fetch()
    assert (df["high"] >= df[["open", "close"]].max(axis=1) - 1e-12).all()
    assert (df["low"] <= df[["open", "close"]].min(axis=1) + 1e-12).all()
    assert (df["high"] >= df["low"]).all()
    assert (df[["open", "high", "low", "close"]] > 0).all().all()
    assert (df["volume"] >= 0).all()


def test_synthetic_symbol_subset_independent_of_order(small_source: SyntheticSource) -> None:
    full = small_source.fetch()
    sub = small_source.fetch(["SYM003"])
    expected = full[full["symbol"] == "SYM003"].reset_index(drop=True)
    pd.testing.assert_frame_equal(sub, expected)


def test_synthetic_unknown_symbol_raises(small_source: SyntheticSource) -> None:
    with pytest.raises(KeyError):
        small_source.fetch(["NOPE"])


def test_synthetic_date_clipping(small_source: SyntheticSource) -> None:
    df = small_source.fetch(["SYM001"], start="2022-03-01", end="2022-03-31")
    assert df["timestamp"].min() >= pd.Timestamp("2022-03-01")
    assert df["timestamp"].max() <= pd.Timestamp("2022-03-31")
    assert len(df) == len(pd.bdate_range("2022-03-01", "2022-03-31"))


def test_synthetic_regime_volatility_ordering() -> None:
    calm = SyntheticSource(
        n_symbols=6, start="2021-01-04", end="2022-12-30", seed=3, regimes=("sideways",)
    ).fetch()
    wild = SyntheticSource(
        n_symbols=6, start="2021-01-04", end="2022-12-30", seed=3, regimes=("volatile",)
    ).fetch()

    def avg_vol(df: pd.DataFrame) -> float:
        return float(
            df.groupby("symbol")["close"].apply(lambda s: s.pct_change().std()).mean()
        )

    assert avg_vol(wild) > 2 * avg_vol(calm)


def test_synthetic_issue_injection_creates_detectable_problems() -> None:
    src = SyntheticSource(n_symbols=3, start="2021-01-04", end="2023-12-29", seed=11,
                          issue_rate=0.02)
    df = src.fetch()
    nonpositive = (df[["open", "high", "low", "close"]] <= 0).any(axis=1).sum()
    inconsistent = (df["high"] < df["low"]).sum()
    dups = df.duplicated(subset=["symbol", "timestamp"]).sum()
    assert nonpositive > 0
    assert inconsistent > 0
    assert dups > 0


def test_synthetic_invalid_args() -> None:
    with pytest.raises(ValueError, match="unknown regimes"):
        SyntheticSource(n_symbols=2, regimes=("moon",))
    with pytest.raises(ValueError, match="issue_rate"):
        SyntheticSource(n_symbols=2, issue_rate=0.5)


def test_file_source_csv_roundtrip(tmp_path: Path, small_source: SyntheticSource) -> None:
    df = small_source.fetch(["SYM001"])
    csv_path = tmp_path / "SYM001.csv"
    df.to_csv(csv_path, index=False)
    src = FileSource(csv_path)
    loaded = src.fetch()
    assert len(loaded) == len(df)
    assert np.allclose(loaded["close"], df["close"])
    assert src.list_symbols() == ["SYM001"]


def test_file_source_parquet_directory(tmp_path: Path, small_source: SyntheticSource) -> None:
    for sym in ["SYM001", "SYM002"]:
        small_source.fetch([sym]).to_parquet(tmp_path / f"{sym}.parquet", index=False)
    src = FileSource(tmp_path)
    assert src.list_symbols() == ["SYM001", "SYM002"]
    df = src.fetch(symbols=["SYM002"], start="2022-02-01")
    assert set(df["symbol"]) == {"SYM002"}
    assert df["timestamp"].min() >= pd.Timestamp("2022-02-01")


def test_file_source_fills_symbol_from_filename(tmp_path: Path) -> None:
    raw = pd.DataFrame(
        {"date": ["2023-01-03"], "open": [1.0], "high": [2.0], "low": [0.5],
         "close": [1.5], "volume": [10.0]}
    )
    raw.to_csv(tmp_path / "tsla.csv", index=False)
    df = FileSource(tmp_path).fetch()
    assert list(df["symbol"]) == ["TSLA"]


def test_file_source_missing_path_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        FileSource(tmp_path / "nope.csv")
    with pytest.raises(ValueError, match="unsupported"):
        bad = tmp_path / "data.txt"
        bad.write_text("hi")
        FileSource(bad)
