"""Tests for the Parquet store."""

from pathlib import Path

import pandas as pd
import pytest

from datastream.sources import SyntheticSource
from datastream.store import ParquetStore


@pytest.fixture()
def bars() -> pd.DataFrame:
    return SyntheticSource(n_symbols=3, start="2022-01-03", end="2022-03-31", seed=5).fetch()


@pytest.fixture()
def store(tmp_path: Path) -> ParquetStore:
    return ParquetStore(tmp_path / "store")


def test_write_read_roundtrip(store: ParquetStore, bars: pd.DataFrame) -> None:
    written = store.write(bars)
    assert set(written) == {"SYM001", "SYM002", "SYM003"}
    out = store.read("SYM002")
    expected = bars[bars["symbol"] == "SYM002"].reset_index(drop=True)
    pd.testing.assert_frame_equal(out, expected)


def test_partitioned_layout(store: ParquetStore, bars: pd.DataFrame) -> None:
    store.write(bars)
    assert (store.root / "symbol=SYM001" / "data.parquet").exists()
    assert store.symbols() == ["SYM001", "SYM002", "SYM003"]
    assert store.has("sym001")
    assert not store.has("SYM999")


def test_read_unknown_symbol_raises(store: ParquetStore) -> None:
    with pytest.raises(KeyError):
        store.read("GHOST")


def test_read_date_window(store: ParquetStore, bars: pd.DataFrame) -> None:
    store.write(bars)
    out = store.read("SYM001", start="2022-02-01", end="2022-02-28")
    assert out["timestamp"].min() >= pd.Timestamp("2022-02-01")
    assert out["timestamp"].max() <= pd.Timestamp("2022-02-28")
    assert len(out) == len(pd.bdate_range("2022-02-01", "2022-02-28"))


def test_upsert_overwrites_on_timestamp(store: ParquetStore, bars: pd.DataFrame) -> None:
    one = bars[bars["symbol"] == "SYM001"].reset_index(drop=True)
    store.write(one)
    patch = one.iloc[[10]].copy()
    patch["close"] = 123.456
    store.write(patch)
    out = store.read("SYM001")
    assert len(out) == len(one)  # no duplicate row added
    assert out.loc[10, "close"] == 123.456


def test_upsert_appends_new_rows(store: ParquetStore, bars: pd.DataFrame) -> None:
    one = bars[bars["symbol"] == "SYM001"].reset_index(drop=True)
    store.write(one.iloc[:30])
    store.write(one.iloc[30:])
    out = store.read("SYM001")
    pd.testing.assert_frame_equal(out, one)


def test_read_many_concatenates(store: ParquetStore, bars: pd.DataFrame) -> None:
    store.write(bars)
    out = store.read_many(["SYM001", "SYM003"])
    assert set(out["symbol"]) == {"SYM001", "SYM003"}
    assert len(out) == (bars["symbol"] != "SYM002").sum()


def test_cache_hits_and_isolation(store: ParquetStore, bars: pd.DataFrame) -> None:
    store.write(bars)
    first = store.read("SYM001")
    assert store.cache_misses == 1
    first.loc[0, "close"] = -1.0  # mutate the returned copy
    second = store.read("SYM001")
    assert store.cache_hits == 1
    assert second.loc[0, "close"] != -1.0  # cache not poisoned


def test_cache_invalidated_on_write(store: ParquetStore, bars: pd.DataFrame) -> None:
    one = bars[bars["symbol"] == "SYM001"].reset_index(drop=True)
    store.write(one)
    store.read("SYM001")
    patch = one.iloc[[0]].copy()
    patch["close"] = 999.0
    store.write(patch)
    assert store.read("SYM001").loc[0, "close"] == 999.0


def test_cache_eviction(tmp_path: Path, bars: pd.DataFrame) -> None:
    store = ParquetStore(tmp_path, cache_size=1)
    store.write(bars)
    store.read("SYM001")
    store.read("SYM002")  # evicts SYM001
    assert store.stats()["cache_entries"] == 1
    store.read("SYM001")
    assert store.cache_misses == 3


def test_stats(store: ParquetStore, bars: pd.DataFrame) -> None:
    store.write(bars)
    stats = store.stats()
    assert stats["symbols"] == 3
    assert stats["file_count"] == 3
    assert stats["total_bytes"] > 0
    assert stats["total_mb"] == stats["total_bytes"] / (1024 * 1024)
