"""Tests for the OHLCV schema and frame normalization."""

import pandas as pd
import pytest

from datastream.schema import CANONICAL_COLUMNS, OHLCVBar, bars_to_frame, normalize_frame


def test_bar_normalizes_symbol() -> None:
    bar = OHLCVBar(
        symbol="  aapl ",
        timestamp=pd.Timestamp("2023-01-03"),
        open=100.0,
        high=101.0,
        low=99.0,
        close=100.5,
        volume=1_000.0,
    )
    assert bar.symbol == "AAPL"
    assert bar.vwap is None
    assert bar.trade_count is None


def test_bar_rejects_empty_symbol() -> None:
    with pytest.raises(ValueError):
        OHLCVBar(
            symbol="  ",
            timestamp=pd.Timestamp("2023-01-03"),
            open=1.0,
            high=1.0,
            low=1.0,
            close=1.0,
            volume=0.0,
        )


def test_bars_to_frame_exact_values() -> None:
    bars = [
        OHLCVBar(
            symbol="msft",
            timestamp=pd.Timestamp("2023-01-04"),
            open=10.0,
            high=11.0,
            low=9.5,
            close=10.5,
            volume=500.0,
        ),
        OHLCVBar(
            symbol="msft",
            timestamp=pd.Timestamp("2023-01-03"),
            open=9.0,
            high=10.0,
            low=8.5,
            close=9.8,
            volume=400.0,
        ),
    ]
    df = bars_to_frame(bars)
    assert list(df["timestamp"]) == [pd.Timestamp("2023-01-03"), pd.Timestamp("2023-01-04")]
    assert df.loc[0, "close"] == 9.8
    assert df.loc[1, "volume"] == 500.0
    assert (df["symbol"] == "MSFT").all()


def test_normalize_frame_maps_aliases() -> None:
    raw = pd.DataFrame(
        {
            "Date": ["2023-01-03", "2023-01-04"],
            "Ticker": ["spy", "spy"],
            "O": [10, 11],
            "H": [12, 12],
            "L": [9, 10],
            "C": [11, 11.5],
            "Vol": ["100", "200"],
        }
    )
    df = normalize_frame(raw)
    assert list(df.columns) == CANONICAL_COLUMNS
    assert df["timestamp"].dtype.kind == "M"
    assert df.loc[0, "open"] == 10.0
    assert df.loc[1, "volume"] == 200.0
    assert df.loc[0, "symbol"] == "SPY"


def test_normalize_frame_promotes_datetime_index_and_symbol_arg() -> None:
    idx = pd.DatetimeIndex(["2023-01-03", "2023-01-04"], name="date")
    raw = pd.DataFrame(
        {"open": [1.0, 2.0], "high": [2.0, 3.0], "low": [0.5, 1.5], "close": [1.5, 2.5],
         "volume": [10.0, 20.0]},
        index=idx,
    )
    df = normalize_frame(raw, symbol="qqq")
    assert (df["symbol"] == "QQQ").all()
    assert list(df["timestamp"]) == list(idx)


def test_normalize_frame_sorts_by_symbol_then_time() -> None:
    raw = pd.DataFrame(
        {
            "timestamp": ["2023-01-04", "2023-01-03", "2023-01-03"],
            "symbol": ["B", "B", "A"],
            "open": [1.0, 1.0, 1.0],
            "high": [1.0, 1.0, 1.0],
            "low": [1.0, 1.0, 1.0],
            "close": [1.0, 1.0, 1.0],
            "volume": [1.0, 1.0, 1.0],
        }
    )
    df = normalize_frame(raw)
    assert list(df["symbol"]) == ["A", "B", "B"]
    assert df.loc[1, "timestamp"] == pd.Timestamp("2023-01-03")


def test_normalize_frame_missing_column_raises() -> None:
    raw = pd.DataFrame({"timestamp": ["2023-01-03"], "open": [1.0], "close": [1.0]})
    with pytest.raises(ValueError, match="missing required columns"):
        normalize_frame(raw, symbol="X")


def test_normalize_frame_missing_symbol_raises() -> None:
    raw = pd.DataFrame(
        {"timestamp": ["2023-01-03"], "open": [1.0], "high": [1.0], "low": [1.0],
         "close": [1.0], "volume": [1.0]}
    )
    with pytest.raises(ValueError, match="no symbol column"):
        normalize_frame(raw)


def test_normalize_frame_keeps_optional_columns() -> None:
    raw = pd.DataFrame(
        {
            "timestamp": ["2023-01-03"],
            "symbol": ["A"],
            "open": [1.0],
            "high": [1.0],
            "low": [1.0],
            "close": [1.0],
            "volume": [1.0],
            "vw": [1.2],
            "n": [42],
        }
    )
    df = normalize_frame(raw)
    assert df.loc[0, "vwap"] == 1.2
    assert df.loc[0, "trade_count"] == 42
