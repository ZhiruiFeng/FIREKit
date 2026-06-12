"""OHLCV bar schema and frame normalization.

The canonical long-format frame has columns::

    timestamp (datetime64[ns]), symbol (str),
    open, high, low, close, volume (float64)

plus optional ``vwap`` (float64) and ``trade_count`` (Int64).
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime

import pandas as pd
from pydantic import BaseModel, ConfigDict, field_validator

CANONICAL_COLUMNS: list[str] = ["timestamp", "symbol", "open", "high", "low", "close", "volume"]
OPTIONAL_COLUMNS: list[str] = ["vwap", "trade_count"]

#: Lowercased source-column aliases -> canonical name.
COLUMN_ALIASES: dict[str, str] = {
    "timestamp": "timestamp",
    "date": "timestamp",
    "datetime": "timestamp",
    "time": "timestamp",
    "t": "timestamp",
    "symbol": "symbol",
    "ticker": "symbol",
    "sym": "symbol",
    "open": "open",
    "o": "open",
    "high": "high",
    "h": "high",
    "low": "low",
    "l": "low",
    "close": "close",
    "c": "close",
    "adj_close": "close",
    "volume": "volume",
    "v": "volume",
    "vol": "volume",
    "vwap": "vwap",
    "vw": "vwap",
    "trade_count": "trade_count",
    "n": "trade_count",
    "trades": "trade_count",
}


class OHLCVBar(BaseModel):
    """A single normalized OHLCV bar."""

    model_config = ConfigDict(frozen=True)

    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    vwap: float | None = None
    trade_count: int | None = None

    @field_validator("symbol")
    @classmethod
    def _normalize_symbol(cls, value: str) -> str:
        value = value.strip().upper()
        if not value:
            raise ValueError("symbol must be non-empty")
        return value

    def to_dict(self) -> dict:
        return self.model_dump()


def bars_to_frame(bars: Iterable[OHLCVBar]) -> pd.DataFrame:
    """Convert an iterable of :class:`OHLCVBar` into a canonical long frame."""
    rows = [bar.to_dict() for bar in bars]
    if not rows:
        return empty_frame()
    return normalize_frame(pd.DataFrame(rows))


def empty_frame() -> pd.DataFrame:
    """Return an empty canonical frame."""
    df = pd.DataFrame({col: pd.Series(dtype="float64") for col in CANONICAL_COLUMNS})
    df["timestamp"] = pd.Series(dtype="datetime64[ns]")
    df["symbol"] = pd.Series(dtype="str")
    return df[CANONICAL_COLUMNS]


def normalize_frame(df: pd.DataFrame, symbol: str | None = None) -> pd.DataFrame:
    """Normalize an arbitrary OHLCV-like frame to the canonical schema.

    - Maps common column aliases (``Date``, ``o``, ``Adj_Close``, ...).
    - Promotes a ``DatetimeIndex`` to the ``timestamp`` column.
    - Fills ``symbol`` from the argument when the column is missing.
    - Coerces dtypes and sorts by ``(symbol, timestamp)``.

    Rows are *not* deduplicated or repaired here; that is the quality
    engine's job, so issues stay observable.
    """
    df = df.copy()
    if isinstance(df.index, pd.DatetimeIndex) and "timestamp" not in (
        str(c).strip().lower() for c in df.columns
    ):
        df = df.reset_index(names="timestamp")

    rename: dict[str, str] = {}
    for col in df.columns:
        alias = COLUMN_ALIASES.get(str(col).strip().lower())
        if alias is not None and alias not in rename.values():
            rename[col] = alias
    df = df.rename(columns=rename)

    if "symbol" not in df.columns:
        if symbol is None:
            raise ValueError("frame has no symbol column and no symbol was provided")
        df["symbol"] = symbol

    missing = [c for c in CANONICAL_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"missing required columns after normalization: {missing}")

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["symbol"] = df["symbol"].astype("str").str.strip().str.upper()
    for col in ["open", "high", "low", "close", "volume", "vwap"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
    if "trade_count" in df.columns:
        df["trade_count"] = pd.to_numeric(df["trade_count"], errors="coerce").astype("Int64")

    keep = CANONICAL_COLUMNS + [c for c in OPTIONAL_COLUMNS if c in df.columns]
    df = df[keep]
    return df.sort_values(["symbol", "timestamp"], kind="stable").reset_index(drop=True)
