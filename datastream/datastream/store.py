"""Parquet-backed local store, partitioned by symbol, with an in-memory read cache."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import pandas as pd

from datastream.schema import CANONICAL_COLUMNS, normalize_frame


class ParquetStore:
    """Stores canonical bars under ``root/symbol=<SYMBOL>/data.parquet``.

    Writes upsert on timestamp (new rows win). Reads are served from an LRU
    cache that is invalidated per symbol on write.
    """

    def __init__(self, root: str | Path, cache_size: int = 256) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        if cache_size < 0:
            raise ValueError("cache_size must be >= 0")
        self.cache_size = cache_size
        self._cache: OrderedDict[str, pd.DataFrame] = OrderedDict()
        self.cache_hits = 0
        self.cache_misses = 0

    def path_for(self, symbol: str) -> Path:
        return self.root / f"symbol={symbol.upper()}" / "data.parquet"

    def symbols(self) -> list[str]:
        out = []
        for child in sorted(self.root.glob("symbol=*")):
            if (child / "data.parquet").exists():
                out.append(child.name.split("=", 1)[1])
        return out

    def has(self, symbol: str) -> bool:
        return self.path_for(symbol).exists()

    # -- writes ---------------------------------------------------------------

    def write(self, df: pd.DataFrame, symbol: str | None = None) -> dict[str, int]:
        """Upsert a canonical (or normalizable) frame; returns rows written per symbol."""
        df = normalize_frame(df, symbol=symbol)
        written: dict[str, int] = {}
        for sym, group in df.groupby("symbol", sort=True):
            path = self.path_for(str(sym))
            path.parent.mkdir(parents=True, exist_ok=True)
            new = group.reset_index(drop=True)
            if path.exists():
                existing = pd.read_parquet(path)
                merged = pd.concat([existing, new], ignore_index=True)
                merged = merged.drop_duplicates(subset=["timestamp"], keep="last")
                new = merged.sort_values("timestamp", kind="stable").reset_index(drop=True)
            new.to_parquet(path, index=False, compression="zstd")
            self._cache.pop(str(sym), None)
            written[str(sym)] = len(new)
        return written

    # -- reads ----------------------------------------------------------------

    def read(
        self,
        symbol: str,
        start: str | pd.Timestamp | None = None,
        end: str | pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        symbol = symbol.upper()
        cached = self._cache.get(symbol)
        if cached is not None:
            self._cache.move_to_end(symbol)
            self.cache_hits += 1
            df = cached
        else:
            self.cache_misses += 1
            path = self.path_for(symbol)
            if not path.exists():
                raise KeyError(f"symbol not in store: {symbol}")
            df = pd.read_parquet(path)
            if self.cache_size > 0:
                self._cache[symbol] = df
                while len(self._cache) > self.cache_size:
                    self._cache.popitem(last=False)
        if start is not None:
            df = df[df["timestamp"] >= pd.Timestamp(start)]
        if end is not None:
            df = df[df["timestamp"] <= pd.Timestamp(end)]
        return df.reset_index(drop=True).copy()

    def read_many(
        self,
        symbols: list[str] | None = None,
        start: str | pd.Timestamp | None = None,
        end: str | pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        symbols = self.symbols() if symbols is None else [s.upper() for s in symbols]
        frames = [self.read(sym, start, end) for sym in symbols]
        if not frames:
            return pd.DataFrame(columns=CANONICAL_COLUMNS)
        return pd.concat(frames, ignore_index=True)

    # -- maintenance ----------------------------------------------------------

    def clear_cache(self) -> None:
        self._cache.clear()

    def stats(self) -> dict:
        files = list(self.root.rglob("*.parquet"))
        total = sum(f.stat().st_size for f in files)
        return {
            "symbols": len(self.symbols()),
            "file_count": len(files),
            "total_bytes": total,
            "total_mb": total / (1024 * 1024),
            "cache_entries": len(self._cache),
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
        }
