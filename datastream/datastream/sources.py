"""Data sources: a common loader interface plus synthetic and file-based sources."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import pandas as pd

from datastream.schema import normalize_frame

#: Regime name -> (annualized drift, annualized volatility) for the GBM generator.
REGIMES: dict[str, tuple[float, float]] = {
    "bull": (0.18, 0.16),
    "bear": (-0.22, 0.30),
    "sideways": (0.02, 0.12),
    "volatile": (0.06, 0.42),
}

TRADING_DAYS_PER_YEAR = 252


class BaseSource(ABC):
    """Abstract loader. Implementations return canonical long-format frames."""

    @abstractmethod
    def list_symbols(self) -> list[str]:
        """All symbols this source can serve."""

    @abstractmethod
    def fetch(
        self,
        symbols: list[str] | None = None,
        start: str | pd.Timestamp | None = None,
        end: str | pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        """Fetch bars as a canonical long frame (see :mod:`datastream.schema`)."""

    @staticmethod
    def _clip_dates(
        df: pd.DataFrame,
        start: str | pd.Timestamp | None,
        end: str | pd.Timestamp | None,
    ) -> pd.DataFrame:
        if start is not None:
            df = df[df["timestamp"] >= pd.Timestamp(start)]
        if end is not None:
            df = df[df["timestamp"] <= pd.Timestamp(end)]
        return df.reset_index(drop=True)


class SyntheticSource(BaseSource):
    """Seeded geometric-Brownian-motion universe generator.

    Each symbol traverses the configured ``regimes`` in order (the date range
    is split into equal segments). With ``issue_rate > 0`` the generator
    deliberately corrupts a deterministic fraction of rows (duplicate
    timestamps, gaps, non-positive prices, OHLC violations, return outliers)
    so the quality engine has something to find.
    """

    def __init__(
        self,
        symbols: list[str] | None = None,
        n_symbols: int = 20,
        start: str = "2021-01-04",
        end: str = "2023-12-29",
        seed: int = 42,
        regimes: tuple[str, ...] = ("bull", "volatile", "bear", "sideways"),
        issue_rate: float = 0.0,
    ) -> None:
        if symbols is None:
            symbols = [f"SYM{i:03d}" for i in range(1, n_symbols + 1)]
        unknown = [r for r in regimes if r not in REGIMES]
        if unknown:
            raise ValueError(f"unknown regimes: {unknown}; choose from {sorted(REGIMES)}")
        if not 0.0 <= issue_rate <= 0.2:
            raise ValueError("issue_rate must be in [0, 0.2]")
        self.symbols = list(symbols)
        self.start = pd.Timestamp(start)
        self.end = pd.Timestamp(end)
        self.seed = seed
        self.regimes = tuple(regimes)
        self.issue_rate = issue_rate
        self._dates = pd.bdate_range(self.start, self.end)
        if len(self._dates) < 30:
            raise ValueError("date range must span at least 30 business days")

    def list_symbols(self) -> list[str]:
        return list(self.symbols)

    def fetch(
        self,
        symbols: list[str] | None = None,
        start: str | pd.Timestamp | None = None,
        end: str | pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        wanted = self.symbols if symbols is None else list(symbols)
        unknown = sorted(set(wanted) - set(self.symbols))
        if unknown:
            raise KeyError(f"symbols not in universe: {unknown}")
        frames = [self._generate_symbol(sym) for sym in wanted]
        df = pd.concat(frames, ignore_index=True)
        df = normalize_frame(df)
        return self._clip_dates(df, start, end)

    # -- generation internals -------------------------------------------------

    def _regime_path(self, rng: np.random.Generator, n: int) -> np.ndarray:
        """Daily log-returns following the regime sequence."""
        bounds = np.linspace(0, n, len(self.regimes) + 1).astype(int)
        out = np.empty(n)
        for i, regime in enumerate(self.regimes):
            mu, sigma = REGIMES[regime]
            seg = slice(bounds[i], bounds[i + 1])
            length = bounds[i + 1] - bounds[i]
            daily_mu = mu / TRADING_DAYS_PER_YEAR
            daily_sigma = sigma / np.sqrt(TRADING_DAYS_PER_YEAR)
            out[seg] = rng.normal(daily_mu - 0.5 * daily_sigma**2, daily_sigma, size=length)
        return out

    def _generate_symbol(self, symbol: str) -> pd.DataFrame:
        idx = self.symbols.index(symbol)
        rng = np.random.default_rng([self.seed, idx])
        n = len(self._dates)

        base_price = float(rng.uniform(8.0, 250.0))
        log_ret = self._regime_path(rng, n)
        close = base_price * np.exp(np.cumsum(log_ret))

        gap = rng.normal(0.0, 0.002, size=n)
        open_ = np.empty(n)
        open_[0] = base_price
        open_[1:] = close[:-1] * (1.0 + gap[1:])

        intraday = np.abs(rng.normal(0.0, 0.008, size=n))
        high = np.maximum(open_, close) * (1.0 + intraday)
        low = np.minimum(open_, close) * (1.0 - intraday)

        base_volume = float(rng.uniform(2e5, 5e6))
        volume = base_volume * rng.lognormal(0.0, 0.45, size=n)
        vwap = (high + low + close) / 3.0

        df = pd.DataFrame(
            {
                "timestamp": self._dates,
                "symbol": symbol,
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": np.round(volume),
                "vwap": vwap,
            }
        )
        if self.issue_rate > 0:
            df = self._inject_issues(df, rng)
        return df

    def _inject_issues(self, df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
        """Corrupt a deterministic sample of rows (never the first 5)."""
        n = len(df)
        n_bad = max(1, int(round(n * self.issue_rate)))
        candidates = rng.choice(np.arange(5, n), size=min(n_bad * 5, n - 5), replace=False)
        kinds = ["nonpositive", "ohlc", "outlier", "duplicate", "gap"]
        drop: list[int] = []
        extra_rows: list[pd.Series] = []
        for k, pos in enumerate(candidates[:n_bad]):
            kind = kinds[k % len(kinds)]
            if kind == "nonpositive":
                df.loc[pos, "close"] = -abs(df.loc[pos, "close"])
            elif kind == "ohlc":
                df.loc[pos, "high"] = df.loc[pos, "low"] * 0.95
            elif kind == "outlier":
                df.loc[pos, ["close", "high"]] = df.loc[pos, ["close", "high"]] * 4.0
            elif kind == "duplicate":
                extra_rows.append(df.loc[pos].copy())
            elif kind == "gap":
                drop.extend(range(pos, min(pos + 5, n)))
        if drop:
            df = df.drop(index=sorted(set(drop)))
        if extra_rows:
            df = pd.concat([df, pd.DataFrame(extra_rows)], ignore_index=True)
        return df.reset_index(drop=True)


class FileSource(BaseSource):
    """Loads CSV or Parquet files from a single file or a directory.

    Directory mode treats each ``<SYMBOL>.csv`` / ``<SYMBOL>.parquet`` file as
    one symbol; single-file mode expects a symbol column (or pass ``symbol=``).
    """

    SUPPORTED = {".csv", ".parquet"}

    def __init__(self, path: str | Path, symbol: str | None = None) -> None:
        self.path = Path(path)
        self.symbol = symbol
        if not self.path.exists():
            raise FileNotFoundError(self.path)
        if self.path.is_file() and self.path.suffix.lower() not in self.SUPPORTED:
            raise ValueError(f"unsupported file type: {self.path.suffix}")

    def _files(self) -> list[Path]:
        if self.path.is_file():
            return [self.path]
        return sorted(
            p for p in self.path.iterdir() if p.suffix.lower() in self.SUPPORTED and p.is_file()
        )

    @staticmethod
    def _read(path: Path) -> pd.DataFrame:
        if path.suffix.lower() == ".csv":
            return pd.read_csv(path)
        return pd.read_parquet(path)

    def list_symbols(self) -> list[str]:
        if self.path.is_file():
            df = self.fetch()
            return sorted(df["symbol"].unique().tolist())
        return sorted(p.stem.upper() for p in self._files())

    def fetch(
        self,
        symbols: list[str] | None = None,
        start: str | pd.Timestamp | None = None,
        end: str | pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        for file in self._files():
            raw = self._read(file)
            fallback = self.symbol if self.path.is_file() else file.stem.upper()
            frames.append(normalize_frame(raw, symbol=fallback))
        if not frames:
            raise FileNotFoundError(f"no CSV/Parquet files found under {self.path}")
        df = pd.concat(frames, ignore_index=True)
        df = normalize_frame(df)
        if symbols is not None:
            df = df[df["symbol"].isin([s.upper() for s in symbols])].reset_index(drop=True)
        return self._clip_dates(df, start, end)
