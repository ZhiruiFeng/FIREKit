"""Panel container (date x symbol wide frames) and a seeded synthetic generator."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

PANEL_FIELDS = ("open", "high", "low", "close", "volume")


@dataclass
class Panel:
    """Aligned OHLCV panels: each field is a DataFrame with a DatetimeIndex
    (dates) and one column per symbol."""

    open: pd.DataFrame
    high: pd.DataFrame
    low: pd.DataFrame
    close: pd.DataFrame
    volume: pd.DataFrame
    _returns: pd.DataFrame | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        ref = self.close
        if not isinstance(ref.index, pd.DatetimeIndex):
            raise TypeError("panel index must be a DatetimeIndex")
        if not ref.index.is_monotonic_increasing:
            raise ValueError("panel index must be sorted ascending")
        for name in PANEL_FIELDS:
            frame = getattr(self, name)
            if not frame.index.equals(ref.index) or not frame.columns.equals(ref.columns):
                raise ValueError(f"panel field '{name}' is not aligned with 'close'")

    @property
    def symbols(self) -> list[str]:
        return list(self.close.columns)

    @property
    def dates(self) -> pd.DatetimeIndex:
        return self.close.index

    @property
    def shape(self) -> tuple[int, int]:
        return self.close.shape

    @property
    def vwap(self) -> pd.DataFrame:
        return (self.high + self.low + self.close) / 3.0

    @property
    def returns(self) -> pd.DataFrame:
        """Daily close-to-close simple returns (cached)."""
        if self._returns is None:
            self._returns = self.close.pct_change()
        return self._returns

    @classmethod
    def from_long(cls, df: pd.DataFrame) -> Panel:
        """Build a panel from a long frame with columns
        ``timestamp, symbol, open, high, low, close, volume``."""
        missing = {"timestamp", "symbol", *PANEL_FIELDS} - set(df.columns)
        if missing:
            raise ValueError(f"long frame missing columns: {sorted(missing)}")
        wide = {
            name: df.pivot_table(index="timestamp", columns="symbol", values=name).sort_index()
            for name in PANEL_FIELDS
        }
        frames = {name: w.astype("float64") for name, w in wide.items()}
        frames = {
            name: w.set_axis(pd.DatetimeIndex(w.index).rename(None), axis=0).rename_axis(
                columns=None
            )
            for name, w in frames.items()
        }
        return cls(**frames)


def make_synthetic_panel(
    n_symbols: int = 50,
    n_days: int = 756,
    seed: int = 7,
    start: str = "2021-01-04",
) -> Panel:
    """Seeded synthetic equity panel with realistic cross-sectional structure.

    Returns are generated as ``r_it = mu_i + beta_i * f_t + e_it + theta * e_i,t-1``:

    - per-symbol persistent drift ``mu_i`` (cross-sectional momentum is real),
    - a common market factor ``f_t`` with per-symbol beta,
    - an MA(1) noise term with negative ``theta`` (short-term reversal is real),
    - volume that co-moves with absolute returns (illiquidity is measurable).
    """
    if n_symbols < 2 or n_days < 30:
        raise ValueError("need at least 2 symbols and 30 days")
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=n_days)
    symbols = [f"SYM{i:03d}" for i in range(1, n_symbols + 1)]

    mu = rng.normal(0.0004, 0.0012, size=n_symbols)
    beta = rng.uniform(0.4, 1.6, size=n_symbols)
    sigma = rng.uniform(0.006, 0.018, size=n_symbols)
    theta = -0.22

    market = rng.normal(0.0002, 0.006, size=n_days)
    eps = rng.normal(0.0, 1.0, size=(n_days, n_symbols)) * sigma
    rets = mu + np.outer(market, beta) + eps
    rets[1:] += theta * eps[:-1]

    base = rng.uniform(10.0, 200.0, size=n_symbols)
    close = base * np.exp(np.cumsum(np.log1p(np.clip(rets, -0.5, 0.5)), axis=0))

    gap = rng.normal(0.0, 0.002, size=(n_days, n_symbols))
    open_ = np.empty_like(close)
    open_[0] = base
    open_[1:] = close[:-1] * (1.0 + gap[1:])

    intraday = np.abs(rng.normal(0.0, 0.006, size=(n_days, n_symbols)))
    high = np.maximum(open_, close) * (1.0 + intraday)
    low = np.minimum(open_, close) * (1.0 - intraday)

    base_volume = rng.uniform(2e5, 5e6, size=n_symbols)
    vol_noise = rng.lognormal(0.0, 0.4, size=(n_days, n_symbols))
    volume = base_volume * vol_noise * (1.0 + 25.0 * np.abs(rets))

    def frame(arr: np.ndarray) -> pd.DataFrame:
        return pd.DataFrame(arr, index=dates, columns=symbols)

    return Panel(
        open=frame(open_),
        high=frame(high),
        low=frame(low),
        close=frame(close),
        volume=frame(np.round(volume)),
    )
