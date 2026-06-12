"""Synthetic market data: bars and deterministic intraday/daily feed generators."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Bar:
    """One OHLCV bar."""

    index: int
    open: float
    high: float
    low: float
    close: float
    volume: float

    def __post_init__(self) -> None:
        if not (self.low <= self.open <= self.high and self.low <= self.close <= self.high):
            raise ValueError(f"inconsistent OHLC for bar {self.index}")
        if self.volume < 0:
            raise ValueError("volume must be non-negative")


class PriceFeed:
    """Per-symbol bar series sharing a common clock (bar index)."""

    def __init__(self, bars: dict[str, list[Bar]]):
        if not bars:
            raise ValueError("feed requires at least one symbol")
        lengths = {len(v) for v in bars.values()}
        if len(lengths) != 1:
            raise ValueError("all symbols must have the same number of bars")
        self._bars = bars
        self.n_bars = lengths.pop()

    @property
    def symbols(self) -> list[str]:
        return list(self._bars)

    def bar(self, symbol: str, index: int) -> Bar:
        return self._bars[symbol][index]

    def bars(self, symbol: str) -> list[Bar]:
        return self._bars[symbol]

    def closes(self, symbol: str) -> np.ndarray:
        return np.array([b.close for b in self._bars[symbol]])

    def vwap(self, symbol: str, start: int, end: int) -> float:
        """Volume-weighted average close over bars [start, end)."""
        sel = self._bars[symbol][start:end]
        vol = sum(b.volume for b in sel)
        if vol == 0:
            return float(np.mean([b.close for b in sel]))
        return sum(b.close * b.volume for b in sel) / vol


def u_shaped_volume_profile(n_bars: int, base: float = 1.0, edge_boost: float = 2.5) -> np.ndarray:
    """Classic intraday U-shape: heavy at the open/close, light midday."""
    x = np.linspace(-1.0, 1.0, n_bars)
    profile = base + edge_boost * x**2
    return profile / profile.sum()


def synthetic_intraday_feed(
    symbol: str = "SYNTH",
    n_bars: int = 390,
    start_price: float = 100.0,
    annual_vol: float = 0.25,
    drift_bps_per_bar: float = 0.0,
    base_bar_volume: float = 20_000.0,
    seed: int = 7,
) -> PriceFeed:
    """Generate one deterministic synthetic trading day of 1-minute bars.

    Prices follow a discretised GBM; volumes follow a noisy U-shaped profile.
    """
    rng = np.random.default_rng(seed)
    bar_vol = annual_vol / np.sqrt(252.0 * n_bars)
    rets = rng.normal(drift_bps_per_bar / 1e4, bar_vol, size=n_bars)
    closes = start_price * np.exp(np.cumsum(rets))

    profile = u_shaped_volume_profile(n_bars)
    volumes = base_bar_volume * n_bars * profile * rng.uniform(0.6, 1.4, size=n_bars)

    bars: list[Bar] = []
    prev_close = start_price
    for i in range(n_bars):
        o = prev_close
        c = float(closes[i])
        wiggle = abs(rng.normal(0.0, bar_vol * prev_close * 0.5))
        hi = max(o, c) + wiggle
        lo = min(o, c) - wiggle
        bars.append(
            Bar(index=i, open=o, high=hi, low=max(lo, 0.01), close=c, volume=float(volumes[i]))
        )
        prev_close = c
    return PriceFeed({symbol: bars})
