"""Factor base class and the built-in factor library (19 factors).

Sign convention: factor values are oriented so that *higher is expected to
be better* under the classic anomaly (e.g. low-volatility factors return
``-vol``). Evaluation is sign-aware anyway (IC can be negative).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from alphalab import ops
from alphalab.panel import Panel


class Factor(ABC):
    """A factor maps a :class:`Panel` to a wide (date x symbol) value frame."""

    name: str = "factor"
    category: str = "custom"
    description: str = ""

    def compute(self, panel: Panel) -> pd.DataFrame:
        """Compute factor values; infinities are scrubbed to NaN."""
        return ops.clean(self._compute(panel))

    @abstractmethod
    def _compute(self, panel: Panel) -> pd.DataFrame: ...

    def __call__(self, panel: Panel) -> pd.DataFrame:
        return self.compute(panel)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r}, category={self.category!r})"


# -- momentum / reversal -------------------------------------------------------


class Momentum(Factor):
    """Total return over ``window`` days, optionally skipping the most recent
    ``skip`` days (the classic 12-1 momentum uses window=231, skip=21)."""

    category = "momentum"

    def __init__(self, window: int, skip: int = 0) -> None:
        if window < 1 or skip < 0:
            raise ValueError("window must be >= 1 and skip >= 0")
        self.window, self.skip = window, skip
        self.name = f"momentum_{window}d" if skip == 0 else f"momentum_{window}d_skip{skip}d"
        self.description = f"{window}-day price momentum, skipping last {skip} days"

    def _compute(self, panel: Panel) -> pd.DataFrame:
        shifted = panel.close.shift(self.skip)
        return shifted / shifted.shift(self.window) - 1.0


class ShortTermReversal(Factor):
    category = "reversal"

    def __init__(self, window: int = 5) -> None:
        self.window = window
        self.name = f"reversal_{window}d"
        self.description = f"Negative {window}-day return (short-term mean reversion)"

    def _compute(self, panel: Panel) -> pd.DataFrame:
        return -(panel.close / panel.close.shift(self.window) - 1.0)


# -- volatility ----------------------------------------------------------------


class LowVolatility(Factor):
    category = "volatility"

    def __init__(self, window: int = 20) -> None:
        self.window = window
        self.name = f"low_volatility_{window}d"
        self.description = f"Negative {window}-day realized volatility of daily returns"

    def _compute(self, panel: Panel) -> pd.DataFrame:
        return -ops.ts_std(panel.returns, self.window)


class LowDownsideVolatility(Factor):
    category = "volatility"

    def __init__(self, window: int = 20) -> None:
        self.window = window
        self.name = f"low_downside_vol_{window}d"
        self.description = f"Negative {window}-day semi-deviation (negative returns only)"

    def _compute(self, panel: Panel) -> pd.DataFrame:
        downside = panel.returns.where(panel.returns < 0)
        return -downside.rolling(self.window, min_periods=max(2, self.window // 4)).std()


class PriceRange(Factor):
    category = "volatility"

    def __init__(self, window: int = 20) -> None:
        self.window = window
        self.name = f"price_range_{window}d"
        self.description = f"{window}-day high-low range as a fraction of close"

    def _compute(self, panel: Panel) -> pd.DataFrame:
        rng = ops.ts_max(panel.high, self.window) - ops.ts_min(panel.low, self.window)
        return -(rng / panel.close)


# -- volume / liquidity ----------------------------------------------------------


class VolumeMomentum(Factor):
    category = "volume"

    def __init__(self, fast: int = 5, slow: int = 20) -> None:
        if fast >= slow:
            raise ValueError("fast window must be shorter than slow window")
        self.fast, self.slow = fast, slow
        self.name = f"volume_momentum_{fast}_{slow}d"
        self.description = f"{fast}d vs {slow}d average volume ratio minus 1"

    def _compute(self, panel: Panel) -> pd.DataFrame:
        return ops.ts_mean(panel.volume, self.fast) / ops.ts_mean(panel.volume, self.slow) - 1.0


class AbnormalVolume(Factor):
    category = "volume"

    def __init__(self, window: int = 20) -> None:
        self.window = window
        self.name = f"abnormal_volume_{window}d"
        self.description = f"Volume z-score vs trailing {window}-day distribution"

    def _compute(self, panel: Panel) -> pd.DataFrame:
        mean = ops.ts_mean(panel.volume, self.window)
        std = ops.ts_std(panel.volume, self.window)
        return (panel.volume - mean) / std


class AmihudIlliquidity(Factor):
    category = "liquidity"

    def __init__(self, window: int = 20) -> None:
        self.window = window
        self.name = f"amihud_illiquidity_{window}d"
        self.description = (
            f"{window}-day mean of |return| per unit dollar volume (x1e6); "
            "illiquidity premium"
        )

    def _compute(self, panel: Panel) -> pd.DataFrame:
        dollar_volume = panel.close * panel.volume
        daily = panel.returns.abs() / dollar_volume * 1e6
        return ops.ts_mean(ops.clean(daily), self.window)


# -- price position / technical ---------------------------------------------------


class High52WeekDistance(Factor):
    category = "technical"

    def __init__(self, window: int = 252) -> None:
        self.window = window
        self.name = f"high_{window}d_distance"
        self.description = f"Close relative to trailing {window}-day high, minus 1"

    def _compute(self, panel: Panel) -> pd.DataFrame:
        return panel.close / ops.ts_max(panel.high, self.window) - 1.0


class OvernightGap(Factor):
    category = "technical"

    def __init__(self, window: int = 5) -> None:
        self.window = window
        self.name = f"overnight_gap_{window}d"
        self.description = f"{window}-day average overnight (close-to-open) return"

    def _compute(self, panel: Panel) -> pd.DataFrame:
        gap = panel.open / panel.close.shift(1) - 1.0
        return ops.ts_mean(gap, self.window)


# -- correlation / beta -----------------------------------------------------------


class LowBeta(Factor):
    """Betting-against-beta: negative rolling beta vs the equal-weight market."""

    category = "beta"

    def __init__(self, window: int = 60) -> None:
        self.window = window
        self.name = f"low_beta_{window}d"
        self.description = f"Negative {window}-day beta vs equal-weight universe return"

    def _compute(self, panel: Panel) -> pd.DataFrame:
        rets = panel.returns
        market = rets.mean(axis=1)
        market_frame = pd.DataFrame(
            np.tile(market.to_numpy()[:, None], (1, rets.shape[1])),
            index=rets.index,
            columns=rets.columns,
        )
        cov = ops.ts_cov(rets, market_frame, self.window)
        var = market.rolling(self.window).var()
        return -cov.div(var, axis=0)


class MarketCorrelation(Factor):
    category = "beta"

    def __init__(self, window: int = 20) -> None:
        self.window = window
        self.name = f"market_corr_{window}d"
        self.description = f"Negative {window}-day correlation with equal-weight market"

    def _compute(self, panel: Panel) -> pd.DataFrame:
        rets = panel.returns
        market = rets.mean(axis=1)
        market_frame = pd.DataFrame(
            np.tile(market.to_numpy()[:, None], (1, rets.shape[1])),
            index=rets.index,
            columns=rets.columns,
        )
        return -ops.ts_corr(rets, market_frame, self.window)


# -- Alpha101 formulaic alphas -----------------------------------------------------


class Alpha003(Factor):
    """alpha#3: ``-1 * correlation(rank(open), rank(volume), 10)``"""

    category = "alpha101"
    name = "alpha_003"
    description = "-corr(rank(open), rank(volume), 10)"

    def _compute(self, panel: Panel) -> pd.DataFrame:
        return -ops.ts_corr(ops.cs_rank(panel.open), ops.cs_rank(panel.volume), 10)


class Alpha006(Factor):
    """alpha#6: ``-1 * correlation(open, volume, 10)``"""

    category = "alpha101"
    name = "alpha_006"
    description = "-corr(open, volume, 10)"

    def _compute(self, panel: Panel) -> pd.DataFrame:
        return -ops.ts_corr(panel.open, panel.volume, 10)


class Alpha012(Factor):
    """alpha#12: ``sign(delta(volume, 1)) * (-1 * delta(close, 1))``"""

    category = "alpha101"
    name = "alpha_012"
    description = "sign(delta(volume,1)) * -delta(close,1)"

    def _compute(self, panel: Panel) -> pd.DataFrame:
        return np.sign(ops.delta(panel.volume, 1)) * (-ops.delta(panel.close, 1))


class Alpha053(Factor):
    """alpha#53: ``-1 * delta(((close - low) - (high - close)) / (close - low), 9)``"""

    category = "alpha101"
    name = "alpha_053"
    description = "-delta(((close-low)-(high-close))/(close-low), 9)"

    def _compute(self, panel: Panel) -> pd.DataFrame:
        inner = ((panel.close - panel.low) - (panel.high - panel.close)) / (
            panel.close - panel.low
        )
        return -ops.delta(ops.clean(inner), 9)


class Alpha101(Factor):
    """alpha#101: ``(close - open) / ((high - low) + 0.001)``"""

    category = "alpha101"
    name = "alpha_101"
    description = "(close - open) / (high - low + 0.001)"

    def _compute(self, panel: Panel) -> pd.DataFrame:
        return (panel.close - panel.open) / ((panel.high - panel.low) + 0.001)


def default_factors() -> list[Factor]:
    """The built-in factor zoo (19 factors across 8 style categories)."""
    return [
        Momentum(21),
        Momentum(63),
        Momentum(126),
        Momentum(231, skip=21),
        ShortTermReversal(5),
        LowVolatility(20),
        LowDownsideVolatility(20),
        PriceRange(20),
        VolumeMomentum(5, 20),
        AbnormalVolume(20),
        AmihudIlliquidity(20),
        High52WeekDistance(252),
        OvernightGap(5),
        LowBeta(60),
        MarketCorrelation(20),
        Alpha003(),
        Alpha006(),
        Alpha012(),
        Alpha053(),
        Alpha101(),
    ]
