"""
Cross-Sectional Signals Contract
================================

API contract for signal generation across asset universe.
These stubs define the public interface; implementations follow.

Contract Tests: tests/contract/test_signals_contract.py
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Callable

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from vectorforge.portfolio.data import PortfolioData


class RankMethod(Enum):
    """Methods for cross-sectional ranking."""

    PERCENTILE = "percentile"  # 0-100 percentile rank
    FRACTIONAL = "fractional"  # 0-1 fractional rank
    ORDINAL = "ordinal"  # 1-N ordinal rank


@dataclass
class SignalResult:
    """
    Output of cross-sectional signal generation.

    Attributes:
        values: Raw signal values (n_symbols × n_dates)
        ranks: Cross-sectional ranks (n_symbols × n_dates)
        symbols: Symbol order matching array rows
        dates: Date order matching array columns
    """

    values: np.ndarray
    ranks: np.ndarray
    symbols: list[str]
    dates: pd.DatetimeIndex

    def top_percentile(self, n: int) -> SignalResult:
        """
        Filter to top n percentile of assets.

        Args:
            n: Percentile threshold (e.g., 10 for top 10%)

        Returns:
            New SignalResult with only top n% assets
        """
        ...

    def bottom_percentile(self, n: int) -> SignalResult:
        """Filter to bottom n percentile of assets."""
        ...

    def between_percentile(self, low: int, high: int) -> SignalResult:
        """Filter to assets between low and high percentile."""
        ...

    def to_weights(
        self,
        method: str = "equal",
        normalize: bool = True,
    ) -> TargetWeights:
        """
        Convert signals to target weights.

        Args:
            method: Weighting method ("equal", "signal_weighted", "market_cap")
            normalize: Whether to normalize weights to sum to 1

        Returns:
            TargetWeights for rebalancing
        """
        ...


@dataclass
class TargetWeights:
    """
    Portfolio target allocation.

    Attributes:
        weights: Target weights (n_symbols × n_dates), NaN for excluded
        symbols: Symbol order
        dates: Rebalance dates
    """

    weights: np.ndarray
    symbols: list[str]
    dates: pd.DatetimeIndex

    @property
    def n_symbols(self) -> int:
        """Number of symbols."""
        ...

    @property
    def n_dates(self) -> int:
        """Number of dates."""
        ...

    def get_weights_at(self, date: pd.Timestamp) -> dict[str, float]:
        """Get symbol → weight mapping for a specific date."""
        ...

    def validate(self) -> TargetWeights:
        """
        Validate weights (no NaN in active positions, reasonable sums).

        Returns:
            Self if valid

        Raises:
            ValueError: If validation fails
        """
        ...

    @classmethod
    def equal_weight(
        cls,
        symbols: list[str],
        dates: pd.DatetimeIndex,
        active_mask: np.ndarray | None = None,
    ) -> TargetWeights:
        """Create equal-weighted portfolio."""
        ...

    @classmethod
    def market_cap_weight(
        cls,
        portfolio_data: PortfolioData,
        dates: pd.DatetimeIndex,
    ) -> TargetWeights:
        """Create market-cap weighted portfolio."""
        ...


class CrossSectionalSignal(ABC):
    """
    Generator for cross-sectional (relative) signals.

    Example:
        >>> signal = CrossSectionalSignal.momentum(lookback=252)
        >>> result = signal.generate(portfolio_data)
        >>> weights = result.top_percentile(10).to_weights()
    """

    @property
    @abstractmethod
    def lookback(self) -> int:
        """Lookback period in days."""
        ...

    @property
    @abstractmethod
    def rank_method(self) -> RankMethod:
        """Ranking method used."""
        ...

    @abstractmethod
    def generate(
        self,
        data: PortfolioData,
        group_field: str | None = None,
    ) -> SignalResult:
        """
        Generate cross-sectional signals.

        Args:
            data: Portfolio data to generate signals from
            group_field: Optional field for sector-neutral ranking

        Returns:
            SignalResult with values and ranks
        """
        ...

    @classmethod
    @abstractmethod
    def momentum(
        cls,
        lookback: int = 252,
        skip_recent: int = 21,
        rank_method: RankMethod = RankMethod.PERCENTILE,
    ) -> CrossSectionalSignal:
        """
        Create momentum signal.

        Args:
            lookback: Total lookback period
            skip_recent: Days to skip for mean reversion avoidance
            rank_method: How to rank assets

        Returns:
            Configured momentum signal generator
        """
        ...

    @classmethod
    @abstractmethod
    def mean_reversion(
        cls,
        lookback: int = 20,
        rank_method: RankMethod = RankMethod.PERCENTILE,
    ) -> CrossSectionalSignal:
        """Create mean reversion (short-term reversal) signal."""
        ...

    @classmethod
    @abstractmethod
    def volatility(
        cls,
        lookback: int = 60,
        rank_method: RankMethod = RankMethod.PERCENTILE,
        inverse: bool = True,
    ) -> CrossSectionalSignal:
        """
        Create volatility signal.

        Args:
            lookback: Volatility calculation window
            rank_method: Ranking method
            inverse: If True, lower volatility ranks higher

        Returns:
            Configured volatility signal generator
        """
        ...

    @classmethod
    @abstractmethod
    def custom(
        cls,
        func: Callable[[np.ndarray], np.ndarray],
        lookback: int,
        name: str = "custom",
        rank_method: RankMethod = RankMethod.PERCENTILE,
    ) -> CrossSectionalSignal:
        """
        Create custom signal from user function.

        Args:
            func: Function (prices: ndarray) → signal values
            lookback: Required lookback period
            name: Signal name for display
            rank_method: Ranking method

        Returns:
            Configured custom signal generator
        """
        ...

    @classmethod
    @abstractmethod
    def relative_strength(
        cls,
        lookback: int = 252,
        benchmark_idx: int | None = None,
    ) -> CrossSectionalSignal:
        """
        Create relative strength signal.

        Compares each asset's momentum to universe average (or benchmark).

        Args:
            lookback: Momentum calculation period
            benchmark_idx: Optional index of benchmark symbol

        Returns:
            Relative strength signal generator
        """
        ...


class SignalCombiner(ABC):
    """
    Combine multiple signals into composite.

    Example:
        >>> momentum = CrossSectionalSignal.momentum(252)
        >>> volatility = CrossSectionalSignal.volatility(60)
        >>> combined = SignalCombiner.average([momentum, volatility])
    """

    @classmethod
    @abstractmethod
    def average(
        cls,
        signals: list[CrossSectionalSignal],
        weights: list[float] | None = None,
    ) -> CrossSectionalSignal:
        """
        Average multiple signals.

        Args:
            signals: Signals to combine
            weights: Optional weights (default: equal)

        Returns:
            Combined signal generator
        """
        ...

    @classmethod
    @abstractmethod
    def rank_average(
        cls,
        signals: list[CrossSectionalSignal],
    ) -> CrossSectionalSignal:
        """Average the ranks of multiple signals."""
        ...
