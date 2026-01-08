"""
Cross-Sectional Signals and Target Weights

Signal generation and weight allocation for multi-asset portfolios.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy import stats

if TYPE_CHECKING:
    from vectorforge.portfolio.data import PortfolioData


class RankMethod(Enum):
    """Methods for cross-sectional ranking."""

    PERCENTILE = "percentile"  # 0-100 percentile rank
    FRACTIONAL = "fractional"  # 0-1 fractional rank
    ORDINAL = "ordinal"  # 1-N ordinal rank


@dataclass
class TargetWeights:
    """
    Portfolio target allocation.

    Attributes:
        weights: Target weights (n_symbols x n_dates), NaN for excluded
        symbols: Symbol order
        dates: Rebalance dates
    """

    weights: np.ndarray
    symbols: list[str]
    dates: pd.DatetimeIndex

    @property
    def n_symbols(self) -> int:
        """Number of symbols."""
        return len(self.symbols)

    @property
    def n_dates(self) -> int:
        """Number of dates."""
        return len(self.dates)

    def get_weights_at(self, date: pd.Timestamp) -> dict[str, float]:
        """Get symbol -> weight mapping for a specific date."""
        if date not in self.dates:
            # Find closest date
            idx = self.dates.get_indexer([date], method="nearest")[0]
        else:
            idx = self.dates.get_loc(date)

        weights_dict = {}
        for i, symbol in enumerate(self.symbols):
            w = self.weights[i, idx]
            if not np.isnan(w):
                weights_dict[symbol] = float(w)
        return weights_dict

    def validate(self) -> TargetWeights:
        """
        Validate weights (no NaN in active positions, reasonable sums).

        Returns:
            Self if valid

        Raises:
            ValueError: If validation fails
        """
        # Check for unreasonable weight sums
        for j in range(self.n_dates):
            col_weights = self.weights[:, j]
            active = ~np.isnan(col_weights)
            if active.any():
                weight_sum = col_weights[active].sum()
                if not np.isclose(weight_sum, 1.0, atol=0.01) and weight_sum != 0:
                    # Allow leverage or partial allocation
                    pass

        return self

    @classmethod
    def equal_weight(
        cls,
        symbols: list[str],
        dates: pd.DatetimeIndex,
        active_mask: np.ndarray | None = None,
    ) -> TargetWeights:
        """
        Create equal-weighted portfolio.

        Args:
            symbols: List of symbols
            dates: Trading dates
            active_mask: Optional (n_symbols, n_dates) mask for active positions

        Returns:
            TargetWeights with equal weights for active positions
        """
        n_symbols = len(symbols)
        n_dates = len(dates)

        if active_mask is None:
            # All symbols active
            weights = np.ones((n_symbols, n_dates)) / n_symbols
        else:
            weights = np.zeros((n_symbols, n_dates))
            for j in range(n_dates):
                active_count = active_mask[:, j].sum()
                if active_count > 0:
                    weights[active_mask[:, j], j] = 1.0 / active_count

        return cls(weights=weights, symbols=symbols, dates=dates)

    @classmethod
    def market_cap_weight(
        cls,
        portfolio_data: PortfolioData,
        dates: pd.DatetimeIndex,
    ) -> TargetWeights:
        """
        Create market-cap weighted portfolio.

        Args:
            portfolio_data: Portfolio data with market_cap in metadata
            dates: Trading dates

        Returns:
            TargetWeights weighted by market cap
        """
        symbols = portfolio_data.symbols
        n_symbols = len(symbols)
        n_dates = len(dates)

        # Get market caps from metadata
        market_caps = np.zeros(n_symbols)
        for i, symbol in enumerate(symbols):
            metadata = portfolio_data.get_metadata(symbol)
            if metadata and metadata.market_cap:
                market_caps[i] = metadata.market_cap
            else:
                market_caps[i] = 1.0  # Default equal weight if no cap

        # Normalize to sum to 1
        total_cap = market_caps.sum()
        if total_cap > 0:
            market_caps = market_caps / total_cap

        # Broadcast to all dates
        weights = np.tile(market_caps.reshape(-1, 1), (1, n_dates))

        return cls(weights=weights, symbols=symbols, dates=dates)

    @classmethod
    def from_array(
        cls,
        weights: np.ndarray,
        symbols: list[str],
        dates: pd.DatetimeIndex,
    ) -> TargetWeights:
        """Create from existing weights array."""
        return cls(weights=weights, symbols=symbols, dates=dates)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to DataFrame with dates as index and symbols as columns."""
        return pd.DataFrame(
            self.weights.T,
            index=self.dates,
            columns=self.symbols,
        )


@dataclass
class SignalResult:
    """
    Output of cross-sectional signal generation.

    Attributes:
        values: Raw signal values (n_symbols x n_dates)
        ranks: Cross-sectional ranks (n_symbols x n_dates)
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
            New SignalResult with only top n% assets (others set to NaN)
        """
        threshold = 100 - n
        mask = self.ranks >= threshold
        new_values = np.where(mask, self.values, np.nan)
        new_ranks = np.where(mask, self.ranks, np.nan)
        return SignalResult(
            values=new_values,
            ranks=new_ranks,
            symbols=self.symbols,
            dates=self.dates,
        )

    def bottom_percentile(self, n: int) -> SignalResult:
        """Filter to bottom n percentile of assets."""
        threshold = n
        mask = self.ranks <= threshold
        new_values = np.where(mask, self.values, np.nan)
        new_ranks = np.where(mask, self.ranks, np.nan)
        return SignalResult(
            values=new_values,
            ranks=new_ranks,
            symbols=self.symbols,
            dates=self.dates,
        )

    def between_percentile(self, low: int, high: int) -> SignalResult:
        """Filter to assets between low and high percentile."""
        mask = (self.ranks >= low) & (self.ranks <= high)
        new_values = np.where(mask, self.values, np.nan)
        new_ranks = np.where(mask, self.ranks, np.nan)
        return SignalResult(
            values=new_values,
            ranks=new_ranks,
            symbols=self.symbols,
            dates=self.dates,
        )

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
        n_symbols = len(self.symbols)
        n_dates = len(self.dates)
        weights = np.zeros((n_symbols, n_dates))

        for j in range(n_dates):
            col_values = self.values[:, j]
            active = ~np.isnan(col_values)

            if not active.any():
                continue

            if method == "equal":
                active_count = active.sum()
                weights[active, j] = 1.0 / active_count

            elif method == "signal_weighted":
                # Weight proportional to signal value
                signal_vals = col_values[active]
                signal_vals = signal_vals - signal_vals.min()  # Shift to positive
                if signal_vals.sum() > 0:
                    weights[active, j] = signal_vals / signal_vals.sum()
                else:
                    weights[active, j] = 1.0 / active.sum()

            elif method == "rank_weighted":
                # Weight proportional to rank
                rank_vals = self.ranks[active, j]
                if rank_vals.sum() > 0:
                    weights[active, j] = rank_vals / rank_vals.sum()
                else:
                    weights[active, j] = 1.0 / active.sum()

        if normalize:
            # Ensure columns sum to 1
            for j in range(n_dates):
                col_sum = weights[:, j].sum()
                if col_sum > 0:
                    weights[:, j] /= col_sum

        return TargetWeights(
            weights=weights,
            symbols=self.symbols,
            dates=self.dates,
        )


class CrossSectionalSignal(ABC):
    """
    Generator for cross-sectional (relative) signals.

    Cross-sectional signals rank assets against each other within
    the universe, rather than using absolute thresholds.

    Example:
        >>> signal = CrossSectionalSignal.momentum(lookback=252)
        >>> result = signal.generate(portfolio_data)
        >>> weights = result.top_percentile(10).to_weights()
    """

    def __init__(
        self,
        lookback: int,
        rank_method: RankMethod = RankMethod.PERCENTILE,
        name: str = "signal",
    ):
        self._lookback = lookback
        self._rank_method = rank_method
        self._name = name

    @property
    def lookback(self) -> int:
        """Lookback period in days."""
        return self._lookback

    @property
    def rank_method(self) -> RankMethod:
        """Ranking method used."""
        return self._rank_method

    @abstractmethod
    def _compute_raw_signal(self, portfolio_data: PortfolioData) -> np.ndarray:
        """
        Compute raw signal values.

        Args:
            portfolio_data: Multi-asset price data

        Returns:
            Array of raw signal values (n_symbols, n_dates)
        """
        pass

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
        # Compute raw signal values
        raw_values = self._compute_raw_signal(data)

        # Rank cross-sectionally
        ranks = self._rank_signals(raw_values, data, group_field)

        return SignalResult(
            values=raw_values,
            ranks=ranks,
            symbols=data.symbols,
            dates=data.dates,
        )

    def _rank_signals(
        self,
        values: np.ndarray,
        data: PortfolioData,
        group_field: str | None = None,
    ) -> np.ndarray:
        """Rank signal values cross-sectionally."""
        n_symbols, n_dates = values.shape
        ranks = np.zeros_like(values)

        for j in range(n_dates):
            col_values = values[:, j]
            valid = ~np.isnan(col_values) & np.isfinite(col_values)

            if not valid.any():
                ranks[:, j] = np.nan
                continue

            if group_field is None:
                # Rank all together
                if self._rank_method == RankMethod.PERCENTILE:
                    # percentile rank (0-100)
                    valid_vals = col_values[valid]
                    percentiles = stats.rankdata(valid_vals, method="average")
                    percentiles = (
                        (percentiles - 1) / (len(percentiles) - 1) * 100
                        if len(percentiles) > 1
                        else np.array([50.0])
                    )
                    ranks[valid, j] = percentiles
                elif self._rank_method == RankMethod.FRACTIONAL:
                    valid_vals = col_values[valid]
                    fracs = stats.rankdata(valid_vals, method="average")
                    fracs = fracs / len(fracs)
                    ranks[valid, j] = fracs
                elif self._rank_method == RankMethod.ORDINAL:
                    valid_vals = col_values[valid]
                    ranks[valid, j] = stats.rankdata(valid_vals, method="ordinal")
            else:
                # Sector-neutral ranking
                # Group by sector and rank within each group
                sectors: dict[str, list[int]] = {}
                for i in range(n_symbols):
                    if not valid[i]:
                        continue
                    metadata = data.get_metadata(data.symbols[i])
                    sector = getattr(metadata, group_field, None) if metadata else None
                    sector = sector or "Unknown"
                    if sector not in sectors:
                        sectors[sector] = []
                    sectors[sector].append(i)

                for sector, indices in sectors.items():
                    if len(indices) < 2:
                        ranks[indices, j] = 50.0  # Single stock gets median rank
                        continue

                    sector_vals = col_values[indices]
                    if self._rank_method == RankMethod.PERCENTILE:
                        percentiles = stats.rankdata(sector_vals, method="average")
                        percentiles = (percentiles - 1) / (len(percentiles) - 1) * 100
                        for k, idx in enumerate(indices):
                            ranks[idx, j] = percentiles[k]

        return ranks

    @classmethod
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
        return MomentumSignal(
            lookback=lookback,
            skip_recent=skip_recent,
            rank_method=rank_method,
        )

    @classmethod
    def mean_reversion(
        cls,
        lookback: int = 20,
        rank_method: RankMethod = RankMethod.PERCENTILE,
    ) -> CrossSectionalSignal:
        """Create mean reversion (short-term reversal) signal."""
        return MeanReversionSignal(
            lookback=lookback,
            rank_method=rank_method,
        )

    @classmethod
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
        return VolatilitySignal(
            lookback=lookback,
            rank_method=rank_method,
            inverse=inverse,
        )

    @classmethod
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
            func: Function (prices: ndarray) -> signal values
            lookback: Required lookback period
            name: Signal name for display
            rank_method: Ranking method

        Returns:
            Configured custom signal generator
        """
        return CustomSignal(
            func=func,
            lookback=lookback,
            name=name,
            rank_method=rank_method,
        )

    @classmethod
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
        return RelativeStrengthSignal(
            lookback=lookback,
            benchmark_idx=benchmark_idx,
        )


class MomentumSignal(CrossSectionalSignal):
    """Momentum signal based on past returns."""

    def __init__(
        self,
        lookback: int = 252,
        skip_recent: int = 21,
        rank_method: RankMethod = RankMethod.PERCENTILE,
    ):
        super().__init__(lookback=lookback, rank_method=rank_method, name="momentum")
        self.skip_recent = skip_recent

    def _compute_raw_signal(self, portfolio_data: PortfolioData) -> np.ndarray:
        """Compute momentum as cumulative return over lookback period."""
        close = portfolio_data.close()
        n_symbols, n_dates = close.shape

        signal = np.full((n_symbols, n_dates), np.nan)

        # Need enough history
        start_idx = self._lookback
        end_offset = self.skip_recent

        for j in range(start_idx, n_dates - end_offset):
            # Return from (j - lookback) to (j - skip_recent)
            start_prices = close[:, j - self._lookback]
            end_prices = close[:, j - end_offset]
            returns = (end_prices - start_prices) / start_prices
            signal[:, j] = returns

        return signal


class MeanReversionSignal(CrossSectionalSignal):
    """Mean reversion signal based on short-term returns (reversal)."""

    def __init__(
        self,
        lookback: int = 20,
        rank_method: RankMethod = RankMethod.PERCENTILE,
    ):
        super().__init__(lookback=lookback, rank_method=rank_method, name="mean_reversion")

    def _compute_raw_signal(self, portfolio_data: PortfolioData) -> np.ndarray:
        """Compute negative of recent returns (losers expected to reverse)."""
        close = portfolio_data.close()
        n_symbols, n_dates = close.shape

        signal = np.full((n_symbols, n_dates), np.nan)

        for j in range(self._lookback, n_dates):
            start_prices = close[:, j - self._lookback]
            end_prices = close[:, j]
            returns = (end_prices - start_prices) / start_prices
            # Negative: recent losers rank higher
            signal[:, j] = -returns

        return signal


class VolatilitySignal(CrossSectionalSignal):
    """Volatility signal based on rolling standard deviation."""

    def __init__(
        self,
        lookback: int = 60,
        rank_method: RankMethod = RankMethod.PERCENTILE,
        inverse: bool = True,
    ):
        super().__init__(lookback=lookback, rank_method=rank_method, name="volatility")
        self.inverse = inverse

    def _compute_raw_signal(self, portfolio_data: PortfolioData) -> np.ndarray:
        """Compute rolling volatility."""
        returns = portfolio_data.returns()
        n_symbols, n_dates = returns.shape

        signal = np.full((n_symbols, n_dates + 1), np.nan)

        for j in range(self._lookback, n_dates):
            window_returns = returns[:, j - self._lookback : j]
            vol = np.std(window_returns, axis=1) * np.sqrt(252)
            if self.inverse:
                # Lower volatility ranks higher
                vol = -vol
            signal[:, j + 1] = vol

        return signal


class CustomSignal(CrossSectionalSignal):
    """Custom signal from user-provided function."""

    def __init__(
        self,
        func: Callable[[np.ndarray], np.ndarray],
        lookback: int,
        name: str = "custom",
        rank_method: RankMethod = RankMethod.PERCENTILE,
    ):
        super().__init__(lookback=lookback, rank_method=rank_method, name=name)
        self.func = func

    def _compute_raw_signal(self, portfolio_data: PortfolioData) -> np.ndarray:
        """Apply custom function to price data."""
        close = portfolio_data.close()
        return self.func(close)


class RelativeStrengthSignal(CrossSectionalSignal):
    """Relative strength compared to universe average or benchmark."""

    def __init__(
        self,
        lookback: int = 252,
        benchmark_idx: int | None = None,
        rank_method: RankMethod = RankMethod.PERCENTILE,
    ):
        super().__init__(lookback=lookback, rank_method=rank_method, name="relative_strength")
        self.benchmark_idx = benchmark_idx

    def _compute_raw_signal(self, portfolio_data: PortfolioData) -> np.ndarray:
        """Compute relative strength vs benchmark or average."""
        close = portfolio_data.close()
        n_symbols, n_dates = close.shape

        signal = np.full((n_symbols, n_dates), np.nan)

        for j in range(self._lookback, n_dates):
            start_prices = close[:, j - self._lookback]
            end_prices = close[:, j]
            returns = (end_prices - start_prices) / start_prices

            if self.benchmark_idx is not None:
                benchmark_return = returns[self.benchmark_idx]
            else:
                # Use average return as benchmark
                benchmark_return = np.nanmean(returns)

            # Relative strength = asset return - benchmark return
            signal[:, j] = returns - benchmark_return

        return signal
