"""
Portfolio Data Container

Multi-symbol OHLCV data container with alignment and validation.

This module provides the core PortfolioData class for managing
multi-asset price data with date alignment, missing data handling,
and validation.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    pass


class MissingDataPolicy(Enum):
    """How to handle missing data during alignment."""

    FORWARD_FILL = "ffill"  # Carry last known value (default)
    INTERPOLATE = "interpolate"  # Linear interpolation
    DROP = "drop"  # Exclude from universe for that period
    ZERO = "zero"  # Fill with zero


@dataclass
class SymbolMetadata:
    """Per-symbol metadata for grouping and weighting."""

    symbol: str
    sector: str | None = None
    industry: str | None = None
    market_cap: float | None = None
    exchange: str | None = None

    def __post_init__(self) -> None:
        if self.market_cap is not None and self.market_cap <= 0:
            raise ValueError(f"market_cap must be > 0, got {self.market_cap}")


# OHLCV field indices in the 3D price array
OHLCV_OPEN = 0
OHLCV_HIGH = 1
OHLCV_LOW = 2
OHLCV_CLOSE = 3
OHLCV_VOLUME = 4
N_FIELDS = 5


class PortfolioData:
    """
    Container for multi-symbol OHLCV data with alignment.

    This class manages price data for multiple symbols aligned to a common
    date index. It provides efficient access patterns for cross-sectional
    operations through NumPy arrays.

    Attributes:
        symbols: Ordered list of ticker symbols
        dates: Aligned trading dates as DatetimeIndex
        n_symbols: Number of symbols in universe
        n_dates: Number of aligned trading dates

    Example:
        >>> data = PortfolioData.from_dict({
        ...     "AAPL": aapl_df,
        ...     "GOOG": goog_df,
        ...     "MSFT": msft_df,
        ... })
        >>> aligned = data.align(policy=MissingDataPolicy.FORWARD_FILL)
        >>> validated = aligned.validate()
        >>> print(f"Universe: {validated.n_symbols} symbols x {validated.n_dates} dates")
    """

    def __init__(
        self,
        symbols: list[str],
        dates: pd.DatetimeIndex,
        prices: np.ndarray,
        tradeable_mask: np.ndarray | None = None,
        metadata: dict[str, SymbolMetadata] | None = None,
    ) -> None:
        """
        Initialize PortfolioData.

        Args:
            symbols: Ordered list of ticker symbols
            dates: Trading dates as DatetimeIndex
            prices: 3D array of shape (n_symbols, n_dates, 5) for OHLCV
            tradeable_mask: 2D boolean array (n_symbols, n_dates) indicating tradeable periods
            metadata: Optional per-symbol metadata
        """
        self._symbols = list(symbols)
        self._dates = dates
        self._prices = prices.astype(np.float32)
        self._metadata = metadata or {}

        # Initialize tradeable mask if not provided
        if tradeable_mask is None:
            # Default: tradeable where we have valid close prices
            self._tradeable_mask = ~np.isnan(prices[:, :, OHLCV_CLOSE])
        else:
            self._tradeable_mask = tradeable_mask

        self._validate_shapes()

    def _validate_shapes(self) -> None:
        """Validate array shapes match."""
        n_symbols = len(self._symbols)
        n_dates = len(self._dates)

        if self._prices.shape != (n_symbols, n_dates, N_FIELDS):
            raise ValueError(
                f"prices shape {self._prices.shape} doesn't match "
                f"expected ({n_symbols}, {n_dates}, {N_FIELDS})"
            )

        if self._tradeable_mask.shape != (n_symbols, n_dates):
            raise ValueError(
                f"tradeable_mask shape {self._tradeable_mask.shape} doesn't match "
                f"expected ({n_symbols}, {n_dates})"
            )

    @property
    def symbols(self) -> list[str]:
        """Ordered list of ticker symbols."""
        return self._symbols.copy()

    @property
    def dates(self) -> pd.DatetimeIndex:
        """Aligned trading dates."""
        return self._dates.copy()

    @property
    def n_symbols(self) -> int:
        """Number of symbols in universe."""
        return len(self._symbols)

    @property
    def n_dates(self) -> int:
        """Number of aligned trading dates."""
        return len(self._dates)

    @property
    def shape(self) -> tuple[int, int, int]:
        """Shape of price array: (n_symbols, n_dates, n_fields)."""
        return self._prices.shape

    def close(self) -> np.ndarray:
        """
        Get close prices as 2D array.

        Returns:
            ndarray of shape (n_symbols, n_dates)
        """
        return self._prices[:, :, OHLCV_CLOSE].copy()

    def open(self) -> np.ndarray:
        """Get open prices as 2D array."""
        return self._prices[:, :, OHLCV_OPEN].copy()

    def high(self) -> np.ndarray:
        """Get high prices as 2D array."""
        return self._prices[:, :, OHLCV_HIGH].copy()

    def low(self) -> np.ndarray:
        """Get low prices as 2D array."""
        return self._prices[:, :, OHLCV_LOW].copy()

    def volume(self) -> np.ndarray:
        """Get volume as 2D array."""
        return self._prices[:, :, OHLCV_VOLUME].copy()

    def returns(self, periods: int = 1) -> np.ndarray:
        """
        Calculate returns for all symbols.

        Args:
            periods: Number of periods for return calculation

        Returns:
            ndarray of shape (n_symbols, n_dates - periods)
        """
        close = self._prices[:, :, OHLCV_CLOSE]
        if periods == 1:
            ret = np.diff(close, axis=1) / close[:, :-1]
        else:
            ret = (close[:, periods:] - close[:, :-periods]) / close[:, :-periods]

        # Handle division by zero
        ret = np.where(np.isfinite(ret), ret, 0.0)
        return ret

    def tradeable_mask(self) -> np.ndarray:
        """
        Get tradeable mask indicating which symbol-dates are valid.

        Returns:
            bool ndarray of shape (n_symbols, n_dates)
        """
        return self._tradeable_mask.copy()

    def get_metadata(self, symbol: str) -> SymbolMetadata | None:
        """Get metadata for a symbol."""
        return self._metadata.get(symbol)

    def set_metadata(self, symbol: str, metadata: SymbolMetadata) -> PortfolioData:
        """Set metadata for a symbol. Returns new instance."""
        if symbol not in self._symbols:
            raise ValueError(f"Symbol {symbol} not in portfolio")

        new_metadata = self._metadata.copy()
        new_metadata[symbol] = metadata

        return PortfolioData(
            symbols=self._symbols,
            dates=self._dates,
            prices=self._prices.copy(),
            tradeable_mask=self._tradeable_mask.copy(),
            metadata=new_metadata,
        )

    @classmethod
    def from_dict(
        cls,
        data: dict[str, pd.DataFrame],
        metadata: dict[str, SymbolMetadata] | None = None,
    ) -> PortfolioData:
        """
        Create PortfolioData from dictionary of DataFrames.

        Each DataFrame should have OHLCV columns (case-insensitive)
        and a DatetimeIndex.

        Args:
            data: Mapping of symbol -> OHLCV DataFrame
            metadata: Optional symbol metadata

        Returns:
            Unaligned PortfolioData instance
        """
        if not data:
            raise ValueError("data dictionary cannot be empty")

        symbols = list(data.keys())

        # Collect all unique dates across all symbols
        all_dates: set[pd.Timestamp] = set()
        for symbol, df in data.items():
            if not isinstance(df.index, pd.DatetimeIndex):
                raise ValueError(f"DataFrame for {symbol} must have DatetimeIndex")
            all_dates.update(df.index)

        # Create sorted date index
        dates = pd.DatetimeIndex(sorted(all_dates))
        n_symbols = len(symbols)
        n_dates = len(dates)

        # Initialize price array with NaN
        prices = np.full((n_symbols, n_dates, N_FIELDS), np.nan, dtype=np.float32)
        tradeable_mask = np.zeros((n_symbols, n_dates), dtype=bool)

        # Fill in data for each symbol
        for i, symbol in enumerate(symbols):
            df = data[symbol]

            # Normalize column names
            col_map = {col.lower(): col for col in df.columns}

            for date_idx, dt in enumerate(dates):
                if dt in df.index:
                    row = df.loc[dt]
                    try:
                        open_val = float(row[col_map.get("open", "Open")])
                        high_val = float(row[col_map.get("high", "High")])
                        low_val = float(row[col_map.get("low", "Low")])
                        close_val = float(row[col_map.get("close", "Close")])
                        volume_val = float(row[col_map.get("volume", "Volume")])

                        prices[i, date_idx, OHLCV_OPEN] = open_val
                        prices[i, date_idx, OHLCV_HIGH] = high_val
                        prices[i, date_idx, OHLCV_LOW] = low_val
                        prices[i, date_idx, OHLCV_CLOSE] = close_val
                        prices[i, date_idx, OHLCV_VOLUME] = volume_val

                        # Only mark as tradeable if close price is valid (not NaN)
                        tradeable_mask[i, date_idx] = not np.isnan(close_val)
                    except (KeyError, TypeError):
                        pass

        return cls(
            symbols=symbols,
            dates=dates,
            prices=prices,
            tradeable_mask=tradeable_mask,
            metadata=metadata,
        )

    @classmethod
    def from_parquet(cls, path: Path | str) -> PortfolioData:
        """
        Load PortfolioData from Parquet file.

        The Parquet file should contain a multi-index DataFrame
        with (symbol, date) index and OHLCV columns.

        Args:
            path: Path to Parquet file

        Returns:
            PortfolioData instance
        """
        path = Path(path)
        df = pd.read_parquet(path)

        if isinstance(df.index, pd.MultiIndex):
            # Multi-index format: (symbol, date)
            symbols = df.index.get_level_values(0).unique().tolist()
            data_dict: dict[str, pd.DataFrame] = {}

            for symbol in symbols:
                symbol_df = df.loc[symbol].copy()
                data_dict[symbol] = symbol_df

            return cls.from_dict(data_dict)
        else:
            raise ValueError("Parquet file must have MultiIndex with (symbol, date) levels")

    def to_parquet(self, path: Path | str) -> None:
        """
        Save PortfolioData to Parquet file.

        Args:
            path: Path to save Parquet file
        """
        path = Path(path)

        # Build multi-index DataFrame
        records = []
        for i, symbol in enumerate(self._symbols):
            for j, dt in enumerate(self._dates):
                records.append(
                    {
                        "symbol": symbol,
                        "date": dt,
                        "open": self._prices[i, j, OHLCV_OPEN],
                        "high": self._prices[i, j, OHLCV_HIGH],
                        "low": self._prices[i, j, OHLCV_LOW],
                        "close": self._prices[i, j, OHLCV_CLOSE],
                        "volume": self._prices[i, j, OHLCV_VOLUME],
                        "tradeable": self._tradeable_mask[i, j],
                    }
                )

        df = pd.DataFrame(records)
        df = df.set_index(["symbol", "date"])
        df.to_parquet(path)

    def align(
        self,
        policy: MissingDataPolicy = MissingDataPolicy.FORWARD_FILL,
        start_date: date | str | None = None,
        end_date: date | str | None = None,
    ) -> PortfolioData:
        """
        Align all symbols to common date index.

        Args:
            policy: How to handle missing data
            start_date: Optional start date filter
            end_date: Optional end date filter

        Returns:
            New PortfolioData with aligned dates
        """
        # Filter dates if specified
        mask = np.ones(len(self._dates), dtype=bool)

        if start_date is not None:
            start_ts = pd.Timestamp(start_date)
            mask &= self._dates >= start_ts

        if end_date is not None:
            end_ts = pd.Timestamp(end_date)
            mask &= self._dates <= end_ts

        if not mask.any():
            raise ValueError("No dates remain after filtering")

        filtered_dates = self._dates[mask]
        filtered_prices = self._prices[:, mask, :].copy()
        filtered_tradeable = self._tradeable_mask[:, mask].copy()

        # Apply missing data policy
        if policy == MissingDataPolicy.FORWARD_FILL:
            # Forward-fill for each symbol and each field
            # Also backfill leading NaN values with first valid value
            for i in range(len(self._symbols)):
                for f in range(N_FIELDS):
                    series = filtered_prices[i, :, f]
                    nan_mask = np.isnan(series)
                    if nan_mask.any() and not nan_mask.all():
                        # Find first valid value
                        first_valid = np.where(~nan_mask)[0][0]
                        # Forward fill from first valid value
                        for j in range(first_valid + 1, len(series)):
                            if np.isnan(series[j]):
                                series[j] = series[j - 1]
                        # Backfill leading NaN values with first valid value
                        if first_valid > 0:
                            series[:first_valid] = series[first_valid]
                        filtered_prices[i, :, f] = series

        elif policy == MissingDataPolicy.INTERPOLATE:
            # Linear interpolation for each symbol and each field
            for i in range(len(self._symbols)):
                for f in range(N_FIELDS):
                    series = filtered_prices[i, :, f]
                    nan_mask = np.isnan(series)
                    if nan_mask.any() and not nan_mask.all():
                        valid_idx = np.where(~nan_mask)[0]
                        invalid_idx = np.where(nan_mask)[0]
                        filtered_prices[i, invalid_idx, f] = np.interp(
                            invalid_idx, valid_idx, series[valid_idx]
                        )

        elif policy == MissingDataPolicy.ZERO:
            # Replace NaN with zero
            filtered_prices = np.nan_to_num(filtered_prices, nan=0.0)

        elif policy == MissingDataPolicy.DROP:
            # Keep NaN but mark as non-tradeable
            nan_close = np.isnan(filtered_prices[:, :, OHLCV_CLOSE])
            filtered_tradeable &= ~nan_close

        # Update tradeable mask based on valid close prices
        if policy != MissingDataPolicy.DROP:
            filtered_tradeable = ~np.isnan(filtered_prices[:, :, OHLCV_CLOSE])

        return PortfolioData(
            symbols=self._symbols,
            dates=filtered_dates,
            prices=filtered_prices,
            tradeable_mask=filtered_tradeable,
            metadata=self._metadata.copy(),
        )

    def validate(
        self,
        min_dates: int = 20,
        max_gap_days: int = 5,
        allow_nan: bool = False,
    ) -> PortfolioData:
        """
        Validate data consistency.

        Args:
            min_dates: Minimum required trading dates
            max_gap_days: Maximum allowed gap between dates
            allow_nan: Whether to allow NaN values

        Returns:
            Self if valid

        Raises:
            ValueError: If validation fails
        """
        # Check minimum dates
        if self.n_dates < min_dates:
            raise ValueError(f"Data has only {self.n_dates} dates, minimum required is {min_dates}")

        # Check for NaN values
        if not allow_nan:
            nan_count = np.isnan(self._prices[:, :, OHLCV_CLOSE]).sum()
            if nan_count > 0:
                raise ValueError(
                    f"Data contains {nan_count} NaN values in close prices. "
                    "Use align() first or set allow_nan=True."
                )

        # Check for date gaps
        if len(self._dates) > 1:
            date_diffs = np.diff(self._dates.values).astype("timedelta64[D]").astype(int)
            max_gap = date_diffs.max()
            if max_gap > max_gap_days:
                gap_idx = np.argmax(date_diffs)
                raise ValueError(
                    f"Maximum date gap is {max_gap} days (between {self._dates[gap_idx]} "
                    f"and {self._dates[gap_idx + 1]}), max allowed is {max_gap_days}. "
                    "This may indicate missing data."
                )

        # Check that each symbol has at least one valid price
        for i, symbol in enumerate(self._symbols):
            if not self._tradeable_mask[i].any():
                raise ValueError(f"Symbol {symbol} has no valid prices")

        return self

    def apply_corporate_actions(
        self,
        actions: list[Any],  # CorporateAction type
    ) -> PortfolioData:
        """
        Apply corporate actions to price data.

        Args:
            actions: List of corporate actions to apply

        Returns:
            New PortfolioData with adjusted prices
        """
        # Import here to avoid circular dependency
        from vectorforge.portfolio.corporate_actions import apply_actions

        return apply_actions(self, actions)

    def filter_symbols(self, symbols: list[str]) -> PortfolioData:
        """Filter to subset of symbols."""
        indices = [self._symbols.index(s) for s in symbols if s in self._symbols]
        if not indices:
            raise ValueError("No valid symbols to filter")

        new_symbols = [self._symbols[i] for i in indices]
        new_prices = self._prices[indices, :, :].copy()
        new_tradeable = self._tradeable_mask[indices, :].copy()
        new_metadata = {s: self._metadata[s] for s in new_symbols if s in self._metadata}

        return PortfolioData(
            symbols=new_symbols,
            dates=self._dates,
            prices=new_prices,
            tradeable_mask=new_tradeable,
            metadata=new_metadata,
        )

    def filter_dates(self, start: date | str, end: date | str) -> PortfolioData:
        """Filter to date range."""
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)

        mask = (self._dates >= start_ts) & (self._dates <= end_ts)
        if not mask.any():
            raise ValueError(f"No dates in range [{start}, {end}]")

        return PortfolioData(
            symbols=self._symbols,
            dates=self._dates[mask],
            prices=self._prices[:, mask, :].copy(),
            tradeable_mask=self._tradeable_mask[:, mask].copy(),
            metadata=self._metadata.copy(),
        )

    def get_symbol_data(self, symbol: str) -> pd.DataFrame:
        """Get OHLCV DataFrame for a single symbol."""
        if symbol not in self._symbols:
            raise ValueError(f"Symbol {symbol} not in portfolio")

        idx = self._symbols.index(symbol)
        data = {
            "open": self._prices[idx, :, OHLCV_OPEN],
            "high": self._prices[idx, :, OHLCV_HIGH],
            "low": self._prices[idx, :, OHLCV_LOW],
            "close": self._prices[idx, :, OHLCV_CLOSE],
            "volume": self._prices[idx, :, OHLCV_VOLUME],
        }
        return pd.DataFrame(data, index=self._dates)

    def __repr__(self) -> str:
        return (
            f"PortfolioData(\n"
            f"  symbols={self.n_symbols},\n"
            f"  dates={self.n_dates},\n"
            f"  date_range=[{self._dates[0].date()}, {self._dates[-1].date()}]\n"
            f")"
        )
