"""
Portfolio Data Contract
=======================

API contract for PortfolioData and related data handling.
These stubs define the public interface; implementations follow.

Contract Tests: tests/contract/test_portfolio_data_contract.py
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date
from enum import Enum
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from pathlib import Path


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


@dataclass
class CorporateAction:
    """Stock split or dividend event."""

    symbol: str
    action_type: Literal["split", "dividend"]
    effective_date: date
    adjustment_factor: float = 1.0  # For splits: new_shares / old_shares
    cash_amount: float = 0.0  # For dividends: per-share amount
    reinvest: bool = False  # Whether to reinvest dividends


class PortfolioData(ABC):
    """
    Container for multi-symbol OHLCV data with alignment.

    Example:
        >>> data = PortfolioData.from_dict({
        ...     "AAPL": aapl_df,
        ...     "GOOG": goog_df,
        ...     "MSFT": msft_df,
        ... })
        >>> aligned = data.align(policy=MissingDataPolicy.FORWARD_FILL)
        >>> validated = aligned.validate()
    """

    @property
    @abstractmethod
    def symbols(self) -> list[str]:
        """Ordered list of ticker symbols."""
        ...

    @property
    @abstractmethod
    def dates(self) -> pd.DatetimeIndex:
        """Aligned trading dates."""
        ...

    @property
    @abstractmethod
    def n_symbols(self) -> int:
        """Number of symbols in universe."""
        ...

    @property
    @abstractmethod
    def n_dates(self) -> int:
        """Number of aligned trading dates."""
        ...

    @property
    @abstractmethod
    def shape(self) -> tuple[int, int, int]:
        """Shape of price array: (n_symbols, n_dates, n_fields)."""
        ...

    @abstractmethod
    def close(self) -> np.ndarray:
        """
        Get close prices as 2D array.

        Returns:
            ndarray of shape (n_symbols, n_dates)
        """
        ...

    @abstractmethod
    def open(self) -> np.ndarray:
        """Get open prices as 2D array."""
        ...

    @abstractmethod
    def high(self) -> np.ndarray:
        """Get high prices as 2D array."""
        ...

    @abstractmethod
    def low(self) -> np.ndarray:
        """Get low prices as 2D array."""
        ...

    @abstractmethod
    def volume(self) -> np.ndarray:
        """Get volume as 2D array."""
        ...

    @abstractmethod
    def returns(self, periods: int = 1) -> np.ndarray:
        """
        Calculate returns for all symbols.

        Args:
            periods: Number of periods for return calculation

        Returns:
            ndarray of shape (n_symbols, n_dates - periods)
        """
        ...

    @abstractmethod
    def tradeable_mask(self) -> np.ndarray:
        """
        Get tradeable mask indicating which symbol-dates are valid.

        Returns:
            bool ndarray of shape (n_symbols, n_dates)
        """
        ...

    @abstractmethod
    def get_metadata(self, symbol: str) -> SymbolMetadata | None:
        """Get metadata for a symbol."""
        ...

    @abstractmethod
    def set_metadata(self, symbol: str, metadata: SymbolMetadata) -> PortfolioData:
        """Set metadata for a symbol. Returns new instance."""
        ...

    @classmethod
    @abstractmethod
    def from_dict(
        cls,
        data: dict[str, pd.DataFrame],
        metadata: dict[str, SymbolMetadata] | None = None,
    ) -> PortfolioData:
        """
        Create PortfolioData from dictionary of DataFrames.

        Args:
            data: Mapping of symbol → OHLCV DataFrame
            metadata: Optional symbol metadata

        Returns:
            Unaligned PortfolioData instance
        """
        ...

    @classmethod
    @abstractmethod
    def from_parquet(cls, path: Path) -> PortfolioData:
        """Load PortfolioData from Parquet file."""
        ...

    @abstractmethod
    def to_parquet(self, path: Path) -> None:
        """Save PortfolioData to Parquet file."""
        ...

    @abstractmethod
    def align(
        self,
        policy: MissingDataPolicy = MissingDataPolicy.FORWARD_FILL,
        start_date: date | None = None,
        end_date: date | None = None,
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
        ...

    @abstractmethod
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
        ...

    @abstractmethod
    def apply_corporate_actions(
        self,
        actions: list[CorporateAction],
    ) -> PortfolioData:
        """
        Apply corporate actions to price data.

        Args:
            actions: List of corporate actions to apply

        Returns:
            New PortfolioData with adjusted prices
        """
        ...

    @abstractmethod
    def filter_symbols(self, symbols: list[str]) -> PortfolioData:
        """Filter to subset of symbols."""
        ...

    @abstractmethod
    def filter_dates(self, start: date, end: date) -> PortfolioData:
        """Filter to date range."""
        ...

    @abstractmethod
    def get_symbol_data(self, symbol: str) -> pd.DataFrame:
        """Get OHLCV DataFrame for a single symbol."""
        ...
