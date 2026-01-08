"""
Corporate Actions Module

Handles stock splits, dividends, mergers, and spinoffs for portfolio backtesting.
"""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from typing import Optional, Union


class CorporateActionType(Enum):
    """Types of corporate actions."""

    SPLIT = "split"
    DIVIDEND = "dividend"
    MERGER = "merger"
    SPINOFF = "spinoff"


@dataclass(frozen=True)
class CorporateAction(ABC):
    """Base class for all corporate actions."""

    symbol: str
    ex_date: date

    @property
    @abstractmethod
    def action_type(self) -> CorporateActionType:
        """Return the type of corporate action."""
        pass


@dataclass(frozen=True)
class Split(CorporateAction):
    """
    Stock split corporate action.

    Attributes:
        symbol: The ticker symbol
        ex_date: The ex-date of the split
        ratio: The split ratio (e.g., 4.0 for 4-for-1 split, 0.1 for 1-for-10 reverse split)
    """

    ratio: float

    @property
    def action_type(self) -> CorporateActionType:
        return CorporateActionType.SPLIT

    def adjust_price(self, price: float) -> float:
        """
        Adjust price for the split.

        For a 4-for-1 split, historical prices are divided by 4.
        For a 1-for-10 reverse split (ratio=0.1), prices are multiplied by 10.
        """
        return price / self.ratio

    def adjust_shares(self, shares: Union[int, float]) -> Union[int, float]:
        """
        Adjust share count for the split.

        For a 4-for-1 split, share count is multiplied by 4.
        For a 1-for-10 reverse split (ratio=0.1), shares are divided by 10.
        """
        result = shares * self.ratio
        if isinstance(shares, int):
            return int(result)
        return result

    def adjust_volume(self, volume: float) -> float:
        """
        Adjust volume for the split.

        Historical volume is multiplied by the split ratio.
        """
        return volume * self.ratio


@dataclass(frozen=True)
class Dividend(CorporateAction):
    """
    Dividend corporate action.

    Attributes:
        symbol: The ticker symbol
        ex_date: The ex-date of the dividend
        amount: The dividend amount per share
    """

    amount: float

    @property
    def action_type(self) -> CorporateActionType:
        return CorporateActionType.DIVIDEND

    def cash_payout(self, shares: Union[int, float]) -> float:
        """Calculate cash payout for given number of shares."""
        return shares * self.amount

    def reinvestment_shares(self, shares: Union[int, float], price: float) -> float:
        """
        Calculate number of shares from dividend reinvestment (DRIP).

        Args:
            shares: Current number of shares held
            price: Price per share at reinvestment

        Returns:
            Number of new shares purchased with dividend
        """
        cash = self.cash_payout(shares)
        return cash / price if price > 0 else 0.0

    def adjustment_factor(self, price: float) -> float:
        """
        Calculate the price adjustment factor for dividend.

        Historical prices are adjusted down by the dividend amount
        to maintain total return continuity.

        Args:
            price: The closing price before ex-date

        Returns:
            Factor to multiply historical prices by (< 1.0)
        """
        if price <= 0:
            return 1.0
        return (price - self.amount) / price


@dataclass(frozen=True)
class Merger(CorporateAction):
    """
    Merger corporate action.

    Attributes:
        symbol: The target (acquired) company ticker
        ex_date: The effective date of the merger
        acquirer: The acquiring company ticker
        ratio: Exchange ratio (acquirer shares per target share)
        cash_per_share: Cash component per target share
    """

    acquirer: str
    ratio: float
    cash_per_share: float = 0.0

    @property
    def action_type(self) -> CorporateActionType:
        return CorporateActionType.MERGER

    def convert_shares(self, target_shares: Union[int, float]) -> tuple[float, float]:
        """
        Convert target shares to acquirer shares and cash.

        Args:
            target_shares: Number of target company shares

        Returns:
            Tuple of (acquirer_shares, cash_received)
        """
        acquirer_shares = target_shares * self.ratio
        cash = target_shares * self.cash_per_share
        return acquirer_shares, cash


@dataclass(frozen=True)
class Spinoff(CorporateAction):
    """
    Spinoff corporate action.

    Attributes:
        symbol: The parent company ticker
        ex_date: The ex-date of the spinoff
        spinoff_symbol: The new spinoff company ticker
        ratio: Spinoff shares received per parent share
    """

    spinoff_symbol: str
    ratio: float

    @property
    def action_type(self) -> CorporateActionType:
        return CorporateActionType.SPINOFF

    def spinoff_shares(self, parent_shares: Union[int, float]) -> float:
        """Calculate spinoff shares received from parent shares."""
        return parent_shares * self.ratio


@dataclass
class CorporateActionsRegistry:
    """
    Registry for managing corporate actions.

    Provides efficient lookup and filtering of corporate actions
    by symbol, date, and type.
    """

    _actions: list[CorporateAction] = field(default_factory=list)
    _by_symbol: dict[str, list[CorporateAction]] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self._actions)

    def __iter__(self):
        return iter(self._actions)

    @classmethod
    def from_list(cls, actions: Sequence[CorporateAction]) -> "CorporateActionsRegistry":
        """
        Create registry from a list of corporate actions.

        Args:
            actions: Sequence of corporate actions

        Returns:
            New CorporateActionsRegistry instance
        """
        registry = cls()
        registry._actions = list(actions)

        # Index by symbol
        for action in actions:
            if action.symbol not in registry._by_symbol:
                registry._by_symbol[action.symbol] = []
            registry._by_symbol[action.symbol].append(action)

        # Sort by date within each symbol
        for symbol in registry._by_symbol:
            registry._by_symbol[symbol].sort(key=lambda a: a.ex_date)

        return registry

    def get_actions_for_symbol(self, symbol: str) -> list[CorporateAction]:
        """
        Get all corporate actions for a symbol.

        Args:
            symbol: The ticker symbol

        Returns:
            List of corporate actions for the symbol, sorted by date
        """
        return self._by_symbol.get(symbol, [])

    def filter_date_range(
        self,
        start: date,
        end: date,
    ) -> "CorporateActionsRegistry":
        """
        Filter actions to those within a date range.

        Args:
            start: Start date (inclusive)
            end: End date (inclusive)

        Returns:
            New registry with filtered actions
        """
        filtered = [action for action in self._actions if start <= action.ex_date <= end]
        return CorporateActionsRegistry.from_list(filtered)

    def get_actions_between(
        self,
        start: date,
        end: date,
    ) -> list[CorporateAction]:
        """
        Get actions between two dates.

        Args:
            start: Start date (inclusive)
            end: End date (inclusive)

        Returns:
            List of actions within the date range
        """
        return [action for action in self._actions if start <= action.ex_date <= end]

    def get_splits(self, symbol: Optional[str] = None) -> list[Split]:
        """Get all splits, optionally filtered by symbol."""
        actions = self._by_symbol.get(symbol, self._actions) if symbol else self._actions
        return [a for a in actions if isinstance(a, Split)]

    def get_dividends(self, symbol: Optional[str] = None) -> list[Dividend]:
        """Get all dividends, optionally filtered by symbol."""
        actions = self._by_symbol.get(symbol, self._actions) if symbol else self._actions
        return [a for a in actions if isinstance(a, Dividend)]

    def get_mergers(self, symbol: Optional[str] = None) -> list[Merger]:
        """Get all mergers, optionally filtered by symbol."""
        actions = self._by_symbol.get(symbol, self._actions) if symbol else self._actions
        return [a for a in actions if isinstance(a, Merger)]

    def get_spinoffs(self, symbol: Optional[str] = None) -> list[Spinoff]:
        """Get all spinoffs, optionally filtered by symbol."""
        actions = self._by_symbol.get(symbol, self._actions) if symbol else self._actions
        return [a for a in actions if isinstance(a, Spinoff)]


def apply_actions(
    portfolio_data: "PortfolioData",
    actions: Sequence[CorporateAction],
) -> "PortfolioData":
    """
    Apply corporate actions to PortfolioData.

    This function adjusts historical prices and volumes for splits and dividends.
    Actions are applied in chronological order (oldest first).

    Args:
        portfolio_data: The portfolio data to adjust
        actions: List of corporate actions to apply

    Returns:
        New PortfolioData with adjusted prices
    """
    # Import here to avoid circular imports
    import numpy as np
    import pandas as pd

    from vectorforge.portfolio.data import (
        OHLCV_CLOSE,
        OHLCV_HIGH,
        OHLCV_LOW,
        OHLCV_OPEN,
        OHLCV_VOLUME,
        PortfolioData,
    )

    if not actions:
        return portfolio_data

    # Create a copy of the price array
    new_prices = portfolio_data._prices.copy()
    symbols = portfolio_data._symbols
    dates = portfolio_data._dates

    # Sort actions by date (process oldest first for correct cumulative adjustment)
    sorted_actions = sorted(actions, key=lambda a: a.ex_date)

    # Process each action
    for action in sorted_actions:
        if action.symbol not in symbols:
            continue

        symbol_idx = symbols.index(action.symbol)

        # Find the ex-date index (first date >= ex_date)
        ex_ts = pd.Timestamp(action.ex_date)
        ex_date_indices = np.where(dates >= ex_ts)[0]

        if len(ex_date_indices) == 0:
            # Ex-date is after all our data - apply to all historical data
            ex_date_idx = len(dates)
        else:
            ex_date_idx = ex_date_indices[0]

        if isinstance(action, Split):
            # Adjust all prices BEFORE the ex-date
            # For a 4-for-1 split: divide historical prices by 4
            if ex_date_idx > 0:
                # Adjust OHLC prices
                for field in [OHLCV_OPEN, OHLCV_HIGH, OHLCV_LOW, OHLCV_CLOSE]:
                    new_prices[symbol_idx, :ex_date_idx, field] /= action.ratio

                # Adjust volume (multiply by ratio)
                new_prices[symbol_idx, :ex_date_idx, OHLCV_VOLUME] *= action.ratio

        elif isinstance(action, Dividend):
            # Adjust historical prices to maintain total return continuity
            # Get the close price just before ex-date to compute adjustment factor
            if ex_date_idx > 0:
                # Use the close price from the day before ex-date
                pre_ex_close = new_prices[symbol_idx, ex_date_idx - 1, OHLCV_CLOSE]

                if pre_ex_close > 0 and not np.isnan(pre_ex_close):
                    adj_factor = action.adjustment_factor(pre_ex_close)

                    # Apply adjustment to all prices before ex-date
                    for field in [OHLCV_OPEN, OHLCV_HIGH, OHLCV_LOW, OHLCV_CLOSE]:
                        new_prices[symbol_idx, :ex_date_idx, field] *= adj_factor

    # Create new PortfolioData instance
    return PortfolioData(
        symbols=symbols,
        dates=dates,
        prices=new_prices,
        tradeable_mask=portfolio_data._tradeable_mask.copy(),
        metadata=portfolio_data._metadata.copy() if portfolio_data._metadata else None,
    )
