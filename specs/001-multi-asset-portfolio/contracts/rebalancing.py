"""
Rebalancing Contract
====================

API contract for portfolio rebalancing logic.
These stubs define the public interface; implementations follow.

Contract Tests: tests/contract/test_rebalancer_contract.py
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from vectorforge.portfolio.data import PortfolioData
    from vectorforge.portfolio.signals import TargetWeights


class RebalanceFrequency(Enum):
    """Calendar-based rebalancing frequencies."""

    DAILY = "daily"
    WEEKLY = "weekly"  # First trading day of week
    MONTHLY = "monthly"  # First trading day of month
    QUARTERLY = "quarterly"  # First trading day of quarter
    ANNUAL = "annual"  # First trading day of year


@dataclass
class RebalanceOrders:
    """
    Output of rebalancing calculation.

    Attributes:
        date: Rebalance date
        trades: Symbol → weight change (positive = buy, negative = sell)
        from_weights: Current weights before rebalance
        to_weights: Target weights after rebalance
        turnover: Total turnover (sum of |delta| / 2)
        constrained: True if turnover limit was applied
    """

    date: date
    trades: dict[str, float]
    from_weights: dict[str, float]
    to_weights: dict[str, float]
    turnover: float
    constrained: bool = False

    @property
    def n_trades(self) -> int:
        """Number of trades in this rebalance."""
        return sum(1 for v in self.trades.values() if abs(v) > 1e-6)

    @property
    def buy_symbols(self) -> list[str]:
        """Symbols being bought."""
        return [s for s, v in self.trades.items() if v > 1e-6]

    @property
    def sell_symbols(self) -> list[str]:
        """Symbols being sold."""
        return [s for s, v in self.trades.items() if v < -1e-6]


@dataclass
class RebalanceResult:
    """
    Complete rebalancing history for a backtest.

    Attributes:
        orders: List of all rebalance orders
        dates: All rebalance dates
        turnover_series: Turnover at each rebalance
    """

    orders: list[RebalanceOrders] = field(default_factory=list)

    @property
    def dates(self) -> list[date]:
        """All rebalance dates."""
        return [o.date for o in self.orders]

    @property
    def total_turnover(self) -> float:
        """Sum of all turnover."""
        return sum(o.turnover for o in self.orders)

    @property
    def avg_turnover(self) -> float:
        """Average turnover per rebalance."""
        if not self.orders:
            return 0.0
        return self.total_turnover / len(self.orders)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to DataFrame with turnover history."""
        ...


class RebalanceTrigger(ABC):
    """
    Abstract trigger determining when to rebalance.

    Implementations define the condition for rebalancing.
    """

    @abstractmethod
    def should_rebalance(
        self,
        current_date: date,
        current_weights: dict[str, float],
        target_weights: dict[str, float],
        last_rebalance: date | None,
    ) -> bool:
        """
        Determine if rebalancing should occur.

        Args:
            current_date: Current simulation date
            current_weights: Current portfolio weights
            target_weights: Target portfolio weights
            last_rebalance: Date of last rebalance (None if never)

        Returns:
            True if should rebalance
        """
        ...


class CalendarTrigger(RebalanceTrigger):
    """
    Trigger based on calendar schedule.

    Example:
        >>> trigger = CalendarTrigger(RebalanceFrequency.MONTHLY)
        >>> trigger.should_rebalance(date(2026, 2, 1), ...)  # True (first of month)
    """

    def __init__(self, frequency: RebalanceFrequency):
        """
        Initialize calendar trigger.

        Args:
            frequency: Rebalancing frequency
        """
        ...

    @property
    def frequency(self) -> RebalanceFrequency:
        """Rebalancing frequency."""
        ...

    def should_rebalance(
        self,
        current_date: date,
        current_weights: dict[str, float],
        target_weights: dict[str, float],
        last_rebalance: date | None,
    ) -> bool:
        ...


class DriftTrigger(RebalanceTrigger):
    """
    Trigger when position drift exceeds threshold.

    Example:
        >>> trigger = DriftTrigger(threshold=0.05)  # 5% drift
        >>> # Rebalances when any position drifts >5% from target
    """

    def __init__(
        self,
        threshold: float,
        measure: str = "absolute",
    ):
        """
        Initialize drift trigger.

        Args:
            threshold: Drift threshold (0.05 = 5%)
            measure: "absolute" or "relative"
        """
        ...

    @property
    def threshold(self) -> float:
        """Drift threshold."""
        ...

    def should_rebalance(
        self,
        current_date: date,
        current_weights: dict[str, float],
        target_weights: dict[str, float],
        last_rebalance: date | None,
    ) -> bool:
        ...

    def get_drifts(
        self,
        current_weights: dict[str, float],
        target_weights: dict[str, float],
    ) -> dict[str, float]:
        """Calculate drift for each position."""
        ...


class HybridTrigger(RebalanceTrigger):
    """
    Combine multiple triggers with OR logic.

    Example:
        >>> trigger = HybridTrigger([
        ...     CalendarTrigger(RebalanceFrequency.MONTHLY),
        ...     DriftTrigger(threshold=0.10),
        ... ])
        >>> # Rebalances monthly OR when 10% drift occurs
    """

    def __init__(self, triggers: list[RebalanceTrigger]):
        """
        Initialize hybrid trigger.

        Args:
            triggers: List of triggers (any can fire)
        """
        ...

    def should_rebalance(
        self,
        current_date: date,
        current_weights: dict[str, float],
        target_weights: dict[str, float],
        last_rebalance: date | None,
    ) -> bool:
        ...


class Rebalancer(ABC):
    """
    Portfolio rebalancing logic.

    Determines when and how to rebalance, respecting constraints.

    Example:
        >>> rebalancer = Rebalancer(
        ...     trigger=CalendarTrigger(RebalanceFrequency.MONTHLY),
        ...     turnover_limit=0.20,
        ... )
        >>> orders = rebalancer.compute_trades(current, target, date)
    """

    @property
    @abstractmethod
    def trigger(self) -> RebalanceTrigger:
        """Rebalancing trigger."""
        ...

    @property
    @abstractmethod
    def turnover_limit(self) -> float | None:
        """Maximum turnover per rebalance (None = unlimited)."""
        ...

    @abstractmethod
    def compute_trades(
        self,
        current_weights: dict[str, float],
        target_weights: dict[str, float],
        current_date: date,
        prices: dict[str, float] | None = None,
    ) -> RebalanceOrders:
        """
        Compute trades to move from current to target weights.

        Args:
            current_weights: Current portfolio weights
            target_weights: Target portfolio weights
            current_date: Date of rebalance
            prices: Optional prices for cost-aware optimization

        Returns:
            RebalanceOrders with trades to execute
        """
        ...

    @abstractmethod
    def run(
        self,
        target_weights: TargetWeights,
        portfolio_data: PortfolioData,
        initial_weights: dict[str, float] | None = None,
    ) -> RebalanceResult:
        """
        Run rebalancing simulation over full period.

        Args:
            target_weights: Target weights at each date
            portfolio_data: Price data for drift calculation
            initial_weights: Starting weights (default: equal or first target)

        Returns:
            Complete rebalancing history
        """
        ...

    @classmethod
    @abstractmethod
    def calendar(
        cls,
        frequency: RebalanceFrequency,
        turnover_limit: float | None = None,
    ) -> Rebalancer:
        """
        Create calendar-based rebalancer.

        Args:
            frequency: Rebalancing frequency
            turnover_limit: Optional turnover constraint

        Returns:
            Configured Rebalancer
        """
        ...

    @classmethod
    @abstractmethod
    def drift(
        cls,
        threshold: float,
        turnover_limit: float | None = None,
    ) -> Rebalancer:
        """
        Create drift-based rebalancer.

        Args:
            threshold: Drift threshold
            turnover_limit: Optional turnover constraint

        Returns:
            Configured Rebalancer
        """
        ...

    @classmethod
    @abstractmethod
    def hybrid(
        cls,
        calendar_frequency: RebalanceFrequency,
        drift_threshold: float,
        turnover_limit: float | None = None,
    ) -> Rebalancer:
        """
        Create hybrid calendar + drift rebalancer.

        Args:
            calendar_frequency: Scheduled rebalancing frequency
            drift_threshold: Drift override threshold
            turnover_limit: Optional turnover constraint

        Returns:
            Configured Rebalancer
        """
        ...
