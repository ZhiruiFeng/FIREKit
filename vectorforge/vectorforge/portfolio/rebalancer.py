"""
Portfolio Rebalancing

Logic for determining when and how to rebalance portfolios.
Supports calendar-based, drift-based, and hybrid triggers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from typing import TYPE_CHECKING

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
        if not self.orders:
            return pd.DataFrame(columns=["date", "turnover", "n_trades", "constrained"])

        records = []
        for order in self.orders:
            records.append(
                {
                    "date": order.date,
                    "turnover": order.turnover,
                    "n_trades": order.n_trades,
                    "constrained": order.constrained,
                }
            )
        return pd.DataFrame(records)


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
        pass


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
        self._frequency = frequency
        self._last_period: tuple[int, int] | None = None

    @property
    def frequency(self) -> RebalanceFrequency:
        """Rebalancing frequency."""
        return self._frequency

    def should_rebalance(
        self,
        current_date: date,
        current_weights: dict[str, float],
        target_weights: dict[str, float],
        last_rebalance: date | None,
    ) -> bool:
        """Determine if calendar trigger fires."""
        if self._frequency == RebalanceFrequency.DAILY:
            return True

        period = self._get_period(current_date)

        if last_rebalance is None:
            # First rebalance
            self._last_period = period
            return True

        last_period = self._get_period(last_rebalance)

        if period != last_period:
            self._last_period = period
            return True

        return False

    def _get_period(self, d: date) -> tuple[int, int]:
        """Get period identifier for the date."""
        if self._frequency == RebalanceFrequency.WEEKLY:
            # ISO week number
            return (d.year, d.isocalendar()[1])
        elif self._frequency == RebalanceFrequency.MONTHLY:
            return (d.year, d.month)
        elif self._frequency == RebalanceFrequency.QUARTERLY:
            return (d.year, (d.month - 1) // 3)
        elif self._frequency == RebalanceFrequency.ANNUAL:
            return (d.year, 0)
        else:
            return (d.year, d.timetuple().tm_yday)


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
        self._threshold = threshold
        self._measure = measure

    @property
    def threshold(self) -> float:
        """Drift threshold."""
        return self._threshold

    def should_rebalance(
        self,
        current_date: date,
        current_weights: dict[str, float],
        target_weights: dict[str, float],
        last_rebalance: date | None,
    ) -> bool:
        """Determine if drift exceeds threshold."""
        drifts = self.get_drifts(current_weights, target_weights)

        if not drifts:
            return False

        max_drift = max(abs(d) for d in drifts.values())
        return max_drift >= self._threshold

    def get_drifts(
        self,
        current_weights: dict[str, float],
        target_weights: dict[str, float],
    ) -> dict[str, float]:
        """Calculate drift for each position."""
        drifts: dict[str, float] = {}

        all_symbols = set(current_weights.keys()) | set(target_weights.keys())

        for symbol in all_symbols:
            current = current_weights.get(symbol, 0.0)
            target = target_weights.get(symbol, 0.0)

            if self._measure == "absolute":
                drift = current - target
            else:  # relative
                if abs(target) > 1e-6:
                    drift = (current - target) / target
                else:
                    drift = current if current != 0 else 0.0

            drifts[symbol] = drift

        return drifts


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
        self._triggers = triggers

    @property
    def triggers(self) -> list[RebalanceTrigger]:
        """List of triggers."""
        return self._triggers

    def should_rebalance(
        self,
        current_date: date,
        current_weights: dict[str, float],
        target_weights: dict[str, float],
        last_rebalance: date | None,
    ) -> bool:
        """Rebalance if any trigger fires."""
        return any(
            t.should_rebalance(current_date, current_weights, target_weights, last_rebalance)
            for t in self._triggers
        )


class Rebalancer:
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

    def __init__(
        self,
        trigger: RebalanceTrigger,
        turnover_limit: float | None = None,
    ):
        """
        Initialize rebalancer.

        Args:
            trigger: Trigger determining when to rebalance
            turnover_limit: Maximum turnover per rebalance (None = unlimited)
        """
        self._trigger = trigger
        self._turnover_limit = turnover_limit

    @property
    def trigger(self) -> RebalanceTrigger:
        """Rebalancing trigger."""
        return self._trigger

    @property
    def turnover_limit(self) -> float | None:
        """Maximum turnover per rebalance (None = unlimited)."""
        return self._turnover_limit

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
        # Calculate all weight changes
        all_symbols = set(current_weights.keys()) | set(target_weights.keys())
        trades: dict[str, float] = {}

        for symbol in all_symbols:
            current = current_weights.get(symbol, 0.0)
            target = target_weights.get(symbol, 0.0)
            trades[symbol] = target - current

        # Calculate raw turnover
        turnover = sum(abs(v) for v in trades.values()) / 2

        constrained = False

        # Apply turnover limit if specified
        if self._turnover_limit is not None and turnover > self._turnover_limit:
            scale = self._turnover_limit / turnover
            trades = {s: v * scale for s, v in trades.items()}
            turnover = self._turnover_limit
            constrained = True

        # Compute actual to_weights
        to_weights = {}
        for symbol in all_symbols:
            to_weights[symbol] = current_weights.get(symbol, 0.0) + trades.get(symbol, 0.0)

        return RebalanceOrders(
            date=current_date,
            trades=trades,
            from_weights=current_weights.copy(),
            to_weights=to_weights,
            turnover=turnover,
            constrained=constrained,
        )

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
        result = RebalanceResult()

        n_symbols = portfolio_data.n_symbols
        n_dates = portfolio_data.n_dates
        symbols = portfolio_data.symbols

        # Initialize weights
        if initial_weights is None:
            initial_weights = target_weights.get_weights_at(portfolio_data.dates[0])

        current_weights = initial_weights.copy()
        last_rebalance: date | None = None

        # Get close prices for drift simulation
        close_prices = portfolio_data.close()

        for t in range(n_dates):
            current_date = portfolio_data.dates[t]
            if hasattr(current_date, "date"):
                current_date = current_date.date()

            target_w = target_weights.get_weights_at(portfolio_data.dates[t])

            # Check if we should rebalance
            if self._trigger.should_rebalance(
                current_date, current_weights, target_w, last_rebalance
            ):
                orders = self.compute_trades(current_weights, target_w, current_date)
                result.orders.append(orders)

                current_weights = orders.to_weights.copy()
                last_rebalance = current_date

            # Simulate drift from returns (if not last day)
            if t < n_dates - 1:
                for i, symbol in enumerate(symbols):
                    if symbol in current_weights:
                        ret = (close_prices[i, t + 1] - close_prices[i, t]) / close_prices[i, t]
                        current_weights[symbol] = current_weights.get(symbol, 0.0) * (1 + ret)

                # Normalize
                total = sum(current_weights.values())
                if total > 0:
                    current_weights = {s: w / total for s, w in current_weights.items()}

        return result

    @classmethod
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
        return cls(
            trigger=CalendarTrigger(frequency),
            turnover_limit=turnover_limit,
        )

    @classmethod
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
        return cls(
            trigger=DriftTrigger(threshold),
            turnover_limit=turnover_limit,
        )

    @classmethod
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
        return cls(
            trigger=HybridTrigger(
                [
                    CalendarTrigger(calendar_frequency),
                    DriftTrigger(drift_threshold),
                ]
            ),
            turnover_limit=turnover_limit,
        )
