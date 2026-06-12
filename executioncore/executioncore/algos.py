"""Execution algorithms: TWAP / VWAP slicing of a parent order into child orders."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from executioncore.broker import PaperBroker
from executioncore.market import PriceFeed
from executioncore.oms import OrderManager
from executioncore.orders import Order, OrderType


class ExecutionAlgo(ABC):
    """Turns a parent quantity into a (bar_index, child_qty) schedule."""

    name: str = "algo"

    @abstractmethod
    def schedule(
        self,
        quantity: float,
        start_index: int,
        n_slices: int,
        feed: PriceFeed | None = None,
        symbol: str | None = None,
    ) -> list[tuple[int, float]]:
        """Child slices on consecutive bars; quantities sum exactly to parent."""

    @staticmethod
    def _exact_sum(quantities: list[float], total: float) -> list[float]:
        """Adjust the last slice so the schedule sums exactly to ``total``."""
        if not quantities:
            return quantities
        quantities[-1] = total - sum(quantities[:-1])
        return quantities


class TWAP(ExecutionAlgo):
    """Time-weighted: equal slices across N consecutive bars."""

    name = "TWAP"

    def schedule(
        self,
        quantity: float,
        start_index: int,
        n_slices: int,
        feed: PriceFeed | None = None,
        symbol: str | None = None,
    ) -> list[tuple[int, float]]:
        if n_slices <= 0:
            raise ValueError("n_slices must be positive")
        per_slice = quantity / n_slices
        quantities = self._exact_sum([per_slice] * n_slices, quantity)
        return [(start_index + i, q) for i, q in enumerate(quantities)]


class VWAP(ExecutionAlgo):
    """Volume-weighted: slices proportional to an expected volume profile.

    If no profile is supplied, the feed's bar volumes over the execution
    window are used as the expected profile (a stand-in for a historical
    intraday volume curve).
    """

    name = "VWAP"

    def __init__(self, volume_profile: Sequence[float] | None = None):
        if volume_profile is not None:
            if len(volume_profile) == 0 or any(v < 0 for v in volume_profile):
                raise ValueError("volume_profile must be non-empty and non-negative")
            if sum(volume_profile) <= 0:
                raise ValueError("volume_profile must have positive mass")
        self.volume_profile = list(volume_profile) if volume_profile is not None else None

    def schedule(
        self,
        quantity: float,
        start_index: int,
        n_slices: int,
        feed: PriceFeed | None = None,
        symbol: str | None = None,
    ) -> list[tuple[int, float]]:
        if n_slices <= 0:
            raise ValueError("n_slices must be positive")
        if self.volume_profile is not None:
            if len(self.volume_profile) != n_slices:
                raise ValueError("volume_profile length must equal n_slices")
            weights = list(self.volume_profile)
        else:
            if feed is None or symbol is None:
                raise ValueError("VWAP without an explicit profile needs feed and symbol")
            weights = [
                feed.bar(symbol, start_index + i).volume for i in range(n_slices)
            ]
        total_weight = sum(weights)
        quantities = self._exact_sum([quantity * w / total_weight for w in weights], quantity)
        return [(start_index + i, q) for i, q in enumerate(quantities)]


class AlgoExecutor:
    """Drives an algo schedule: submits child market orders as the clock advances.

    The parent order is a virtual record (never sent to the broker); children
    carry ``parent_id`` and ``algo`` tags so analytics can roll them up.
    """

    def __init__(
        self,
        oms: OrderManager,
        parent: Order,
        algo: ExecutionAlgo,
        start_index: int,
        n_slices: int,
        decision_index: int | None = None,
    ):
        if parent.order_type != OrderType.MARKET:
            raise ValueError("parent order must be a market-style instruction")
        self.oms = oms
        self.parent = parent
        self.algo = algo
        parent.algo = algo.name
        feed = getattr(oms.broker, "feed", None)
        if decision_index is not None:
            # The trading decision is stamped at this bar's close (useful when
            # scripting a session where the decision precedes execution).
            if feed is None:
                raise ValueError("decision_index requires a feed-backed broker")
            parent.created_index = decision_index
            parent.decision_price = feed.bar(parent.symbol, decision_index).close
        else:
            parent.created_index = getattr(oms.broker, "current_index", -1)
            parent.decision_price = oms.broker.mark_price(parent.symbol)
        self.schedule = algo.schedule(
            parent.quantity, start_index, n_slices, feed=feed, symbol=parent.symbol
        )
        self._due: dict[int, float] = dict(self.schedule)
        self.children: list[Order] = []

    def on_clock(self, next_bar_index: int) -> Order | None:
        """Submit the child due to execute on ``next_bar_index``, if any."""
        qty = self._due.pop(next_bar_index, None)
        if qty is None or qty <= 0:
            return None
        child = Order(
            symbol=self.parent.symbol,
            side=self.parent.side,
            quantity=qty,
            order_type=OrderType.MARKET,
            time_in_force=self.parent.time_in_force,
            algo=self.algo.name,
            parent_id=self.parent.order_id,
        )
        self.children.append(child)
        self.oms.submit(child)
        return child

    @property
    def is_complete(self) -> bool:
        return not self._due and all(c.is_terminal for c in self.children)

    @property
    def filled_quantity(self) -> float:
        return sum(c.filled_quantity for c in self.children)

    @property
    def avg_fill_price(self) -> float:
        filled = self.filled_quantity
        if filled == 0:
            return 0.0
        return sum(c.filled_quantity * c.avg_fill_price for c in self.children) / filled

    @property
    def total_commission(self) -> float:
        return sum(c.commission for c in self.children)

    @property
    def arrival_price(self) -> float | None:
        for child in self.children:
            if child.arrival_price is not None:
                return child.arrival_price
        return None

    @property
    def window(self) -> tuple[int, int]:
        """(first_bar, last_bar + 1) of the schedule."""
        bars = [b for b, _ in self.schedule]
        return min(bars), max(bars) + 1


def run_session(
    broker: PaperBroker,
    submissions: dict[int, list[Order]] | None = None,
    executors: Sequence[AlgoExecutor] = (),
    oms: OrderManager | None = None,
) -> None:
    """Run a full session: scheduled submissions + algo executors + matching.

    ``submissions[i]`` are orders submitted just before bar ``i`` is processed
    (they become eligible at bar ``i`` with zero latency).
    """
    submissions = submissions or {}
    while broker.has_next_bar:
        next_index = broker.current_index + 1
        for executor in executors:
            executor.on_clock(next_index)
        for order in submissions.get(next_index, []):
            if oms is not None:
                oms.submit(order)
            else:
                broker.submit_order(order)
        broker.advance()
    broker.end_session()
