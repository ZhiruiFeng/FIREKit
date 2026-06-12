"""Broker abstraction and the PaperBroker bar-by-bar fill simulator."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass

from executioncore.costs import CommissionModel, NoCommission, NoSlippage, SlippageModel
from executioncore.market import Bar, PriceFeed
from executioncore.orders import (
    Fill,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    TimeInForce,
)


@dataclass
class OrderEvent:
    """One row of the audit trail (blotter)."""

    bar_index: int
    order_id: str
    event: str  # accepted | triggered | fill | cancelled | replaced | expired | rejected
    symbol: str
    side: str
    quantity: float
    price: float | None = None
    detail: str = ""


class Broker(ABC):
    """Minimal broker interface; a live adapter would implement the same API."""

    @abstractmethod
    def submit_order(self, order: Order) -> str:
        """Accept an order, returning its id."""

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """Cancel a working order; returns False if not cancellable."""

    @abstractmethod
    def replace_order(
        self,
        order_id: str,
        quantity: float | None = None,
        limit_price: float | None = None,
        stop_price: float | None = None,
    ) -> bool:
        """Amend a working order in place."""

    @abstractmethod
    def get_order(self, order_id: str) -> Order:
        """Look up an order by id."""

    @abstractmethod
    def open_orders(self) -> list[Order]:
        """All currently working orders."""

    @abstractmethod
    def mark_price(self, symbol: str) -> float:
        """Current mark for a symbol."""


class PaperBroker(Broker):
    """Simulates fills against a :class:`PriceFeed`, one bar at a time.

    Semantics:

    - Orders are submitted *between* bars; an order becomes eligible at bar
      ``submission_point + 1 + latency_bars`` (it can never fill on data that
      was already known at submission).
    - Market orders fill at the eligible bar's open plus adverse slippage.
    - Limit buys fill when ``bar.low <= limit`` at ``min(bar.open, limit)``
      (sells mirrored); slippage never pushes a fill through the limit price.
    - Stop orders trigger when the bar trades through the stop and then fill
      as market (or as limit for stop-limit) from the trigger bar onward.
    - Per-bar fill size is capped at ``participation_cap * bar.volume``
      (per order), producing partial fills that continue on later bars.
    - IOC fills whatever is possible on the first eligible bar and cancels
      the rest; FOK cancels unless the full size fits under the cap.
    - DAY orders expire at session end (the feed is one session); GTC orders
      stay working.
    """

    def __init__(
        self,
        feed: PriceFeed,
        slippage: SlippageModel | None = None,
        commission: CommissionModel | None = None,
        latency_bars: int = 0,
        participation_cap: float = 1.0,
    ):
        if latency_bars < 0:
            raise ValueError("latency_bars must be >= 0")
        if not 0 < participation_cap <= 1.0:
            raise ValueError("participation_cap must be in (0, 1]")
        self.feed = feed
        self.slippage = slippage or NoSlippage()
        self.commission_model = commission or NoCommission()
        self.latency_bars = latency_bars
        self.participation_cap = participation_cap

        self.current_index = -1  # last processed bar
        self._orders: dict[str, Order] = {}
        self._working: list[str] = []
        self._eligible_at: dict[str, int] = {}
        self.events: list[OrderEvent] = []
        self.fill_listeners: list[Callable[[Fill], None]] = []
        self.event_listeners: list[Callable[[OrderEvent], None]] = []
        self.session_closed = False

    # ------------------------------------------------------------- broker API
    def submit_order(self, order: Order) -> str:
        if order.status != OrderStatus.NEW:
            raise ValueError(f"order {order.order_id} is not NEW")
        if order.symbol not in self.feed.symbols:
            order.transition_to(OrderStatus.REJECTED)
            order.reject_reason = f"unknown symbol {order.symbol}"
            self._emit(order, "rejected", detail=order.reject_reason)
            return order.order_id
        if self.session_closed:
            order.transition_to(OrderStatus.REJECTED)
            order.reject_reason = "session closed"
            self._emit(order, "rejected", detail=order.reject_reason)
            return order.order_id

        order.transition_to(OrderStatus.ACCEPTED)
        order.submitted_index = self.current_index + 1
        order.arrival_price = self.mark_price(order.symbol)
        self._orders[order.order_id] = order
        self._working.append(order.order_id)
        self._eligible_at[order.order_id] = self.current_index + 1 + self.latency_bars
        self._emit(order, "accepted", detail=f"tif={order.time_in_force.value}")
        return order.order_id

    def cancel_order(self, order_id: str) -> bool:
        order = self._orders.get(order_id)
        if order is None or not order.is_active:
            return False
        order.transition_to(OrderStatus.CANCELLED)
        self._working.remove(order_id)
        self._emit(order, "cancelled", detail="user cancel")
        return True

    def replace_order(
        self,
        order_id: str,
        quantity: float | None = None,
        limit_price: float | None = None,
        stop_price: float | None = None,
    ) -> bool:
        order = self._orders.get(order_id)
        if order is None or not order.is_active:
            return False
        if quantity is not None:
            if quantity <= order.filled_quantity:
                return False
            order.quantity = quantity
        if limit_price is not None:
            if order.order_type not in (OrderType.LIMIT, OrderType.STOP_LIMIT):
                return False
            order.limit_price = limit_price
        if stop_price is not None:
            if order.order_type not in (OrderType.STOP, OrderType.STOP_LIMIT):
                return False
            order.stop_price = stop_price
        self._emit(
            order,
            "replaced",
            detail=f"qty={order.quantity} limit={order.limit_price} stop={order.stop_price}",
        )
        return True

    def get_order(self, order_id: str) -> Order:
        return self._orders[order_id]

    def open_orders(self) -> list[Order]:
        return [self._orders[oid] for oid in self._working]

    def mark_price(self, symbol: str) -> float:
        if self.current_index < 0:
            return self.feed.bar(symbol, 0).open
        return self.feed.bar(symbol, self.current_index).close

    # -------------------------------------------------------------- clock
    @property
    def has_next_bar(self) -> bool:
        return self.current_index + 1 < self.feed.n_bars

    def advance(self) -> int:
        """Process the next bar against all working orders."""
        if not self.has_next_bar:
            raise RuntimeError("feed exhausted")
        self.current_index += 1
        for order_id in list(self._working):
            order = self._orders[order_id]
            if self.current_index < self._eligible_at[order_id]:
                continue
            bar = self.feed.bar(order.symbol, self.current_index)
            self._process_order(order, bar)
        return self.current_index

    def run_to_close(self) -> None:
        """Process every remaining bar, then expire DAY orders."""
        while self.has_next_bar:
            self.advance()
        self.end_session()

    def end_session(self) -> None:
        """Expire working DAY orders; GTC orders stay working."""
        self.session_closed = True
        for order_id in list(self._working):
            order = self._orders[order_id]
            if order.time_in_force == TimeInForce.DAY:
                order.transition_to(OrderStatus.EXPIRED)
                self._working.remove(order_id)
                self._emit(order, "expired", detail="end of session")

    # ----------------------------------------------------------- matching
    def _process_order(self, order: Order, bar: Bar) -> None:
        ref = self._marketable_ref(order, bar)
        if ref is None:
            if order.time_in_force in (TimeInForce.IOC, TimeInForce.FOK):
                self._auto_cancel(order, "not marketable on arrival")
            return

        cap_qty = self.participation_cap * bar.volume
        if order.time_in_force == TimeInForce.FOK and order.remaining_quantity > cap_qty:
            self._auto_cancel(order, "FOK size exceeds available liquidity")
            return

        fill_qty = min(order.remaining_quantity, cap_qty)
        if fill_qty <= 0:
            if order.time_in_force in (TimeInForce.IOC, TimeInForce.FOK):
                self._auto_cancel(order, "no liquidity")
            return

        price = self.slippage.fill_price(ref, order.side, fill_qty, bar)
        if order.limit_price is not None:
            # Slippage cannot push a limit fill through its limit price.
            if order.side == OrderSide.BUY:
                price = min(price, order.limit_price)
            else:
                price = max(price, order.limit_price)
        commission = self.commission_model.commission(fill_qty, price)
        fill = order.apply_fill(fill_qty, price, commission, bar.index)
        self._emit(order, "fill", price=price, quantity=fill_qty, detail=f"comm={commission:.4f}")
        for listener in self.fill_listeners:
            listener(fill)

        if order.status == OrderStatus.FILLED:
            self._working.remove(order.order_id)
        elif order.time_in_force == TimeInForce.IOC:
            self._auto_cancel(order, "IOC remainder cancelled")

    def _marketable_ref(self, order: Order, bar: Bar) -> float | None:
        """Reference fill price for this bar, or None if not marketable."""
        if order.order_type == OrderType.MARKET:
            return bar.open

        if order.order_type == OrderType.LIMIT:
            return self._limit_ref(order.side, order.limit_price, bar)

        # Stop and stop-limit: handle the trigger first.
        triggered_this_bar = False
        if not order.triggered:
            if order.side == OrderSide.BUY:
                if bar.high >= order.stop_price:
                    order.triggered = True
                    triggered_this_bar = True
            elif bar.low <= order.stop_price:
                order.triggered = True
                triggered_this_bar = True
            if order.triggered:
                self._emit(order, "triggered", price=order.stop_price)
            else:
                return None

        if order.order_type == OrderType.STOP:
            if triggered_this_bar:
                # Market is at/through the stop when the trigger happens.
                if order.side == OrderSide.BUY:
                    return max(bar.open, order.stop_price)
                return min(bar.open, order.stop_price)
            return bar.open

        # Stop-limit after trigger behaves as a limit order.
        if triggered_this_bar:
            if order.side == OrderSide.BUY:
                if order.limit_price >= order.stop_price:
                    return min(max(bar.open, order.stop_price), order.limit_price)
                return order.limit_price if bar.low <= order.limit_price else None
            if order.limit_price <= order.stop_price:
                return max(min(bar.open, order.stop_price), order.limit_price)
            return order.limit_price if bar.high >= order.limit_price else None
        return self._limit_ref(order.side, order.limit_price, bar)

    @staticmethod
    def _limit_ref(side: OrderSide, limit_price: float, bar: Bar) -> float | None:
        if side == OrderSide.BUY:
            if bar.low <= limit_price:
                return min(bar.open, limit_price)
            return None
        if bar.high >= limit_price:
            return max(bar.open, limit_price)
        return None

    # ------------------------------------------------------------- events
    def _auto_cancel(self, order: Order, reason: str) -> None:
        order.transition_to(OrderStatus.CANCELLED)
        if order.order_id in self._working:
            self._working.remove(order.order_id)
        self._emit(order, "cancelled", detail=reason)

    def _emit(
        self,
        order: Order,
        event: str,
        price: float | None = None,
        quantity: float | None = None,
        detail: str = "",
    ) -> None:
        evt = OrderEvent(
            bar_index=self.current_index,
            order_id=order.order_id,
            event=event,
            symbol=order.symbol,
            side=order.side.value,
            quantity=quantity if quantity is not None else order.quantity,
            price=price,
            detail=detail,
        )
        self.events.append(evt)
        for listener in self.event_listeners:
            listener(evt)
