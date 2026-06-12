"""Order model: types, time-in-force, validated lifecycle state machine."""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from enum import StrEnum

_ORDER_COUNTER = itertools.count(1)


class OrderSide(StrEnum):
    BUY = "buy"
    SELL = "sell"

    @property
    def sign(self) -> int:
        return 1 if self is OrderSide.BUY else -1


class OrderType(StrEnum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class TimeInForce(StrEnum):
    DAY = "day"
    GTC = "gtc"
    IOC = "ioc"
    FOK = "fok"


class OrderStatus(StrEnum):
    NEW = "new"
    ACCEPTED = "accepted"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


#: Validated lifecycle: NEW -> ACCEPTED -> PARTIALLY_FILLED -> FILLED, with
#: CANCELLED / REJECTED / EXPIRED terminal branches.
VALID_TRANSITIONS: dict[OrderStatus, frozenset[OrderStatus]] = {
    OrderStatus.NEW: frozenset(
        {OrderStatus.ACCEPTED, OrderStatus.REJECTED, OrderStatus.CANCELLED}
    ),
    OrderStatus.ACCEPTED: frozenset(
        {
            OrderStatus.PARTIALLY_FILLED,
            OrderStatus.FILLED,
            OrderStatus.CANCELLED,
            OrderStatus.REJECTED,
            OrderStatus.EXPIRED,
        }
    ),
    OrderStatus.PARTIALLY_FILLED: frozenset(
        {
            OrderStatus.PARTIALLY_FILLED,
            OrderStatus.FILLED,
            OrderStatus.CANCELLED,
            OrderStatus.EXPIRED,
        }
    ),
    OrderStatus.FILLED: frozenset(),
    OrderStatus.CANCELLED: frozenset(),
    OrderStatus.REJECTED: frozenset(),
    OrderStatus.EXPIRED: frozenset(),
}

TERMINAL_STATUSES = frozenset(
    {
        OrderStatus.FILLED,
        OrderStatus.CANCELLED,
        OrderStatus.REJECTED,
        OrderStatus.EXPIRED,
    }
)


class InvalidTransitionError(ValueError):
    """Raised when an order is moved through an illegal status transition."""


class OrderValidationError(ValueError):
    """Raised when an order is constructed with inconsistent fields."""


def _next_order_id() -> str:
    return f"ORD-{next(_ORDER_COUNTER):06d}"


@dataclass
class Fill:
    """A single execution against an order."""

    order_id: str
    symbol: str
    side: OrderSide
    quantity: float
    price: float
    commission: float
    bar_index: int

    @property
    def notional(self) -> float:
        return self.quantity * self.price


@dataclass
class Order:
    """A single order with a validated lifecycle state machine."""

    symbol: str
    side: OrderSide
    quantity: float
    order_type: OrderType = OrderType.MARKET
    limit_price: float | None = None
    stop_price: float | None = None
    time_in_force: TimeInForce = TimeInForce.DAY
    algo: str | None = None
    parent_id: str | None = None

    # Managed by the system.
    order_id: str = field(default_factory=_next_order_id)
    status: OrderStatus = OrderStatus.NEW
    filled_quantity: float = 0.0
    avg_fill_price: float = 0.0
    commission: float = 0.0
    created_index: int | None = None
    submitted_index: int | None = None
    decision_price: float | None = None
    arrival_price: float | None = None
    triggered: bool = False
    reject_reason: str | None = None
    fills: list[Fill] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.quantity <= 0:
            raise OrderValidationError("quantity must be positive")
        if self.order_type in (OrderType.LIMIT, OrderType.STOP_LIMIT):
            if self.limit_price is None or self.limit_price <= 0:
                raise OrderValidationError(
                    f"{self.order_type.value} order requires a positive limit_price"
                )
        if self.order_type in (OrderType.STOP, OrderType.STOP_LIMIT):
            if self.stop_price is None or self.stop_price <= 0:
                raise OrderValidationError(
                    f"{self.order_type.value} order requires a positive stop_price"
                )
        if self.order_type == OrderType.MARKET and self.limit_price is not None:
            raise OrderValidationError("market order must not carry a limit_price")

    # ------------------------------------------------------------------ state
    @property
    def remaining_quantity(self) -> float:
        return self.quantity - self.filled_quantity

    @property
    def is_terminal(self) -> bool:
        return self.status in TERMINAL_STATUSES

    @property
    def is_active(self) -> bool:
        return self.status in (OrderStatus.ACCEPTED, OrderStatus.PARTIALLY_FILLED)

    def transition_to(self, new_status: OrderStatus) -> None:
        if new_status not in VALID_TRANSITIONS[self.status]:
            raise InvalidTransitionError(
                f"order {self.order_id}: illegal transition "
                f"{self.status.value} -> {new_status.value}"
            )
        self.status = new_status

    # ------------------------------------------------------------------ fills
    def apply_fill(self, quantity: float, price: float, commission: float, bar_index: int) -> Fill:
        """Apply an execution; updates average price and lifecycle status."""
        if quantity <= 0:
            raise OrderValidationError("fill quantity must be positive")
        if quantity > self.remaining_quantity + 1e-9:
            raise OrderValidationError(
                f"fill quantity {quantity} exceeds remaining {self.remaining_quantity}"
            )
        fill = Fill(
            order_id=self.order_id,
            symbol=self.symbol,
            side=self.side,
            quantity=quantity,
            price=price,
            commission=commission,
            bar_index=bar_index,
        )
        prev_filled = self.filled_quantity
        self.filled_quantity = prev_filled + quantity
        self.avg_fill_price = (
            prev_filled * self.avg_fill_price + quantity * price
        ) / self.filled_quantity
        self.commission += commission
        self.fills.append(fill)

        if self.remaining_quantity <= 1e-9:
            self.filled_quantity = self.quantity
            self.transition_to(OrderStatus.FILLED)
        else:
            self.transition_to(OrderStatus.PARTIALLY_FILLED)
        return fill
