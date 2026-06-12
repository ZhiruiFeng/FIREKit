"""Order management: submission with risk checks, positions, PnL, cash, blotter."""

from __future__ import annotations

from dataclasses import dataclass, field

from executioncore.broker import Broker, OrderEvent, PaperBroker
from executioncore.orders import Fill, Order, OrderSide, OrderStatus
from executioncore.risk import RiskContext, RiskValidator


@dataclass
class Position:
    """Average-cost position with realized PnL tracking (supports shorts)."""

    symbol: str
    quantity: float = 0.0
    avg_cost: float = 0.0
    realized_pnl: float = 0.0

    def apply_fill(self, side: OrderSide, quantity: float, price: float) -> float:
        """Update the position for a fill; returns realized PnL of this fill."""
        delta = side.sign * quantity
        pos = self.quantity
        realized = 0.0
        if pos == 0.0 or pos * delta > 0:
            new_qty = pos + delta
            self.avg_cost = (abs(pos) * self.avg_cost + quantity * price) / abs(new_qty)
            self.quantity = new_qty
        else:
            closed = min(abs(delta), abs(pos))
            direction = 1.0 if pos > 0 else -1.0
            realized = (price - self.avg_cost) * closed * direction
            new_qty = pos + delta
            if pos * new_qty < 0:  # crossed through zero; remainder opens at fill price
                self.avg_cost = price
            elif new_qty == 0.0:
                self.avg_cost = 0.0
            self.quantity = new_qty
        self.realized_pnl += realized
        return realized

    def unrealized_pnl(self, mark: float) -> float:
        return (mark - self.avg_cost) * self.quantity

    def market_value(self, mark: float) -> float:
        return self.quantity * mark


@dataclass
class PortfolioSnapshot:
    cash: float
    equity: float
    realized_pnl: float
    unrealized_pnl: float
    total_commission: float
    positions: dict[str, float] = field(default_factory=dict)


class OrderManager:
    """OMS: routes orders through risk checks to a broker and keeps the books."""

    def __init__(
        self,
        broker: Broker,
        cash: float = 1_000_000.0,
        validators: tuple[RiskValidator, ...] = (),
    ):
        self.broker = broker
        self.starting_cash = cash
        self.cash = cash
        self.validators = list(validators)
        self.positions: dict[str, Position] = {}
        self.orders: dict[str, Order] = {}
        self.blotter: list[OrderEvent] = []
        self.total_commission = 0.0
        if isinstance(broker, PaperBroker):
            broker.fill_listeners.append(self._on_fill)
            broker.event_listeners.append(self.blotter.append)

    # ----------------------------------------------------------- order flow
    def submit(self, order: Order) -> Order:
        """Run pre-trade risk checks, then hand the order to the broker."""
        order.decision_price = self.broker.mark_price(order.symbol)
        ctx = RiskContext(
            reference_price=order.decision_price,
            position_quantity=self.position(order.symbol).quantity,
            cash=self.cash,
            equity=self.equity(),
        )
        for validator in self.validators:
            result = validator.validate(order, ctx)
            if not result.ok:
                order.transition_to(OrderStatus.REJECTED)
                order.reject_reason = f"{validator.name}: {result.reason}"
                self.orders[order.order_id] = order
                self.blotter.append(
                    OrderEvent(
                        bar_index=getattr(self.broker, "current_index", -1),
                        order_id=order.order_id,
                        event="rejected",
                        symbol=order.symbol,
                        side=order.side.value,
                        quantity=order.quantity,
                        detail=order.reject_reason,
                    )
                )
                return order
        self.orders[order.order_id] = order
        self.broker.submit_order(order)
        return order

    def cancel(self, order_id: str) -> bool:
        return self.broker.cancel_order(order_id)

    def replace(self, order_id: str, **kwargs: float) -> bool:
        return self.broker.replace_order(order_id, **kwargs)

    # ------------------------------------------------------------- accounting
    def _on_fill(self, fill: Fill) -> None:
        position = self.position(fill.symbol)
        position.apply_fill(fill.side, fill.quantity, fill.price)
        self.cash -= fill.side.sign * fill.quantity * fill.price
        self.cash -= fill.commission
        self.total_commission += fill.commission

    def position(self, symbol: str) -> Position:
        if symbol not in self.positions:
            self.positions[symbol] = Position(symbol=symbol)
        return self.positions[symbol]

    def realized_pnl(self) -> float:
        return sum(p.realized_pnl for p in self.positions.values())

    def unrealized_pnl(self) -> float:
        return sum(
            p.unrealized_pnl(self.broker.mark_price(sym)) for sym, p in self.positions.items()
        )

    def equity(self) -> float:
        market_value = sum(
            p.market_value(self.broker.mark_price(sym)) for sym, p in self.positions.items()
        )
        return self.cash + market_value

    def snapshot(self) -> PortfolioSnapshot:
        return PortfolioSnapshot(
            cash=self.cash,
            equity=self.equity(),
            realized_pnl=self.realized_pnl(),
            unrealized_pnl=self.unrealized_pnl(),
            total_commission=self.total_commission,
            positions={s: p.quantity for s, p in self.positions.items()},
        )
