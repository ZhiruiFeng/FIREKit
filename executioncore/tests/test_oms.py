"""OrderManager: positions, PnL, cash accounting, blotter, risk hooks."""

import pytest

from executioncore.broker import PaperBroker
from executioncore.costs import PerShareCommission
from executioncore.oms import OrderManager, Position
from executioncore.orders import Order, OrderSide, OrderStatus, OrderType
from executioncore.risk import MaxNotional, MaxOrderSize, PriceBand
from tests.conftest import SYMBOL, make_feed


def flat_feed(price: float = 100.0, n: int = 6):
    return make_feed([(price, price + 0.5, price - 0.5, price, 10_000.0)] * n)


class TestPositionMath:
    def test_average_cost_across_buys(self):
        pos = Position(SYMBOL)
        pos.apply_fill(OrderSide.BUY, 100, 10.0)
        pos.apply_fill(OrderSide.BUY, 50, 13.0)
        assert pos.quantity == 150
        assert pos.avg_cost == pytest.approx(11.0)
        assert pos.realized_pnl == 0.0

    def test_realized_pnl_on_partial_sell(self):
        pos = Position(SYMBOL)
        pos.apply_fill(OrderSide.BUY, 100, 10.0)
        realized = pos.apply_fill(OrderSide.SELL, 40, 12.0)
        assert realized == pytest.approx(80.0)
        assert pos.quantity == 60
        assert pos.avg_cost == pytest.approx(10.0)  # unchanged on reduction

    def test_short_then_cover(self):
        pos = Position(SYMBOL)
        pos.apply_fill(OrderSide.SELL, 100, 50.0)
        assert pos.quantity == -100
        assert pos.avg_cost == pytest.approx(50.0)
        realized = pos.apply_fill(OrderSide.BUY, 100, 45.0)
        assert realized == pytest.approx(500.0)
        assert pos.quantity == 0
        assert pos.avg_cost == 0.0

    def test_cross_through_zero_resets_basis(self):
        pos = Position(SYMBOL)
        pos.apply_fill(OrderSide.BUY, 100, 10.0)
        realized = pos.apply_fill(OrderSide.SELL, 150, 12.0)
        assert realized == pytest.approx(200.0)  # only 100 shares closed
        assert pos.quantity == -50
        assert pos.avg_cost == pytest.approx(12.0)  # new short basis

    def test_unrealized_pnl(self):
        pos = Position(SYMBOL)
        pos.apply_fill(OrderSide.BUY, 100, 10.0)
        assert pos.unrealized_pnl(11.5) == pytest.approx(150.0)
        short = Position(SYMBOL)
        short.apply_fill(OrderSide.SELL, 100, 10.0)
        assert short.unrealized_pnl(9.0) == pytest.approx(100.0)


class TestAccounting:
    def test_cash_and_commission_on_round_trip(self):
        broker = PaperBroker(flat_feed(), commission=PerShareCommission(0.01))
        oms = OrderManager(broker, cash=100_000.0)
        oms.submit(Order(SYMBOL, OrderSide.BUY, 500))
        broker.advance()
        assert oms.cash == pytest.approx(100_000 - 500 * 100.0 - 5.0)
        oms.submit(Order(SYMBOL, OrderSide.SELL, 500))
        broker.advance()
        assert oms.cash == pytest.approx(100_000 - 10.0)
        assert oms.total_commission == pytest.approx(10.0)
        assert oms.realized_pnl() == pytest.approx(0.0)

    def test_equity_marks_open_positions(self):
        feed = make_feed(
            [
                (100.0, 100.5, 99.5, 100.0, 10_000.0),
                (100.0, 102.5, 99.5, 102.0, 10_000.0),
            ]
        )
        broker = PaperBroker(feed)
        oms = OrderManager(broker, cash=100_000.0)
        oms.submit(Order(SYMBOL, OrderSide.BUY, 100))
        broker.advance()  # filled at 100
        broker.advance()  # mark moves to 102
        assert oms.unrealized_pnl() == pytest.approx(200.0)
        assert oms.equity() == pytest.approx(100_000.0 + 200.0)

    def test_blotter_audit_trail(self):
        broker = PaperBroker(flat_feed())
        oms = OrderManager(broker)
        order = oms.submit(Order(SYMBOL, OrderSide.BUY, 100))
        broker.advance()
        events = [(e.event, e.order_id) for e in oms.blotter]
        assert ("accepted", order.order_id) in events
        assert ("fill", order.order_id) in events


class TestRiskHooks:
    def test_max_order_size_rejects_before_broker(self):
        broker = PaperBroker(flat_feed())
        oms = OrderManager(broker, validators=(MaxOrderSize(1_000),))
        order = oms.submit(Order(SYMBOL, OrderSide.BUY, 5_000))
        assert order.status == OrderStatus.REJECTED
        assert "max_order_size" in order.reject_reason
        assert broker.open_orders() == []
        assert any(e.event == "rejected" for e in oms.blotter)

    def test_max_notional_uses_limit_price(self):
        broker = PaperBroker(flat_feed())
        oms = OrderManager(broker, validators=(MaxNotional(50_000),))
        bad = oms.submit(
            Order(SYMBOL, OrderSide.BUY, 600, OrderType.LIMIT, limit_price=100.0)
        )
        assert bad.status == OrderStatus.REJECTED
        ok = oms.submit(Order(SYMBOL, OrderSide.BUY, 400, OrderType.LIMIT, limit_price=100.0))
        assert ok.status == OrderStatus.ACCEPTED

    def test_price_band_fat_finger(self):
        broker = PaperBroker(flat_feed())
        oms = OrderManager(broker, validators=(PriceBand(0.05),))
        bad = oms.submit(
            Order(SYMBOL, OrderSide.BUY, 100, OrderType.LIMIT, limit_price=120.0)
        )
        assert bad.status == OrderStatus.REJECTED
        assert "price_band" in bad.reject_reason
        near = oms.submit(
            Order(SYMBOL, OrderSide.BUY, 100, OrderType.LIMIT, limit_price=101.0)
        )
        assert near.status == OrderStatus.ACCEPTED
        market = oms.submit(Order(SYMBOL, OrderSide.BUY, 100))
        assert market.status == OrderStatus.ACCEPTED

    def test_passing_orders_reach_broker(self):
        broker = PaperBroker(flat_feed())
        oms = OrderManager(
            broker, validators=(MaxOrderSize(1_000), MaxNotional(150_000), PriceBand(0.05))
        )
        order = oms.submit(Order(SYMBOL, OrderSide.BUY, 900))
        broker.advance()
        assert order.status == OrderStatus.FILLED
