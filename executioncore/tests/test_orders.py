"""Order model and lifecycle state-machine tests."""

import pytest

from executioncore.orders import (
    InvalidTransitionError,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    OrderValidationError,
)


def buy(qty: float = 100, **kwargs) -> Order:
    return Order("TEST", OrderSide.BUY, qty, **kwargs)


class TestOrderValidation:
    def test_order_ids_are_unique(self):
        assert buy().order_id != buy().order_id

    def test_non_positive_quantity_rejected(self):
        with pytest.raises(OrderValidationError):
            buy(0)
        with pytest.raises(OrderValidationError):
            buy(-5)

    def test_limit_requires_limit_price(self):
        with pytest.raises(OrderValidationError):
            buy(order_type=OrderType.LIMIT)

    def test_stop_requires_stop_price(self):
        with pytest.raises(OrderValidationError):
            buy(order_type=OrderType.STOP)

    def test_stop_limit_requires_both_prices(self):
        with pytest.raises(OrderValidationError):
            buy(order_type=OrderType.STOP_LIMIT, limit_price=101.0)
        with pytest.raises(OrderValidationError):
            buy(order_type=OrderType.STOP_LIMIT, stop_price=101.0)

    def test_market_must_not_have_limit_price(self):
        with pytest.raises(OrderValidationError):
            buy(order_type=OrderType.MARKET, limit_price=100.0)


class TestStateMachine:
    def test_full_lifecycle(self):
        order = buy()
        assert order.status == OrderStatus.NEW
        order.transition_to(OrderStatus.ACCEPTED)
        order.apply_fill(100, 50.0, 0.0, bar_index=0)
        assert order.status == OrderStatus.FILLED
        assert order.is_terminal

    def test_partial_then_full(self):
        order = buy(100)
        order.transition_to(OrderStatus.ACCEPTED)
        order.apply_fill(40, 50.0, 0.0, bar_index=0)
        assert order.status == OrderStatus.PARTIALLY_FILLED
        assert order.remaining_quantity == 60
        order.apply_fill(60, 51.0, 0.0, bar_index=1)
        assert order.status == OrderStatus.FILLED
        assert order.remaining_quantity == 0

    def test_new_cannot_jump_to_filled(self):
        with pytest.raises(InvalidTransitionError):
            buy().transition_to(OrderStatus.FILLED)

    def test_terminal_states_are_frozen(self):
        order = buy()
        order.transition_to(OrderStatus.ACCEPTED)
        order.transition_to(OrderStatus.CANCELLED)
        with pytest.raises(InvalidTransitionError):
            order.transition_to(OrderStatus.ACCEPTED)

    def test_rejected_is_terminal(self):
        order = buy()
        order.transition_to(OrderStatus.REJECTED)
        with pytest.raises(InvalidTransitionError):
            order.transition_to(OrderStatus.FILLED)

    def test_partially_filled_can_expire(self):
        order = buy(100)
        order.transition_to(OrderStatus.ACCEPTED)
        order.apply_fill(10, 50.0, 0.0, bar_index=0)
        order.transition_to(OrderStatus.EXPIRED)
        assert order.is_terminal


class TestFills:
    def test_avg_fill_price_is_quantity_weighted(self):
        order = buy(150)
        order.transition_to(OrderStatus.ACCEPTED)
        order.apply_fill(100, 10.0, 0.0, bar_index=0)
        order.apply_fill(50, 13.0, 0.0, bar_index=1)
        assert order.avg_fill_price == pytest.approx((100 * 10 + 50 * 13) / 150)

    def test_commission_accumulates(self):
        order = buy(100)
        order.transition_to(OrderStatus.ACCEPTED)
        order.apply_fill(50, 10.0, 1.25, bar_index=0)
        order.apply_fill(50, 10.0, 1.75, bar_index=1)
        assert order.commission == pytest.approx(3.0)

    def test_overfill_raises(self):
        order = buy(100)
        order.transition_to(OrderStatus.ACCEPTED)
        with pytest.raises(OrderValidationError):
            order.apply_fill(101, 10.0, 0.0, bar_index=0)

    def test_fill_records_are_kept(self):
        order = buy(100)
        order.transition_to(OrderStatus.ACCEPTED)
        fill = order.apply_fill(100, 10.0, 0.5, bar_index=3)
        assert order.fills == [fill]
        assert fill.notional == pytest.approx(1000.0)
        assert fill.bar_index == 3
