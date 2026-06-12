"""PaperBroker matching semantics: order types, TIF, latency, partial fills."""

import pytest

from executioncore.broker import PaperBroker
from executioncore.costs import FixedBpsSlippage
from executioncore.orders import Order, OrderSide, OrderStatus, OrderType, TimeInForce
from tests.conftest import SYMBOL


def buy(qty: float = 100, **kwargs) -> Order:
    return Order(SYMBOL, OrderSide.BUY, qty, **kwargs)


def sell(qty: float = 100, **kwargs) -> Order:
    return Order(SYMBOL, OrderSide.SELL, qty, **kwargs)


class TestMarketOrders:
    def test_fills_at_next_bar_open(self, flat_feed):
        broker = PaperBroker(flat_feed)
        order = buy(500)
        broker.submit_order(order)
        broker.advance()
        assert order.status == OrderStatus.FILLED
        assert order.avg_fill_price == pytest.approx(100.0)
        assert order.fills[0].bar_index == 0

    def test_latency_delays_eligibility(self, flat_feed):
        broker = PaperBroker(flat_feed, latency_bars=2)
        order = buy(500)
        broker.submit_order(order)
        broker.advance()
        broker.advance()
        assert order.status == OrderStatus.ACCEPTED  # bars 0 and 1: not yet eligible
        broker.advance()
        assert order.status == OrderStatus.FILLED
        assert order.fills[0].bar_index == 2

    def test_arrival_price_is_mark_at_submission(self, trend_feed):
        broker = PaperBroker(trend_feed)
        first = buy(10)
        broker.submit_order(first)
        assert first.arrival_price == pytest.approx(100.0)  # bar 0 open
        broker.advance()
        second = buy(10)
        broker.submit_order(second)
        assert second.arrival_price == pytest.approx(100.2)  # bar 0 close

    def test_unknown_symbol_rejected(self, flat_feed):
        broker = PaperBroker(flat_feed)
        order = Order("NOPE", OrderSide.BUY, 10)
        broker.submit_order(order)
        assert order.status == OrderStatus.REJECTED


class TestLimitOrders:
    def test_buy_fills_only_when_price_crosses(self, trend_feed):
        broker = PaperBroker(trend_feed)
        order = buy(100, order_type=OrderType.LIMIT, limit_price=99.2)
        broker.submit_order(order)
        broker.advance()  # bar 0: low 99.8 > 99.2
        assert order.status == OrderStatus.ACCEPTED
        broker.advance()  # bar 1: low 99.0 <= 99.2
        assert order.status == OrderStatus.FILLED
        assert order.avg_fill_price == pytest.approx(99.2)  # min(open 100.2, limit)

    def test_buy_fills_at_open_when_open_below_limit(self, trend_feed):
        broker = PaperBroker(trend_feed)
        broker.advance()
        broker.advance()  # current bar = 1
        order = buy(100, order_type=OrderType.LIMIT, limit_price=99.6)
        broker.submit_order(order)
        broker.advance()  # bar 2 opens at 99.2 < limit
        assert order.status == OrderStatus.FILLED
        assert order.avg_fill_price == pytest.approx(99.2)

    def test_sell_fills_when_high_crosses(self, trend_feed):
        broker = PaperBroker(trend_feed)
        order = sell(100, order_type=OrderType.LIMIT, limit_price=101.5)
        broker.submit_order(order)
        for _ in range(4):
            broker.advance()  # bars 0-3: high < 101.5 until bar 4
        assert order.status == OrderStatus.ACCEPTED
        broker.advance()  # bar 4: high 102.0
        assert order.status == OrderStatus.FILLED
        assert order.avg_fill_price == pytest.approx(101.5)  # max(open 100.8, limit)

    def test_unmarketable_day_limit_expires(self, flat_feed):
        broker = PaperBroker(flat_feed)
        order = buy(100, order_type=OrderType.LIMIT, limit_price=99.4)
        broker.submit_order(order)
        broker.run_to_close()
        assert order.status == OrderStatus.EXPIRED
        assert order.filled_quantity == 0


class TestStopOrders:
    def test_stop_sell_triggers_on_low(self, trend_feed):
        broker = PaperBroker(trend_feed)
        order = sell(100, order_type=OrderType.STOP, stop_price=99.1)
        broker.submit_order(order)
        broker.advance()  # bar 0: low 99.8 > 99.1, no trigger
        assert not order.triggered
        broker.advance()  # bar 1: low 99.0 <= 99.1
        assert order.status == OrderStatus.FILLED
        assert order.avg_fill_price == pytest.approx(99.1)  # min(open 100.2, stop)

    def test_stop_buy_triggers_on_high(self, trend_feed):
        broker = PaperBroker(trend_feed)
        order = buy(100, order_type=OrderType.STOP, stop_price=101.2)
        broker.submit_order(order)
        for _ in range(4):
            broker.advance()
        assert order.status == OrderStatus.ACCEPTED  # bar 3 high 101.0 < stop
        broker.advance()  # bar 4: high 102.0 >= 101.2
        assert order.status == OrderStatus.FILLED
        assert order.avg_fill_price == pytest.approx(101.2)  # max(open 100.8, stop)

    def test_stop_limit_fills_within_limit(self, trend_feed):
        broker = PaperBroker(trend_feed)
        order = buy(
            100, order_type=OrderType.STOP_LIMIT, stop_price=100.9, limit_price=101.2
        )
        broker.submit_order(order)
        for _ in range(4):
            broker.advance()  # triggers on bar 3 (high 101.0 >= 100.9)
        assert order.triggered
        assert order.status == OrderStatus.FILLED
        assert order.avg_fill_price == pytest.approx(100.9)

    def test_stop_limit_rests_when_limit_below_stop(self, trend_feed):
        # Buy stop 100.9 with limit 99.0: triggers but the limit is far below
        # the market, so it never fills and expires.
        broker = PaperBroker(trend_feed)
        order = buy(
            100, order_type=OrderType.STOP_LIMIT, stop_price=100.9, limit_price=98.0
        )
        broker.submit_order(order)
        broker.run_to_close()
        assert order.triggered
        assert order.status == OrderStatus.EXPIRED


class TestPartialFillsAndTif:
    def test_participation_cap_partial_fills_across_bars(self, flat_feed):
        broker = PaperBroker(flat_feed, participation_cap=0.10)  # 1000/bar
        order = buy(2_500, time_in_force=TimeInForce.GTC)
        broker.submit_order(order)
        broker.advance()
        assert order.status == OrderStatus.PARTIALLY_FILLED
        assert order.filled_quantity == pytest.approx(1_000)
        broker.advance()
        assert order.filled_quantity == pytest.approx(2_000)
        broker.advance()
        assert order.status == OrderStatus.FILLED
        assert order.filled_quantity == pytest.approx(2_500)
        assert [f.quantity for f in order.fills] == [1_000, 1_000, 500]

    def test_ioc_fills_then_cancels_remainder(self, flat_feed):
        broker = PaperBroker(flat_feed, participation_cap=0.10)
        order = buy(2_500, time_in_force=TimeInForce.IOC)
        broker.submit_order(order)
        broker.advance()
        assert order.status == OrderStatus.CANCELLED
        assert order.filled_quantity == pytest.approx(1_000)

    def test_ioc_unmarketable_is_cancelled(self, flat_feed):
        broker = PaperBroker(flat_feed)
        order = buy(
            100, order_type=OrderType.LIMIT, limit_price=99.0, time_in_force=TimeInForce.IOC
        )
        broker.submit_order(order)
        broker.advance()
        assert order.status == OrderStatus.CANCELLED
        assert order.filled_quantity == 0

    def test_fok_fills_fully_when_liquidity_allows(self, flat_feed):
        broker = PaperBroker(flat_feed, participation_cap=0.10)
        order = buy(800, time_in_force=TimeInForce.FOK)
        broker.submit_order(order)
        broker.advance()
        assert order.status == OrderStatus.FILLED

    def test_fok_cancels_when_size_exceeds_liquidity(self, flat_feed):
        broker = PaperBroker(flat_feed, participation_cap=0.10)
        order = buy(2_500, time_in_force=TimeInForce.FOK)
        broker.submit_order(order)
        broker.advance()
        assert order.status == OrderStatus.CANCELLED
        assert order.filled_quantity == 0

    def test_day_expires_gtc_survives_session_end(self, flat_feed):
        broker = PaperBroker(flat_feed)
        day = buy(100, order_type=OrderType.LIMIT, limit_price=99.0)
        gtc = buy(100, order_type=OrderType.LIMIT, limit_price=99.0, time_in_force=TimeInForce.GTC)
        broker.submit_order(day)
        broker.submit_order(gtc)
        broker.run_to_close()
        assert day.status == OrderStatus.EXPIRED
        assert gtc.status == OrderStatus.ACCEPTED
        assert broker.open_orders() == [gtc]


class TestCancelReplace:
    def test_cancel_working_order(self, flat_feed):
        broker = PaperBroker(flat_feed)
        order = buy(100, order_type=OrderType.LIMIT, limit_price=99.0)
        broker.submit_order(order)
        assert broker.cancel_order(order.order_id) is True
        assert order.status == OrderStatus.CANCELLED
        assert broker.cancel_order(order.order_id) is False

    def test_cannot_cancel_filled_order(self, flat_feed):
        broker = PaperBroker(flat_feed)
        order = buy(100)
        broker.submit_order(order)
        broker.advance()
        assert broker.cancel_order(order.order_id) is False
        assert order.status == OrderStatus.FILLED

    def test_replace_limit_price_enables_fill(self, flat_feed):
        broker = PaperBroker(flat_feed)
        order = buy(100, order_type=OrderType.LIMIT, limit_price=99.0)
        broker.submit_order(order)
        broker.advance()
        assert order.status == OrderStatus.ACCEPTED
        assert broker.replace_order(order.order_id, limit_price=99.6) is True
        broker.advance()  # low 99.5 <= 99.6
        assert order.status == OrderStatus.FILLED
        assert order.avg_fill_price == pytest.approx(99.6)

    def test_replace_quantity_below_filled_fails(self, flat_feed):
        broker = PaperBroker(flat_feed, participation_cap=0.10)
        order = buy(2_500, time_in_force=TimeInForce.GTC)
        broker.submit_order(order)
        broker.advance()  # 1000 filled
        assert broker.replace_order(order.order_id, quantity=800) is False
        assert broker.replace_order(order.order_id, quantity=1_500) is True
        broker.advance()
        assert order.status == OrderStatus.FILLED
        assert order.filled_quantity == pytest.approx(1_500)


class TestSlippageIntegration:
    def test_market_buy_pays_slippage(self, flat_feed):
        broker = PaperBroker(flat_feed, slippage=FixedBpsSlippage(10.0))
        order = buy(100)
        broker.submit_order(order)
        broker.advance()
        assert order.avg_fill_price == pytest.approx(100.0 * 1.001)

    def test_limit_fill_never_exceeds_limit(self, flat_feed):
        broker = PaperBroker(flat_feed, slippage=FixedBpsSlippage(10.0))
        order = buy(100, order_type=OrderType.LIMIT, limit_price=100.05)
        broker.submit_order(order)
        broker.advance()
        # raw slipped price would be 100.10; clipped to the limit
        assert order.avg_fill_price == pytest.approx(100.05)
