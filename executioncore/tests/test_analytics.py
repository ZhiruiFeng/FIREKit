"""Execution analytics: fill rate, slippage, implementation shortfall."""

import pytest

from executioncore.algos import TWAP, AlgoExecutor, run_session
from executioncore.analytics import (
    algo_comparison,
    fill_rate,
    implementation_shortfall,
    order_shortfall,
    slippage_bps,
)
from executioncore.broker import PaperBroker
from executioncore.oms import OrderManager
from executioncore.orders import Order, OrderSide, OrderStatus
from tests.conftest import SYMBOL, make_feed


def order_with(status, qty=100.0, filled=0.0, avg=0.0, arrival=None, side=OrderSide.BUY):
    order = Order(SYMBOL, side, qty)
    order.status = status
    order.filled_quantity = filled
    order.avg_fill_price = avg
    order.arrival_price = arrival
    return order


class TestFillRate:
    def test_exact_ratio(self):
        orders = [
            order_with(OrderStatus.FILLED, qty=100, filled=100),
            order_with(OrderStatus.PARTIALLY_FILLED, qty=100, filled=40),
            order_with(OrderStatus.EXPIRED, qty=100, filled=0),
        ]
        assert fill_rate(orders) == pytest.approx(140 / 300)

    def test_rejected_orders_excluded(self):
        orders = [
            order_with(OrderStatus.FILLED, qty=100, filled=100),
            order_with(OrderStatus.REJECTED, qty=1_000_000),
        ]
        assert fill_rate(orders) == pytest.approx(1.0)

    def test_empty_is_zero(self):
        assert fill_rate([]) == 0.0


class TestSlippageBps:
    def test_buy_adverse_is_positive(self):
        order = order_with(OrderStatus.FILLED, filled=100, avg=100.10, arrival=100.0)
        assert slippage_bps(order) == pytest.approx(10.0)

    def test_sell_adverse_is_positive(self):
        order = order_with(
            OrderStatus.FILLED, filled=100, avg=99.90, arrival=100.0, side=OrderSide.SELL
        )
        assert slippage_bps(order) == pytest.approx(10.0)

    def test_unfilled_returns_none(self):
        order = order_with(OrderStatus.EXPIRED, filled=0, arrival=100.0)
        assert slippage_bps(order) is None


class TestImplementationShortfall:
    def test_perold_decomposition_buy(self):
        report = implementation_shortfall(
            side=OrderSide.BUY,
            quantity=1_000,
            filled_quantity=600,
            avg_fill_price=100.10,
            decision_price=100.00,
            arrival_price=100.05,
            final_price=100.20,
            commission=5.0,
        )
        assert report.delay_cost == pytest.approx(0.05 * 600)  # 30
        assert report.trading_cost == pytest.approx(0.05 * 600)  # 30
        assert report.opportunity_cost == pytest.approx(0.20 * 400)  # 80
        assert report.total_cost == pytest.approx(30 + 30 + 80 + 5)
        assert report.total_bps == pytest.approx(145 / (100.0 * 1_000) * 1e4)

    def test_components_sum_to_total(self):
        report = implementation_shortfall(
            OrderSide.SELL, 500, 500, 99.9, 100.0, 99.95, 99.5, commission=2.5
        )
        assert report.total_cost == pytest.approx(
            report.delay_cost + report.trading_cost + report.opportunity_cost + report.commission
        )

    def test_sell_signs(self):
        # Sell: price falling before execution is a cost.
        report = implementation_shortfall(
            OrderSide.SELL,
            quantity=100,
            filled_quantity=100,
            avg_fill_price=99.5,
            decision_price=100.0,
            arrival_price=99.8,
            final_price=99.0,
        )
        assert report.delay_cost == pytest.approx(20.0)  # -(99.8-100)*100
        assert report.trading_cost == pytest.approx(30.0)  # -(99.5-99.8)*100
        assert report.opportunity_cost == 0.0

    def test_fully_unfilled_is_pure_opportunity(self):
        report = implementation_shortfall(
            OrderSide.BUY, 100, 0, None, 100.0, 100.0, 101.0
        )
        assert report.delay_cost == 0.0
        assert report.trading_cost == 0.0
        assert report.opportunity_cost == pytest.approx(100.0)
        assert report.total_bps == pytest.approx(100.0)

    def test_order_shortfall_uses_order_fields(self):
        order = Order(SYMBOL, OrderSide.BUY, 100)
        order.decision_price = 100.0
        order.arrival_price = 100.0
        order.status = OrderStatus.FILLED
        order.filled_quantity = 100
        order.avg_fill_price = 100.2
        order.commission = 1.0
        report = order_shortfall(order, final_price=100.5)
        assert report.trading_cost == pytest.approx(20.0)
        assert report.total_cost == pytest.approx(21.0)


class TestAlgoComparison:
    def test_comparison_rows(self):
        feed = make_feed([(100.0, 100.5, 99.5, 100.0, 50_000.0)] * 8)
        broker = PaperBroker(feed)
        oms = OrderManager(broker)
        parent = Order(SYMBOL, OrderSide.BUY, 1_000)
        executor = AlgoExecutor(oms, parent, TWAP(), start_index=1, n_slices=4)
        run_session(broker, executors=[executor], oms=oms)
        rows = algo_comparison([executor], feed)
        assert len(rows) == 1
        row = rows[0]
        assert row["algo"] == "TWAP"
        assert row["filled_quantity"] == pytest.approx(1_000)
        assert row["avg_fill_price"] == pytest.approx(100.0)
        assert row["benchmark_vwap"] == pytest.approx(100.0)
        assert row["slippage_vs_arrival_bps"] == pytest.approx(0.0)
        assert row["shortfall_bps"] == pytest.approx(0.0)
