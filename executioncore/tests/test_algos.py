"""TWAP/VWAP slicing and AlgoExecutor scheduling."""

import pytest

from executioncore.algos import TWAP, VWAP, AlgoExecutor, run_session
from executioncore.broker import PaperBroker
from executioncore.oms import OrderManager
from executioncore.orders import Order, OrderSide, OrderType
from tests.conftest import SYMBOL, make_feed


def feed_with_volumes(volumes, price: float = 100.0):
    return make_feed([(price, price + 0.5, price - 0.5, price, v) for v in volumes])


class TestTwapSchedule:
    def test_slices_sum_to_parent_quantity(self):
        schedule = TWAP().schedule(6_000, start_index=10, n_slices=20)
        assert len(schedule) == 20
        assert sum(q for _, q in schedule) == pytest.approx(6_000, abs=1e-9)
        assert all(q == pytest.approx(300.0) for _, q in schedule)

    def test_consecutive_bars(self):
        schedule = TWAP().schedule(100, start_index=5, n_slices=4)
        assert [b for b, _ in schedule] == [5, 6, 7, 8]

    def test_non_divisible_quantity_exact_sum(self):
        schedule = TWAP().schedule(1_000, start_index=0, n_slices=7)
        assert sum(q for _, q in schedule) == pytest.approx(1_000, abs=1e-9)

    def test_invalid_slice_count(self):
        with pytest.raises(ValueError):
            TWAP().schedule(100, 0, 0)


class TestVwapSchedule:
    def test_explicit_profile_weights(self):
        algo = VWAP(volume_profile=[1.0, 2.0, 1.0])
        schedule = algo.schedule(400, start_index=0, n_slices=3)
        assert [q for _, q in schedule] == [pytest.approx(100), pytest.approx(200), pytest.approx(100)]
        assert sum(q for _, q in schedule) == pytest.approx(400, abs=1e-9)

    def test_profile_length_must_match(self):
        with pytest.raises(ValueError):
            VWAP(volume_profile=[1.0, 2.0]).schedule(400, 0, 3)

    def test_profile_from_feed_volumes(self):
        feed = feed_with_volumes([10_000, 30_000, 10_000, 50_000])
        algo = VWAP()
        schedule = algo.schedule(1_000, start_index=0, n_slices=4, feed=feed, symbol=SYMBOL)
        weights = [q / 1_000 for _, q in schedule]
        assert weights == [
            pytest.approx(0.1),
            pytest.approx(0.3),
            pytest.approx(0.1),
            pytest.approx(0.5),
        ]

    def test_feed_required_without_profile(self):
        with pytest.raises(ValueError):
            VWAP().schedule(400, 0, 3)

    def test_negative_profile_rejected(self):
        with pytest.raises(ValueError):
            VWAP(volume_profile=[1.0, -1.0])


class TestAlgoExecutor:
    def test_twap_executes_full_parent(self):
        feed = feed_with_volumes([100_000] * 10)
        broker = PaperBroker(feed)
        oms = OrderManager(broker)
        parent = Order(SYMBOL, OrderSide.BUY, 1_200)
        executor = AlgoExecutor(oms, parent, TWAP(), start_index=2, n_slices=4)
        run_session(broker, executors=[executor], oms=oms)
        assert executor.is_complete
        assert executor.filled_quantity == pytest.approx(1_200)
        assert len(executor.children) == 4
        assert all(c.parent_id == parent.order_id for c in executor.children)
        assert all(c.algo == "TWAP" for c in executor.children)
        # Children scheduled on bars 2..5 fill at those bars (zero latency).
        fill_bars = [c.fills[0].bar_index for c in executor.children]
        assert fill_bars == [2, 3, 4, 5]

    def test_executor_avg_price_is_weighted(self):
        prices = [100.0, 100.0, 102.0, 102.0]
        feed = make_feed([(p, p + 0.5, p - 0.5, p, 100_000.0) for p in prices])
        broker = PaperBroker(feed)
        oms = OrderManager(broker)
        parent = Order(SYMBOL, OrderSide.BUY, 300)
        executor = AlgoExecutor(
            oms, parent, VWAP(volume_profile=[1.0, 2.0]), start_index=1, n_slices=2
        )
        run_session(broker, executors=[executor], oms=oms)
        # 100 shares at bar1 open (100.0) + 200 at bar2 open (102.0)
        assert executor.avg_fill_price == pytest.approx((100 * 100.0 + 200 * 102.0) / 300)

    def test_parent_must_be_market_style(self):
        feed = feed_with_volumes([100_000] * 4)
        broker = PaperBroker(feed)
        oms = OrderManager(broker)
        parent = Order(SYMBOL, OrderSide.BUY, 100, OrderType.LIMIT, limit_price=100.0)
        with pytest.raises(ValueError):
            AlgoExecutor(oms, parent, TWAP(), start_index=0, n_slices=2)

    def test_parent_decision_price_recorded(self):
        feed = feed_with_volumes([100_000] * 4)
        broker = PaperBroker(feed)
        oms = OrderManager(broker)
        parent = Order(SYMBOL, OrderSide.BUY, 100)
        executor = AlgoExecutor(oms, parent, TWAP(), start_index=1, n_slices=2)
        assert parent.decision_price == pytest.approx(100.0)
        assert executor.window == (1, 3)
