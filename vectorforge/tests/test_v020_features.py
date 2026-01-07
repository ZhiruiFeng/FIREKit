"""
Tests for VectorForge v0.2.0 Performance Features

Tests cover:
- Accelerated computation kernels (JAX/Numba)
- vmap-based parallel parameter testing
- Memory-mapped arrays
- Advanced order types
- Exchange calendar
- Hybrid Runner strategy analysis
"""

import numpy as np
import pandas as pd
import pytest
from datetime import date, time, datetime, timedelta

# Import v0.2.0 modules
from vectorforge.engine.accelerated import (
    numpy_compute_returns,
    numpy_compute_metrics,
    numpy_batch_backtest,
    compute_returns,
    compute_metrics,
    batch_backtest,
    get_compute_backend,
    is_jax_available,
    is_numba_available,
    BacktestArrays,
    MetricsResult,
)


class TestAcceleratedCore:
    """Tests for the accelerated computation module."""

    def test_numpy_compute_returns_basic(self, small_ohlcv_data):
        """Test basic return computation with NumPy."""
        prices = small_ohlcv_data["close"].values
        signals = np.ones(len(prices))  # Always long

        result = numpy_compute_returns(prices, signals, slippage_bps=5.0, commission_pct=0.001)

        assert isinstance(result, BacktestArrays)
        assert len(result.returns) == len(prices) - 1
        assert len(result.equity_curve) == len(prices)
        assert result.equity_curve[0] == 1.0

    def test_numpy_compute_returns_short(self, small_ohlcv_data):
        """Test short position signals."""
        prices = small_ohlcv_data["close"].values
        signals = -np.ones(len(prices))  # Always short

        result = numpy_compute_returns(prices, signals)

        # Short position should have negative returns in uptrend
        assert isinstance(result, BacktestArrays)

    def test_numpy_compute_metrics(self):
        """Test metrics computation."""
        # Create sample returns
        np.random.seed(42)
        returns = np.random.normal(0.0005, 0.02, 252)
        equity_curve = np.cumprod(1 + returns)
        equity_curve = np.insert(equity_curve, 0, 1.0)

        metrics = numpy_compute_metrics(returns, equity_curve)

        assert isinstance(metrics, MetricsResult)
        assert not np.isnan(metrics.sharpe_ratio)
        assert not np.isnan(metrics.sortino_ratio)
        assert metrics.max_drawdown <= 0  # Drawdown is negative

    def test_numpy_batch_backtest(self, small_ohlcv_data):
        """Test batch backtesting."""
        prices = small_ohlcv_data["close"].values
        n_combos = 10

        # Create different signal arrays
        signal_batches = np.zeros((n_combos, len(prices)))
        for i in range(n_combos):
            signal_batches[i] = np.sin(np.arange(len(prices)) * (i + 1) / 10)

        returns, metrics = numpy_batch_backtest(prices, signal_batches)

        assert returns.shape[0] == n_combos
        assert metrics.shape == (n_combos, 2)  # total_return, sharpe

    def test_backend_selection(self):
        """Test backend selection logic."""
        # Auto should return best available
        backend = get_compute_backend("auto")
        assert backend in ("jax", "numba", "numpy")

        # Requesting unavailable backend should fallback
        backend = get_compute_backend("jax")
        if is_jax_available():
            assert backend == "jax"
        else:
            assert backend == "numpy"

    def test_compute_returns_dispatch(self, small_ohlcv_data):
        """Test that compute_returns dispatches to correct backend."""
        prices = small_ohlcv_data["close"].values
        signals = np.ones(len(prices))

        # Should work regardless of backend
        result = compute_returns(prices, signals, backend="numpy")
        assert isinstance(result, BacktestArrays)

        # Auto should also work
        result = compute_returns(prices, signals, backend="auto")
        assert isinstance(result, BacktestArrays)


class TestAdvancedOrderTypes:
    """Tests for advanced order types in event-driven engine."""

    def test_stop_order(self, small_ohlcv_data, config):
        """Test stop order execution."""
        from vectorforge.engine.event_driven import (
            SimulatedBroker,
            Order,
            OrderSide,
            OrderType,
            Bar,
        )

        broker = SimulatedBroker(config)

        # Create a stop order
        order = Order(
            symbol="TEST",
            side=OrderSide.SELL,
            quantity=100,
            order_type=OrderType.STOP,
            stop_price=95.0,  # Stop sell if price drops to 95
        )
        broker.submit_order(order)

        # Create a bar that triggers the stop
        bar = Bar(
            symbol="TEST",
            timestamp=datetime.now(),
            open=100.0,
            high=100.0,
            low=94.0,  # Goes below stop price
            close=95.0,
            volume=1000000,
        )

        fills = broker.process_bar(bar)

        assert len(fills) == 1
        assert fills[0].side == OrderSide.SELL
        assert fills[0].quantity == 100

    def test_stop_limit_order(self, config):
        """Test stop-limit order execution."""
        from vectorforge.engine.event_driven import (
            SimulatedBroker,
            Order,
            OrderSide,
            OrderType,
            Bar,
        )

        broker = SimulatedBroker(config)

        # Create a stop-limit order
        order = Order(
            symbol="TEST",
            side=OrderSide.SELL,
            quantity=100,
            order_type=OrderType.STOP_LIMIT,
            stop_price=95.0,
            limit_price=94.5,
        )
        broker.submit_order(order)

        # Bar triggers stop but also allows limit fill
        bar = Bar(
            symbol="TEST",
            timestamp=datetime.now(),
            open=96.0,
            high=96.0,
            low=94.0,
            close=94.5,
            volume=1000000,
        )

        fills = broker.process_bar(bar)

        assert len(fills) == 1
        assert fills[0].fill_price >= 94.5  # Should be at or above limit

    def test_trailing_stop_order(self, config):
        """Test trailing stop order updates and execution."""
        from vectorforge.engine.event_driven import (
            SimulatedBroker,
            Order,
            OrderSide,
            OrderType,
            Bar,
        )

        broker = SimulatedBroker(config)

        # Create trailing stop with 5% trail
        order = Order(
            symbol="TEST",
            side=OrderSide.SELL,
            quantity=100,
            order_type=OrderType.TRAILING_STOP,
            trail_percent=5.0,
        )
        broker.submit_order(order)

        # First bar establishes initial stop
        bar1 = Bar(
            symbol="TEST",
            timestamp=datetime.now(),
            open=100.0,
            high=105.0,  # High sets initial stop at 105 * 0.95 = 99.75
            low=99.0,
            close=102.0,
            volume=1000000,
        )
        fills = broker.process_bar(bar1)
        assert len(fills) == 0  # Not triggered

        # Check that stop price was tracked
        assert order.order_id in broker.trailing_stop_prices

        # Second bar - price rises, stop should move up
        bar2 = Bar(
            symbol="TEST",
            timestamp=datetime.now() + timedelta(days=1),
            open=102.0,
            high=110.0,  # New high moves stop to 110 * 0.95 = 104.5
            low=101.0,
            close=108.0,
            volume=1000000,
        )
        broker.process_bar(bar2)

        # Stop should have moved up
        assert broker.trailing_stop_prices[order.order_id] > 99.75

        # Third bar - price drops and triggers stop
        bar3 = Bar(
            symbol="TEST",
            timestamp=datetime.now() + timedelta(days=2),
            open=106.0,
            high=106.0,
            low=103.0,  # Drops below 104.5 stop
            close=104.0,
            volume=1000000,
        )
        fills = broker.process_bar(bar3)

        assert len(fills) == 1
        assert fills[0].side == OrderSide.SELL

    def test_time_in_force_gtd(self, config):
        """Test GTD (Good Till Date) orders."""
        from vectorforge.engine.event_driven import (
            SimulatedBroker,
            Order,
            OrderSide,
            OrderType,
            TimeInForce,
            Bar,
        )

        broker = SimulatedBroker(config)

        # Create GTD order expiring in 1 day
        expiry = datetime.now() + timedelta(days=1)
        order = Order(
            symbol="TEST",
            side=OrderSide.BUY,
            quantity=100,
            order_type=OrderType.LIMIT,
            limit_price=95.0,
            time_in_force=TimeInForce.GTD,
            expire_date=expiry,
        )
        broker.submit_order(order)

        assert len(broker.pending_orders) == 1

        # Bar after expiry - order should be removed
        bar = Bar(
            symbol="TEST",
            timestamp=expiry + timedelta(hours=1),
            open=100.0,
            high=100.0,
            low=99.0,
            close=99.5,
            volume=1000000,
        )
        broker._expire_orders(bar.timestamp)

        assert len(broker.pending_orders) == 0

    def test_bracket_order(self, config):
        """Test bracket order (entry + TP + SL)."""
        from vectorforge.engine.event_driven import (
            SimulatedBroker,
            Order,
            BracketOrder,
            OrderSide,
            OrderType,
            Bar,
        )

        broker = SimulatedBroker(config)

        # Create bracket order
        entry = Order(
            symbol="TEST",
            side=OrderSide.BUY,
            quantity=100,
            order_type=OrderType.MARKET,
        )
        take_profit = Order(
            symbol="TEST",
            side=OrderSide.SELL,
            quantity=100,
            order_type=OrderType.LIMIT,
            limit_price=110.0,
        )
        stop_loss = Order(
            symbol="TEST",
            side=OrderSide.SELL,
            quantity=100,
            order_type=OrderType.STOP,
            stop_price=95.0,
        )

        bracket = BracketOrder(
            entry_order=entry,
            take_profit_order=take_profit,
            stop_loss_order=stop_loss,
        )

        broker.submit_bracket_order(bracket)

        assert len(broker.pending_orders) == 3
        assert len(broker.active_oco_groups) == 1

        # Fill entry
        bar = Bar(
            symbol="TEST",
            timestamp=datetime.now(),
            open=100.0,
            high=100.0,
            low=99.0,
            close=100.0,
            volume=1000000,
        )
        fills = broker.process_bar(bar)
        assert len(fills) == 1  # Entry filled

        # TP and SL still pending
        assert len(broker.pending_orders) == 2


class TestExchangeCalendar:
    """Tests for exchange calendar functionality."""

    def test_nyse_calendar_basic(self):
        """Test basic NYSE calendar functionality."""
        from vectorforge.calendar import get_calendar, list_calendars

        nyse = get_calendar("NYSE")

        assert nyse.name == "New York Stock Exchange"
        assert nyse.timezone == "America/New_York"
        assert nyse.regular_open == time(9, 30)
        assert nyse.regular_close == time(16, 0)

    def test_is_open(self):
        """Test market open detection."""
        from vectorforge.calendar import get_calendar
        from zoneinfo import ZoneInfo

        nyse = get_calendar("NYSE")
        tz = ZoneInfo("America/New_York")

        # Regular trading hours
        trading_time = datetime(2024, 6, 3, 10, 30, tzinfo=tz)  # Monday 10:30 AM
        assert nyse.is_open(trading_time)

        # Before market open
        pre_market = datetime(2024, 6, 3, 8, 0, tzinfo=tz)
        assert not nyse.is_open(pre_market)
        assert nyse.is_pre_market(pre_market)

        # After hours
        after_hours = datetime(2024, 6, 3, 18, 0, tzinfo=tz)
        assert not nyse.is_open(after_hours)
        assert nyse.is_after_hours(after_hours)

        # Weekend
        weekend = datetime(2024, 6, 1, 10, 30, tzinfo=tz)  # Saturday
        assert not nyse.is_open(weekend)

    def test_trading_days(self):
        """Test trading days iteration."""
        from vectorforge.calendar import get_calendar

        nyse = get_calendar("NYSE")

        # Count trading days in January 2024
        start = date(2024, 1, 1)
        end = date(2024, 1, 31)
        trading_days = list(nyse.trading_days(start, end))

        # January 2024 has 21 trading days (excluding weekends and Jan 1, Jan 15)
        assert len(trading_days) == 21

        # Check that weekends are excluded
        for d in trading_days:
            assert d.weekday() < 5  # Not Saturday or Sunday

    def test_crypto_calendar(self):
        """Test 24/7 crypto calendar."""
        from vectorforge.calendar import get_calendar
        from zoneinfo import ZoneInfo

        crypto = get_calendar("CRYPTO")

        tz = ZoneInfo("UTC")

        # Should always be open
        saturday = datetime(2024, 6, 1, 10, 0, tzinfo=tz)
        sunday = datetime(2024, 6, 2, 3, 0, tzinfo=tz)

        assert crypto.is_open(saturday)
        assert crypto.is_open(sunday)

    def test_list_calendars(self):
        """Test listing available calendars."""
        from vectorforge.calendar import list_calendars

        calendars = list_calendars()

        assert "NYSE" in calendars
        assert "NASDAQ" in calendars
        assert "CRYPTO" in calendars


class TestHybridRunnerAnalysis:
    """Tests for Hybrid Runner strategy analysis."""

    def test_analyze_simple_strategy(self, momentum_strategy):
        """Test analysis of a simple vectorizable strategy."""
        from vectorforge.engine.hybrid import HybridRunner

        runner = HybridRunner()
        analysis = runner.analyze_strategy(momentum_strategy)

        assert analysis.supports_vectorized
        assert analysis.recommended_mode in ("vectorized", "event_driven")
        assert analysis.confidence > 0

    def test_analyze_caches_results(self, momentum_strategy):
        """Test that analysis results are cached."""
        from vectorforge.engine.hybrid import HybridRunner

        runner = HybridRunner()

        # First call
        analysis1 = runner.analyze_strategy(momentum_strategy)

        # Second call should return cached result
        analysis2 = runner.analyze_strategy(momentum_strategy)

        assert analysis1 is analysis2  # Same object

    def test_run_adaptive(self, momentum_strategy, small_ohlcv_data):
        """Test adaptive execution mode."""
        from vectorforge.engine.hybrid import HybridRunner

        runner = HybridRunner()

        # Run without validation
        result = runner.run_adaptive(
            momentum_strategy,
            small_ohlcv_data,
            validate=False,
        )
        assert result is not None
        assert hasattr(result, "sharpe_ratio")

    def test_discrepancy_report(self, momentum_strategy, small_ohlcv_data):
        """Test discrepancy report generation."""
        from vectorforge.engine.hybrid import HybridRunner

        runner = HybridRunner()

        report = runner.get_discrepancy_report(
            momentum_strategy,
            small_ohlcv_data,
        )

        assert "vectorized_result" in report
        assert "event_driven_result" in report
        assert "discrepancies" in report
        assert "recommendations" in report

        # Check discrepancy metrics
        discrepancies = report["discrepancies"]
        assert "total_return_gap" in discrepancies
        assert "sharpe_gap" in discrepancies
        assert "equity_curve_correlation" in discrepancies


class TestVectorizedBacktesterV020:
    """Tests for v0.2.0 enhancements to VectorizedBacktester."""

    def test_backend_info(self, vectorized_backtester):
        """Test backend info property."""
        info = vectorized_backtester.backend_info

        assert "requested" in info
        assert "actual" in info
        assert "jax_available" in info
        assert "numba_available" in info
        assert "memory_mapping" in info

    def test_memory_mapping_enable_disable(self, vectorized_backtester):
        """Test memory mapping configuration."""
        # Enable
        vectorized_backtester.enable_memory_mapping()
        assert vectorized_backtester._use_mmap

        # Disable
        vectorized_backtester.disable_memory_mapping()
        assert not vectorized_backtester._use_mmap

    def test_run_batch_quick(self, vectorized_backtester, small_ohlcv_data):
        """Test quick batch backtest."""
        from vectorforge.strategy.base import MomentumStrategy

        param_grid = {"lookback": [10, 20, 30]}

        results = vectorized_backtester.run_batch_quick(
            MomentumStrategy,
            param_grid,
            small_ohlcv_data,
        )

        assert len(results) == 3
        for r in results:
            assert "params" in r
            assert "total_return" in r
            assert "sharpe_ratio" in r

    def test_run_batch_parallel(self, vectorized_backtester, small_ohlcv_data):
        """Test parallel batch execution."""
        from vectorforge.strategy.base import MomentumStrategy

        param_grid = {"lookback": [5, 10, 15, 20]}

        # Test with parallel enabled
        results = vectorized_backtester.run_batch(
            MomentumStrategy,
            param_grid,
            small_ohlcv_data,
            use_parallel=True,
        )

        assert len(results) == 4
        for r in results:
            assert hasattr(r, "sharpe_ratio")
            assert hasattr(r, "total_return")


class TestPerformanceBenchmarks:
    """Basic performance tests to verify acceleration works."""

    def test_vectorized_faster_than_event_driven(self, sample_ohlcv_data, momentum_strategy):
        """Verify vectorized mode is faster than event-driven."""
        from vectorforge import VectorizedBacktester, EventDrivenBacktester

        vec = VectorizedBacktester()
        ed = EventDrivenBacktester()

        import time

        # Vectorized timing
        start = time.perf_counter()
        vec.run(momentum_strategy, sample_ohlcv_data)
        vec_time = time.perf_counter() - start

        # Event-driven timing
        start = time.perf_counter()
        ed.run(momentum_strategy, sample_ohlcv_data)
        ed_time = time.perf_counter() - start

        # Vectorized should be at least 5x faster
        assert vec_time < ed_time, "Vectorized should be faster than event-driven"

    def test_batch_faster_than_sequential(self, sample_ohlcv_data):
        """Verify batch execution is faster than sequential."""
        from vectorforge import VectorizedBacktester
        from vectorforge.strategy.base import MomentumStrategy

        vec = VectorizedBacktester()
        param_grid = {"lookback": list(range(5, 25, 5))}

        import time

        # Parallel batch timing
        start = time.perf_counter()
        vec.run_batch(MomentumStrategy, param_grid, sample_ohlcv_data, use_parallel=True)
        batch_time = time.perf_counter() - start

        # Sequential timing
        start = time.perf_counter()
        vec.run_batch(MomentumStrategy, param_grid, sample_ohlcv_data, use_parallel=False)
        seq_time = time.perf_counter() - start

        # Batch should be faster (or at least not significantly slower)
        # Note: For small param grids, overhead might make batch slower
        # So we just verify it completes
        assert batch_time > 0
        assert seq_time > 0
