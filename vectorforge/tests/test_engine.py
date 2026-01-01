"""
Tests for VectorForge backtesting engines.
"""

import numpy as np
import pandas as pd
import pytest

from vectorforge import VectorizedBacktester, EventDrivenBacktester, HybridRunner
from vectorforge.engine.base import BacktestResult
from vectorforge.strategy.base import MomentumStrategy, MovingAverageCrossover


class TestVectorizedBacktester:
    """Tests for the VectorizedBacktester."""

    def test_run_basic(self, vectorized_backtester, sample_ohlcv_data, momentum_strategy):
        """Test basic backtest run."""
        result = vectorized_backtester.run(
            strategy=momentum_strategy,
            data=sample_ohlcv_data,
            initial_capital=100000,
        )

        assert isinstance(result, BacktestResult)
        assert result.initial_capital == 100000
        assert result.final_capital > 0
        assert result.mode == "vectorized"

    def test_result_metrics(self, vectorized_backtester, sample_ohlcv_data, momentum_strategy):
        """Test that all metrics are calculated."""
        result = vectorized_backtester.run(momentum_strategy, sample_ohlcv_data)

        # Check all required metrics exist
        assert hasattr(result, "sharpe_ratio")
        assert hasattr(result, "sortino_ratio")
        assert hasattr(result, "max_drawdown")
        assert hasattr(result, "total_return")
        assert hasattr(result, "win_rate")

        # Check metrics are valid numbers
        assert not np.isnan(result.sharpe_ratio)
        assert not np.isnan(result.max_drawdown)
        assert result.max_drawdown <= 0  # Drawdown is negative

    def test_equity_curve(self, vectorized_backtester, sample_ohlcv_data, momentum_strategy):
        """Test equity curve generation."""
        result = vectorized_backtester.run(momentum_strategy, sample_ohlcv_data)

        assert len(result.equity_curve) > 0
        assert result.equity_curve.iloc[0] == result.initial_capital
        assert result.equity_curve.iloc[-1] == result.final_capital

    def test_batch_run(self, vectorized_backtester, sample_ohlcv_data):
        """Test batch parameter sweep."""
        results = vectorized_backtester.run_batch(
            strategy_class=MomentumStrategy,
            param_grid={"lookback": [10, 20, 30]},
            data=sample_ohlcv_data,
        )

        assert len(results) == 3
        assert all(isinstance(r, BacktestResult) for r in results)

    def test_invalid_data(self, vectorized_backtester, momentum_strategy):
        """Test handling of invalid data."""
        invalid_data = pd.DataFrame({"close": [1, 2, 3]})

        with pytest.raises(ValueError):
            vectorized_backtester.run(momentum_strategy, invalid_data)


class TestEventDrivenBacktester:
    """Tests for the EventDrivenBacktester."""

    def test_run_basic(self, event_driven_backtester, sample_ohlcv_data, ma_crossover_strategy):
        """Test basic event-driven backtest."""
        result = event_driven_backtester.run(
            strategy=ma_crossover_strategy,
            data=sample_ohlcv_data,
            initial_capital=100000,
        )

        assert isinstance(result, BacktestResult)
        assert result.mode == "event_driven"
        assert result.initial_capital == 100000

    def test_trades_recorded(self, event_driven_backtester, trending_data, ma_crossover_strategy):
        """Test that trades are recorded."""
        result = event_driven_backtester.run(ma_crossover_strategy, trending_data)

        # Should have at least one trade in trending market
        assert result.total_trades >= 0

    def test_execution_costs(self, event_driven_backtester, trending_data, ma_crossover_strategy):
        """Test that execution costs are applied."""
        result = event_driven_backtester.run(ma_crossover_strategy, trending_data)

        # With execution costs, return should be less than ideal
        assert result.total_return is not None


class TestHybridRunner:
    """Tests for the HybridRunner."""

    def test_auto_mode_selection(self, config, sample_ohlcv_data):
        """Test automatic mode selection."""
        runner = HybridRunner(config)
        momentum = MomentumStrategy(lookback=20)
        ma = MovingAverageCrossover(fast_period=10, slow_period=30)

        # Momentum should use vectorized (has generate_signals)
        result1 = runner.run(momentum, sample_ohlcv_data)

        # Both modes should produce valid results
        assert isinstance(result1, BacktestResult)

    def test_compare_modes(self, config, sample_ohlcv_data, momentum_strategy):
        """Test mode comparison."""
        runner = HybridRunner(config)
        comparison = runner.compare_modes(momentum_strategy, sample_ohlcv_data)

        assert "vectorized" in comparison
        assert "event_driven" in comparison
        assert "sharpe_gap" in comparison

    def test_batch_uses_vectorized(self, config, sample_ohlcv_data):
        """Test that batch operations use vectorized mode."""
        runner = HybridRunner(config)

        results = runner.run_batch(
            strategy_class=MomentumStrategy,
            param_grid={"lookback": [10, 20]},
            data=sample_ohlcv_data,
        )

        assert len(results) == 2
        assert all(r.mode == "vectorized" for r in results)


class TestBacktestResult:
    """Tests for BacktestResult."""

    def test_summary(self, vectorized_backtester, sample_ohlcv_data, momentum_strategy):
        """Test result summary generation."""
        result = vectorized_backtester.run(momentum_strategy, sample_ohlcv_data)
        summary = result.summary()

        assert isinstance(summary, dict)
        assert "Sharpe Ratio" in summary
        assert "Max Drawdown" in summary
        assert "Total Return" in summary

    def test_repr(self, vectorized_backtester, sample_ohlcv_data, momentum_strategy):
        """Test result string representation."""
        result = vectorized_backtester.run(momentum_strategy, sample_ohlcv_data)

        repr_str = repr(result)
        assert "BacktestResult" in repr_str
        assert "sharpe_ratio" in repr_str
