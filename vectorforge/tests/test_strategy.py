"""
Tests for VectorForge strategies.
"""

import numpy as np
import pytest

from vectorforge.strategy.base import BaseStrategy, MomentumStrategy, MovingAverageCrossover


class TestMomentumStrategy:
    """Tests for MomentumStrategy."""

    def test_signal_generation(self, sample_ohlcv_data):
        """Test signal generation."""
        strategy = MomentumStrategy(lookback=20)
        signals = strategy.generate_signals(close=sample_ohlcv_data["close"].values)

        assert len(signals) == len(sample_ohlcv_data)
        assert set(np.unique(signals)).issubset({-1, 0, 1})

    def test_signal_padding(self):
        """Test that signals are padded for warmup period."""
        close = np.linspace(100, 110, 50)
        strategy = MomentumStrategy(lookback=20)
        signals = strategy.generate_signals(close=close)

        # First `lookback` signals should be 0
        assert all(signals[:20] == 0)

    def test_trending_market(self, trending_data):
        """Test signals in trending market."""
        strategy = MomentumStrategy(lookback=10)
        signals = strategy.generate_signals(close=trending_data["close"].values)

        # Most signals should be positive in uptrend
        active_signals = signals[signals != 0]
        if len(active_signals) > 0:
            assert np.mean(active_signals) > 0

    def test_different_lookbacks(self, sample_ohlcv_data):
        """Test different lookback periods."""
        close = sample_ohlcv_data["close"].values

        for lookback in [5, 10, 20, 50]:
            strategy = MomentumStrategy(lookback=lookback)
            signals = strategy.generate_signals(close=close)
            assert len(signals) == len(close)


class TestMovingAverageCrossover:
    """Tests for MovingAverageCrossover."""

    def test_signal_generation(self, sample_ohlcv_data):
        """Test MA crossover signal generation."""
        strategy = MovingAverageCrossover(fast_period=10, slow_period=30)
        signals = strategy.generate_signals(close=sample_ohlcv_data["close"].values)

        assert len(signals) == len(sample_ohlcv_data)
        assert set(np.unique(signals)).issubset({-1, 0, 1})

    def test_warmup_period(self):
        """Test warmup period is slow_period."""
        close = np.linspace(100, 110, 100)
        strategy = MovingAverageCrossover(fast_period=10, slow_period=30)
        signals = strategy.generate_signals(close=close)

        # First slow_period signals should be 0
        assert all(signals[:30] == 0)

    def test_crossover_detection(self):
        """Test that crossovers are detected."""
        # Create data where fast MA crosses slow MA
        close = np.concatenate([
            np.linspace(100, 90, 50),   # Downtrend
            np.linspace(90, 110, 50),   # Uptrend
        ])

        strategy = MovingAverageCrossover(fast_period=5, slow_period=20)
        signals = strategy.generate_signals(close=close)

        # Should have signal changes
        signal_changes = np.abs(np.diff(signals[signals != 0]))
        assert np.sum(signal_changes) > 0

    def test_reset(self):
        """Test strategy reset."""
        strategy = MovingAverageCrossover(fast_period=10, slow_period=30)
        strategy._prices = [1, 2, 3]  # Add some state

        strategy.reset()
        assert len(strategy._prices) == 0


class TestBaseStrategy:
    """Tests for BaseStrategy interface."""

    def test_name_property(self):
        """Test strategy name."""
        strategy = MomentumStrategy(lookback=20)
        assert strategy.name == "MomentumStrategy"

    def test_params(self):
        """Test parameter storage."""
        strategy = MomentumStrategy(lookback=20)
        assert strategy.params == {"lookback": 20}

    def test_repr(self):
        """Test string representation."""
        strategy = MomentumStrategy(lookback=20)
        repr_str = repr(strategy)
        assert "MomentumStrategy" in repr_str
        assert "lookback=20" in repr_str

    def test_not_implemented(self, sample_ohlcv_data):
        """Test that abstract methods raise NotImplementedError."""

        class EmptyStrategy(BaseStrategy):
            pass

        strategy = EmptyStrategy()

        with pytest.raises(NotImplementedError):
            strategy.generate_signals(close=sample_ohlcv_data["close"].values)
