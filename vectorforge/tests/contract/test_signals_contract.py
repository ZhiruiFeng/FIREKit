"""
Contract Tests for Cross-Sectional Signals

These tests define the expected API contract for signal generation.
"""

import numpy as np
import pandas as pd

from vectorforge.portfolio.data import PortfolioData
from vectorforge.portfolio.signals import (
    CrossSectionalSignal,
    RankMethod,
    SignalResult,
    TargetWeights,
)


class TestCrossSectionalSignal:
    """Contract tests for CrossSectionalSignal."""

    def test_momentum_signal_creation(self):
        """momentum() should create momentum signal generator."""
        signal = CrossSectionalSignal.momentum(lookback=252, skip_recent=21)

        assert signal.lookback == 252
        assert hasattr(signal, "skip_recent")
        assert signal.skip_recent == 21

    def test_signal_generate_returns_signal_result(self):
        """generate() should return SignalResult with values and ranks."""
        portfolio_data = self._make_portfolio_data()
        signal = CrossSectionalSignal.momentum(lookback=60, skip_recent=5)

        result = signal.generate(portfolio_data)

        assert isinstance(result, SignalResult)
        assert hasattr(result, "values")
        assert hasattr(result, "ranks")
        assert result.values.shape == (portfolio_data.n_symbols, portfolio_data.n_dates)

    def test_signal_result_has_correct_shape(self):
        """SignalResult should have arrays matching portfolio dimensions."""
        portfolio_data = self._make_portfolio_data()
        signal = CrossSectionalSignal.momentum(lookback=60, skip_recent=5)

        result = signal.generate(portfolio_data)

        n_symbols = portfolio_data.n_symbols
        n_dates = portfolio_data.n_dates
        assert result.values.shape == (n_symbols, n_dates)
        assert result.ranks.shape == (n_symbols, n_dates)
        assert result.symbols == portfolio_data.symbols
        assert len(result.dates) == n_dates

    def test_mean_reversion_signal(self):
        """mean_reversion() should create reversal signal."""
        signal = CrossSectionalSignal.mean_reversion(lookback=20)

        assert signal.lookback == 20
        assert signal._name == "mean_reversion"

    def test_volatility_signal(self):
        """volatility() should create volatility signal."""
        signal = CrossSectionalSignal.volatility(lookback=60, inverse=True)

        assert signal.lookback == 60
        assert signal.inverse is True

    def test_custom_signal(self):
        """custom() should create signal from user function."""

        def custom_func(prices):
            # Simple example: price change over last period
            return np.diff(prices, axis=1, prepend=prices[:, :1])

        signal = CrossSectionalSignal.custom(
            func=custom_func,
            lookback=1,
            name="price_change",
        )

        assert signal.lookback == 1
        assert signal._name == "price_change"

    def test_signal_ranks_are_percentiles(self):
        """Default ranks should be percentile (0-100)."""
        portfolio_data = self._make_portfolio_data()
        signal = CrossSectionalSignal.momentum(lookback=60, skip_recent=5)

        result = signal.generate(portfolio_data)

        # Check a column with valid ranks
        for j in range(result.ranks.shape[1]):
            col_ranks = result.ranks[:, j]
            valid = ~np.isnan(col_ranks)
            if valid.any():
                assert col_ranks[valid].min() >= 0
                assert col_ranks[valid].max() <= 100

    @staticmethod
    def _make_portfolio_data() -> PortfolioData:
        """Create sample portfolio data."""
        dates = pd.date_range("2023-01-01", periods=120, freq="B")
        n = len(dates)

        np.random.seed(42)
        symbols = ["SYM1", "SYM2", "SYM3", "SYM4", "SYM5"]
        data = {}

        for i, sym in enumerate(symbols):
            prices = 100 * np.exp(np.cumsum(np.random.normal(0.0005, 0.02, n)))
            data[sym] = pd.DataFrame(
                {
                    "open": prices * 0.99,
                    "high": prices * 1.01,
                    "low": prices * 0.98,
                    "close": prices,
                    "volume": [1e6] * n,
                },
                index=dates,
            )

        return PortfolioData.from_dict(data).align().validate(min_dates=60)


class TestSignalResultFiltering:
    """Contract tests for SignalResult filtering methods."""

    def test_top_percentile_filters_to_top_n(self):
        """top_percentile(20) should keep only top 20% of assets."""
        portfolio_data = self._make_portfolio_data()
        signal = CrossSectionalSignal.momentum(lookback=60, skip_recent=5)

        result = signal.generate(portfolio_data)
        filtered = result.top_percentile(20)

        # Check that non-top assets are NaN
        for j in range(filtered.ranks.shape[1]):
            col_ranks = filtered.ranks[:, j]
            valid = ~np.isnan(col_ranks)
            if valid.any():
                # All valid ranks should be >= 80 (top 20%)
                assert col_ranks[valid].min() >= 80

    def test_bottom_percentile_filters_to_bottom_n(self):
        """bottom_percentile(20) should keep only bottom 20% of assets."""
        portfolio_data = self._make_portfolio_data()
        signal = CrossSectionalSignal.momentum(lookback=60, skip_recent=5)

        result = signal.generate(portfolio_data)
        filtered = result.bottom_percentile(20)

        # Check that non-bottom assets are NaN
        for j in range(filtered.ranks.shape[1]):
            col_ranks = filtered.ranks[:, j]
            valid = ~np.isnan(col_ranks)
            if valid.any():
                # All valid ranks should be <= 20
                assert col_ranks[valid].max() <= 20

    def test_between_percentile_filters_range(self):
        """between_percentile(30, 70) should keep middle 40%."""
        portfolio_data = self._make_portfolio_data()
        signal = CrossSectionalSignal.momentum(lookback=60, skip_recent=5)

        result = signal.generate(portfolio_data)
        filtered = result.between_percentile(30, 70)

        for j in range(filtered.ranks.shape[1]):
            col_ranks = filtered.ranks[:, j]
            valid = ~np.isnan(col_ranks)
            if valid.any():
                assert col_ranks[valid].min() >= 30
                assert col_ranks[valid].max() <= 70

    def test_to_weights_equal_weighting(self):
        """to_weights(method='equal') should create equal weights."""
        portfolio_data = self._make_portfolio_data()
        signal = CrossSectionalSignal.momentum(lookback=60, skip_recent=5)

        result = signal.generate(portfolio_data)
        filtered = result.top_percentile(40)  # Top 40% = 2 stocks
        weights = filtered.to_weights(method="equal")

        assert isinstance(weights, TargetWeights)

        # Check weights sum to 1 (approximately)
        for j in range(weights.n_dates):
            col_weights = weights.weights[:, j]
            weight_sum = col_weights.sum()
            if weight_sum > 0:
                assert abs(weight_sum - 1.0) < 0.01

    def test_to_weights_signal_weighted(self):
        """to_weights(method='signal_weighted') should weight by signal value."""
        portfolio_data = self._make_portfolio_data()
        signal = CrossSectionalSignal.momentum(lookback=60, skip_recent=5)

        result = signal.generate(portfolio_data)
        weights = result.top_percentile(50).to_weights(method="signal_weighted")

        # Weights should still sum to 1
        for j in range(weights.n_dates):
            col_weights = weights.weights[:, j]
            weight_sum = col_weights.sum()
            if weight_sum > 0:
                assert abs(weight_sum - 1.0) < 0.01

    @staticmethod
    def _make_portfolio_data() -> PortfolioData:
        """Create sample portfolio data."""
        dates = pd.date_range("2023-01-01", periods=120, freq="B")
        n = len(dates)

        np.random.seed(42)
        symbols = ["A", "B", "C", "D", "E"]
        data = {}

        for i, sym in enumerate(symbols):
            # Different drift per symbol to create ranking spread
            drift = -0.001 + i * 0.0005
            prices = 100 * np.exp(np.cumsum(np.random.normal(drift, 0.02, n)))
            data[sym] = pd.DataFrame(
                {
                    "open": prices * 0.99,
                    "high": prices * 1.01,
                    "low": prices * 0.98,
                    "close": prices,
                    "volume": [1e6] * n,
                },
                index=dates,
            )

        return PortfolioData.from_dict(data).align().validate(min_dates=60)


class TestRankMethod:
    """Contract tests for RankMethod enum."""

    def test_rank_method_values(self):
        """RankMethod should have expected values."""
        assert RankMethod.PERCENTILE.value == "percentile"
        assert RankMethod.FRACTIONAL.value == "fractional"
        assert RankMethod.ORDINAL.value == "ordinal"

    def test_fractional_rank_method(self):
        """Fractional rank should produce 0-1 values."""
        portfolio_data = self._make_portfolio_data()
        signal = CrossSectionalSignal.momentum(
            lookback=60,
            skip_recent=5,
            rank_method=RankMethod.FRACTIONAL,
        )

        result = signal.generate(portfolio_data)

        for j in range(result.ranks.shape[1]):
            col_ranks = result.ranks[:, j]
            valid = ~np.isnan(col_ranks)
            if valid.any():
                assert col_ranks[valid].min() >= 0
                assert col_ranks[valid].max() <= 1

    @staticmethod
    def _make_portfolio_data() -> PortfolioData:
        dates = pd.date_range("2023-01-01", periods=120, freq="B")
        n = len(dates)
        np.random.seed(42)

        data = {}
        for i, sym in enumerate(["X", "Y", "Z"]):
            prices = 100 * np.exp(np.cumsum(np.random.normal(0.0005, 0.02, n)))
            data[sym] = pd.DataFrame(
                {
                    "open": prices * 0.99,
                    "high": prices * 1.01,
                    "low": prices * 0.98,
                    "close": prices,
                    "volume": [1e6] * n,
                },
                index=dates,
            )

        return PortfolioData.from_dict(data).align().validate(min_dates=60)
