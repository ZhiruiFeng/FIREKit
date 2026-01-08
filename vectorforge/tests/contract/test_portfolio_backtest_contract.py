"""
Contract Tests for Portfolio Backtest

These tests define the expected API contract for portfolio backtesting.
They must FAIL before implementation and PASS after.
"""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from vectorforge.portfolio.data import PortfolioData

# ============================================================================
# T014: Contract test for PortfolioBacktestResult
# ============================================================================


class TestPortfolioBacktestResult:
    """Contract tests for PortfolioBacktestResult."""

    def test_portfolio_backtest_result_has_weights_history(self):
        """PortfolioBacktestResult should have weights_history DataFrame."""
        from vectorforge.engine.base import PortfolioBacktestResult

        # Given: minimal result data
        dates = pd.date_range("2023-01-01", periods=10, freq="B")
        weights = pd.DataFrame(
            {"AAPL": [0.5] * 10, "GOOG": [0.5] * 10},
            index=dates,
        )
        returns = pd.Series([0.01] * 10, index=dates)
        equity = pd.Series([100000 * (1.01**i) for i in range(10)], index=dates)

        # When: Creating result
        result = PortfolioBacktestResult(
            total_return=0.1,
            annual_return=0.1,
            sharpe_ratio=1.5,
            sortino_ratio=2.0,
            calmar_ratio=1.0,
            max_drawdown=-0.05,
            total_trades=20,
            win_rate=0.6,
            profit_factor=1.5,
            avg_trade_return=0.001,
            equity_curve=equity,
            returns=returns,
            weights_history=weights,
            asset_returns=pd.DataFrame(),
        )

        # Then: Should have weights_history
        assert hasattr(result, "weights_history")
        assert isinstance(result.weights_history, pd.DataFrame)
        assert set(result.weights_history.columns) == {"AAPL", "GOOG"}

    def test_portfolio_backtest_result_has_asset_returns(self):
        """PortfolioBacktestResult should have asset_returns DataFrame."""
        from vectorforge.engine.base import PortfolioBacktestResult

        dates = pd.date_range("2023-01-01", periods=10, freq="B")
        asset_returns = pd.DataFrame(
            {"AAPL": [0.01] * 10, "GOOG": [0.02] * 10},
            index=dates,
        )

        result = PortfolioBacktestResult(
            total_return=0.1,
            annual_return=0.1,
            sharpe_ratio=1.5,
            sortino_ratio=2.0,
            calmar_ratio=1.0,
            max_drawdown=-0.05,
            total_trades=20,
            win_rate=0.6,
            profit_factor=1.5,
            avg_trade_return=0.001,
            equity_curve=pd.Series(),
            returns=pd.Series(),
            weights_history=pd.DataFrame(),
            asset_returns=asset_returns,
        )

        assert hasattr(result, "asset_returns")
        assert isinstance(result.asset_returns, pd.DataFrame)

    def test_portfolio_backtest_result_has_rebalance_dates(self):
        """PortfolioBacktestResult should track rebalance dates."""
        from vectorforge.engine.base import PortfolioBacktestResult

        rebalance_dates = [date(2023, 1, 2), date(2023, 2, 1)]

        result = PortfolioBacktestResult(
            total_return=0.1,
            annual_return=0.1,
            sharpe_ratio=1.5,
            sortino_ratio=2.0,
            calmar_ratio=1.0,
            max_drawdown=-0.05,
            total_trades=20,
            win_rate=0.6,
            profit_factor=1.5,
            avg_trade_return=0.001,
            equity_curve=pd.Series(),
            returns=pd.Series(),
            weights_history=pd.DataFrame(),
            asset_returns=pd.DataFrame(),
            rebalance_dates=rebalance_dates,
        )

        assert hasattr(result, "rebalance_dates")
        assert result.rebalance_dates == rebalance_dates

    def test_portfolio_backtest_result_has_turnover_history(self):
        """PortfolioBacktestResult should have turnover_history Series."""
        from vectorforge.engine.base import PortfolioBacktestResult

        dates = pd.date_range("2023-01-01", periods=3, freq="MS")
        turnover = pd.Series([0.0, 0.15, 0.12], index=dates)

        result = PortfolioBacktestResult(
            total_return=0.1,
            annual_return=0.1,
            sharpe_ratio=1.5,
            sortino_ratio=2.0,
            calmar_ratio=1.0,
            max_drawdown=-0.05,
            total_trades=20,
            win_rate=0.6,
            profit_factor=1.5,
            avg_trade_return=0.001,
            equity_curve=pd.Series(),
            returns=pd.Series(),
            weights_history=pd.DataFrame(),
            asset_returns=pd.DataFrame(),
            turnover_history=turnover,
        )

        assert hasattr(result, "turnover_history")
        assert isinstance(result.turnover_history, pd.Series)


# ============================================================================
# T015: Contract test for VectorizedEngine.run_portfolio()
# ============================================================================


class TestVectorizedEngineRunPortfolio:
    """Contract tests for VectorizedEngine.run_portfolio()."""

    def test_run_portfolio_accepts_portfolio_data(self):
        """run_portfolio() should accept PortfolioData and return PortfolioBacktestResult."""
        from vectorforge import VectorizedBacktester
        from vectorforge.engine.base import PortfolioBacktestResult
        from vectorforge.portfolio.signals import TargetWeights

        # Given: Portfolio data and weights
        portfolio_data = self._make_portfolio_data()
        weights = TargetWeights.equal_weight(
            symbols=portfolio_data.symbols,
            dates=portfolio_data.dates,
        )

        # When: Running portfolio backtest
        engine = VectorizedBacktester()
        result = engine.run_portfolio(
            strategy=weights,
            data=portfolio_data,
            initial_capital=100000,
        )

        # Then: Should return PortfolioBacktestResult
        assert isinstance(result, PortfolioBacktestResult)
        assert hasattr(result, "weights_history")
        assert hasattr(result, "asset_returns")

    def test_run_portfolio_computes_returns(self):
        """run_portfolio() should compute portfolio returns from weighted asset returns."""
        from vectorforge import VectorizedBacktester
        from vectorforge.portfolio.signals import TargetWeights

        portfolio_data = self._make_portfolio_data()
        weights = TargetWeights.equal_weight(
            symbols=portfolio_data.symbols,
            dates=portfolio_data.dates,
        )

        engine = VectorizedBacktester()
        result = engine.run_portfolio(
            strategy=weights,
            data=portfolio_data,
            initial_capital=100000,
        )

        # Returns should be computed
        assert len(result.returns) > 0
        assert result.total_return != 0 or len(portfolio_data.dates) < 10

    def test_run_portfolio_tracks_equity_curve(self):
        """run_portfolio() should track equity curve over time."""
        from vectorforge import VectorizedBacktester
        from vectorforge.portfolio.signals import TargetWeights

        portfolio_data = self._make_portfolio_data()
        weights = TargetWeights.equal_weight(
            symbols=portfolio_data.symbols,
            dates=portfolio_data.dates,
        )

        engine = VectorizedBacktester()
        result = engine.run_portfolio(
            strategy=weights,
            data=portfolio_data,
            initial_capital=100000,
        )

        # Equity curve should start at initial capital
        assert len(result.equity_curve) > 0
        assert result.equity_curve.iloc[0] == pytest.approx(100000, rel=0.01)

    def test_run_portfolio_computes_sharpe_ratio(self):
        """run_portfolio() should compute Sharpe ratio."""
        from vectorforge import VectorizedBacktester
        from vectorforge.portfolio.signals import TargetWeights

        portfolio_data = self._make_portfolio_data()
        weights = TargetWeights.equal_weight(
            symbols=portfolio_data.symbols,
            dates=portfolio_data.dates,
        )

        engine = VectorizedBacktester()
        result = engine.run_portfolio(
            strategy=weights,
            data=portfolio_data,
            initial_capital=100000,
        )

        # Sharpe should be a valid number
        assert np.isfinite(result.sharpe_ratio)

    @staticmethod
    def _make_portfolio_data() -> PortfolioData:
        """Create sample portfolio data for testing."""
        dates = pd.date_range("2023-01-01", periods=60, freq="B")
        n = len(dates)

        # Create deterministic price series
        np.random.seed(42)
        aapl_prices = 100 * np.exp(np.cumsum(np.random.normal(0.0005, 0.02, n)))
        goog_prices = 150 * np.exp(np.cumsum(np.random.normal(0.0003, 0.018, n)))

        aapl = pd.DataFrame(
            {
                "open": aapl_prices * 0.99,
                "high": aapl_prices * 1.01,
                "low": aapl_prices * 0.98,
                "close": aapl_prices,
                "volume": [1e6] * n,
            },
            index=dates,
        )

        goog = pd.DataFrame(
            {
                "open": goog_prices * 0.99,
                "high": goog_prices * 1.01,
                "low": goog_prices * 0.98,
                "close": goog_prices,
                "volume": [5e5] * n,
            },
            index=dates,
        )

        return PortfolioData.from_dict({"AAPL": aapl, "GOOG": goog}).align().validate(min_dates=30)


# ============================================================================
# Additional contract tests
# ============================================================================


class TestPortfolioStrategy:
    """Contract tests for PortfolioStrategy base class."""

    def test_portfolio_strategy_has_generate_weights(self):
        """PortfolioStrategy should have generate_weights method."""
        from vectorforge.strategy.base import PortfolioStrategy

        class TestStrategy(PortfolioStrategy):
            def generate_weights(self, portfolio_data, current_weights=None):
                # Simple equal weight
                n = portfolio_data.n_symbols
                return np.ones(n) / n

        strategy = TestStrategy()
        assert hasattr(strategy, "generate_weights")
        assert callable(strategy.generate_weights)

    def test_portfolio_strategy_can_be_subclassed(self):
        """Users should be able to create custom PortfolioStrategy."""
        from vectorforge.strategy.base import PortfolioStrategy

        class MomentumPortfolioStrategy(PortfolioStrategy):
            def __init__(self, lookback: int = 252):
                super().__init__(lookback=lookback)
                self.lookback = lookback

            def generate_weights(self, portfolio_data, current_weights=None):
                # Calculate momentum
                returns = portfolio_data.returns(self.lookback)
                mom = returns.sum(axis=1)
                # Long top half, short bottom half
                weights = np.where(mom > np.median(mom), 1.0, 0.0)
                return weights / weights.sum()

        strategy = MomentumPortfolioStrategy(lookback=126)
        assert strategy.lookback == 126


class TestTargetWeights:
    """Contract tests for TargetWeights dataclass."""

    def test_equal_weight_creates_uniform_weights(self):
        """equal_weight() should create uniform weights across symbols."""
        from vectorforge.portfolio.signals import TargetWeights

        dates = pd.date_range("2023-01-01", periods=10, freq="B")
        symbols = ["AAPL", "GOOG", "MSFT"]

        weights = TargetWeights.equal_weight(symbols=symbols, dates=dates)

        # All weights should be 1/3
        assert weights.n_symbols == 3
        assert weights.n_dates == 10

        w = weights.get_weights_at(dates[0])
        assert len(w) == 3
        assert all(abs(v - 1 / 3) < 0.01 for v in w.values())

    def test_target_weights_has_weights_array(self):
        """TargetWeights should have weights as numpy array."""
        from vectorforge.portfolio.signals import TargetWeights

        dates = pd.date_range("2023-01-01", periods=10, freq="B")

        weights = TargetWeights.equal_weight(
            symbols=["A", "B"],
            dates=dates,
        )

        assert isinstance(weights.weights, np.ndarray)
        assert weights.weights.shape == (2, 10)

    def test_get_weights_at_returns_dict(self):
        """get_weights_at() should return symbol -> weight dict."""
        from vectorforge.portfolio.signals import TargetWeights

        dates = pd.date_range("2023-01-01", periods=5, freq="B")

        weights = TargetWeights.equal_weight(
            symbols=["X", "Y"],
            dates=dates,
        )

        w = weights.get_weights_at(dates[2])

        assert isinstance(w, dict)
        assert "X" in w and "Y" in w
        assert w["X"] == pytest.approx(0.5)
        assert w["Y"] == pytest.approx(0.5)
