"""
Tests for VectorForge analysis tools.
"""

import numpy as np
import pandas as pd
import pytest

from vectorforge.analysis.metrics import PerformanceMetrics
from vectorforge.analysis.drawdown import DrawdownAnalyzer
from vectorforge.analysis.trades import TradeAnalyzer


class TestPerformanceMetrics:
    """Tests for PerformanceMetrics."""

    @pytest.fixture
    def sample_returns(self):
        """Generate sample returns series."""
        np.random.seed(42)
        returns = pd.Series(
            np.random.normal(0.0005, 0.02, 252),
            index=pd.date_range("2023-01-01", periods=252, freq="B"),
        )
        return returns

    def test_sharpe_ratio(self, sample_returns):
        """Test Sharpe ratio calculation."""
        metrics = PerformanceMetrics(sample_returns)
        sharpe = metrics.sharpe_ratio()

        assert isinstance(sharpe, float)
        assert not np.isnan(sharpe)

    def test_sortino_ratio(self, sample_returns):
        """Test Sortino ratio calculation."""
        metrics = PerformanceMetrics(sample_returns)
        sortino = metrics.sortino_ratio()

        assert isinstance(sortino, float)
        # Sortino should be >= Sharpe for positive skew
        sharpe = metrics.sharpe_ratio()

    def test_max_drawdown(self, sample_returns):
        """Test max drawdown calculation."""
        metrics = PerformanceMetrics(sample_returns)
        max_dd = metrics.max_drawdown()

        assert max_dd <= 0  # Drawdown is always non-positive
        assert max_dd >= -1  # Can't lose more than 100%

    def test_var_cvar(self, sample_returns):
        """Test VaR and CVaR calculations."""
        metrics = PerformanceMetrics(sample_returns)

        var = metrics.var(confidence=0.95)
        cvar = metrics.cvar(confidence=0.95)

        assert var < 0  # VaR is a loss
        assert cvar <= var  # CVaR is worse than VaR

    def test_win_rate(self, sample_returns):
        """Test win rate calculation."""
        metrics = PerformanceMetrics(sample_returns)
        win_rate = metrics.win_rate()

        assert 0 <= win_rate <= 1

    def test_profit_factor(self, sample_returns):
        """Test profit factor calculation."""
        metrics = PerformanceMetrics(sample_returns)
        pf = metrics.profit_factor()

        assert pf >= 0

    def test_generate_report(self, sample_returns):
        """Test report generation."""
        metrics = PerformanceMetrics(sample_returns)
        report = metrics.generate_report()

        assert isinstance(report, dict)
        assert "Sharpe Ratio" in report
        assert "Max Drawdown" in report
        assert "Win Rate" in report

    def test_with_benchmark(self, sample_returns):
        """Test information ratio with benchmark."""
        benchmark = pd.Series(
            np.random.normal(0.0003, 0.015, 252),
            index=sample_returns.index,
        )

        metrics = PerformanceMetrics(sample_returns, benchmark=benchmark)
        ir = metrics.information_ratio()

        assert isinstance(ir, float)
        assert not np.isnan(ir)


class TestDrawdownAnalyzer:
    """Tests for DrawdownAnalyzer."""

    @pytest.fixture
    def sample_equity(self):
        """Generate sample equity curve."""
        returns = pd.Series(
            np.random.normal(0.001, 0.02, 252),
            index=pd.date_range("2023-01-01", periods=252, freq="B"),
        )
        equity = 100000 * (1 + returns).cumprod()
        return equity

    def test_max_drawdown(self, sample_equity):
        """Test max drawdown calculation."""
        analyzer = DrawdownAnalyzer(sample_equity)
        max_dd = analyzer.max_drawdown()

        assert max_dd <= 0
        assert max_dd >= -1

    def test_drawdown_series(self, sample_equity):
        """Test drawdown series generation."""
        analyzer = DrawdownAnalyzer(sample_equity)
        dd_series = analyzer.drawdown_series()

        assert len(dd_series) == len(sample_equity)
        assert (dd_series <= 0).all()

    def test_time_underwater(self, sample_equity):
        """Test time underwater calculation."""
        analyzer = DrawdownAnalyzer(sample_equity)
        time_uw = analyzer.time_underwater()

        assert 0 <= time_uw <= 1

    def test_ulcer_index(self, sample_equity):
        """Test Ulcer Index calculation."""
        analyzer = DrawdownAnalyzer(sample_equity)
        ulcer = analyzer.ulcer_index()

        assert ulcer >= 0

    def test_summary(self, sample_equity):
        """Test summary generation."""
        analyzer = DrawdownAnalyzer(sample_equity)
        summary = analyzer.summary()

        assert isinstance(summary, dict)
        assert "Max Drawdown" in summary
        assert "Recovery Factor" in summary


class TestTradeAnalyzer:
    """Tests for TradeAnalyzer."""

    @pytest.fixture
    def sample_trades(self):
        """Generate sample trades DataFrame."""
        np.random.seed(42)
        n_trades = 50

        trades = pd.DataFrame({
            "timestamp": pd.date_range("2023-01-01", periods=n_trades, freq="W"),
            "symbol": ["AAPL"] * n_trades,
            "side": np.random.choice(["buy", "sell"], n_trades),
            "quantity": np.random.randint(10, 100, n_trades),
            "price": np.random.uniform(150, 200, n_trades),
            "pnl": np.random.normal(100, 500, n_trades),
            "commission": np.random.uniform(1, 10, n_trades),
        })

        return trades

    def test_total_trades(self, sample_trades):
        """Test total trades count."""
        analyzer = TradeAnalyzer(sample_trades)
        assert analyzer.total_trades == 50

    def test_win_rate(self, sample_trades):
        """Test win rate calculation."""
        analyzer = TradeAnalyzer(sample_trades)
        win_rate = analyzer.win_rate()

        assert 0 <= win_rate <= 1

    def test_profit_factor(self, sample_trades):
        """Test profit factor calculation."""
        analyzer = TradeAnalyzer(sample_trades)
        pf = analyzer.profit_factor()

        assert pf >= 0

    def test_expectancy(self, sample_trades):
        """Test expectancy calculation."""
        analyzer = TradeAnalyzer(sample_trades)
        expectancy = analyzer.expectancy()

        assert isinstance(expectancy, float)

    def test_consecutive_stats(self, sample_trades):
        """Test consecutive wins/losses."""
        analyzer = TradeAnalyzer(sample_trades)

        cons_wins = analyzer.consecutive_wins()
        cons_losses = analyzer.consecutive_losses()

        assert cons_wins >= 0
        assert cons_losses >= 0

    def test_summary(self, sample_trades):
        """Test summary generation."""
        analyzer = TradeAnalyzer(sample_trades)
        summary = analyzer.summary()

        assert isinstance(summary, dict)
        assert "Win Rate" in summary
        assert "Profit Factor" in summary
        assert "Expectancy" in summary

    def test_empty_trades(self):
        """Test handling of empty trades."""
        empty = pd.DataFrame()
        analyzer = TradeAnalyzer(empty)

        assert analyzer.total_trades == 0
        assert analyzer.win_rate() == 0
