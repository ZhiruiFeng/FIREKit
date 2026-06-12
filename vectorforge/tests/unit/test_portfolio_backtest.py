"""
Unit tests for portfolio backtesting (T023, US1).

Covers VectorizedBacktester.run_portfolio(), portfolio equity curve
calculation, per-asset contribution tracking, and basic portfolio
Sharpe/Sortino/drawdown metrics (FR-001, FR-017, FR-022).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vectorforge import VectorizedBacktester
from vectorforge.engine.base import PortfolioBacktestResult
from vectorforge.portfolio.data import PortfolioData
from vectorforge.portfolio.signals import TargetWeights
from vectorforge.strategy.base import PortfolioStrategy


def make_ohlcv_df(close: np.ndarray, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Build a simple OHLCV DataFrame from a close price series."""
    close = np.asarray(close, dtype=float)
    return pd.DataFrame(
        {
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": np.full(len(close), 1e6),
        },
        index=dates,
    )


def constant_growth_prices(n: int, daily_return: float, initial: float = 100.0) -> np.ndarray:
    """Price series compounding at a constant daily return."""
    return initial * np.cumprod(np.full(n, 1.0 + daily_return))


@pytest.fixture
def dates() -> pd.DatetimeIndex:
    return pd.date_range("2023-01-02", periods=126, freq="B")


@pytest.fixture
def identical_growth_data(dates: pd.DatetimeIndex) -> PortfolioData:
    """Two assets with identical +1%/day returns (different price levels)."""
    n = len(dates)
    g = constant_growth_prices(n, 0.01)
    return PortfolioData.from_dict(
        {
            "A": make_ohlcv_df(g, dates),
            "B": make_ohlcv_df(2.0 * g, dates),
        }
    )


@pytest.fixture
def mixed_growth_data(dates: pd.DatetimeIndex) -> PortfolioData:
    """Asset A grows +1%/day, asset B is flat."""
    n = len(dates)
    return PortfolioData.from_dict(
        {
            "A": make_ohlcv_df(constant_growth_prices(n, 0.01), dates),
            "B": make_ohlcv_df(np.full(n, 100.0), dates),
        }
    )


@pytest.fixture
def engine() -> VectorizedBacktester:
    return VectorizedBacktester()


class TestRunPortfolioBasics:
    """run_portfolio() returns a well-formed PortfolioBacktestResult."""

    def test_returns_portfolio_backtest_result(self, engine, identical_growth_data):
        weights = TargetWeights.equal_weight(
            identical_growth_data.symbols, identical_growth_data.dates
        )
        result = engine.run_portfolio(weights, identical_growth_data, initial_capital=100_000)

        assert isinstance(result, PortfolioBacktestResult)

    def test_result_has_portfolio_fields(self, engine, identical_growth_data):
        weights = TargetWeights.equal_weight(
            identical_growth_data.symbols, identical_growth_data.dates
        )
        result = engine.run_portfolio(weights, identical_growth_data, initial_capital=100_000)

        assert isinstance(result.weights_history, pd.DataFrame)
        assert isinstance(result.asset_returns, pd.DataFrame)
        assert isinstance(result.asset_contributions, pd.DataFrame)
        assert isinstance(result.rebalance_dates, list)
        assert isinstance(result.turnover_history, pd.Series)

    def test_equity_curve_starts_at_initial_capital(self, engine, identical_growth_data):
        weights = TargetWeights.equal_weight(
            identical_growth_data.symbols, identical_growth_data.dates
        )
        result = engine.run_portfolio(weights, identical_growth_data, initial_capital=250_000)

        assert result.equity_curve.iloc[0] == pytest.approx(250_000)
        assert result.initial_capital == pytest.approx(250_000)

    def test_series_lengths_and_indices(self, engine, identical_growth_data):
        n = identical_growth_data.n_dates
        weights = TargetWeights.equal_weight(
            identical_growth_data.symbols, identical_growth_data.dates
        )
        result = engine.run_portfolio(weights, identical_growth_data, initial_capital=100_000)

        assert len(result.equity_curve) == n
        assert len(result.returns) == n - 1
        assert result.equity_curve.index.equals(identical_growth_data.dates)
        assert result.returns.index.equals(identical_growth_data.dates[1:])
        assert result.start_date == identical_growth_data.dates[0]
        assert result.end_date == identical_growth_data.dates[-1]

    def test_weights_history_columns_match_symbols(self, engine, identical_growth_data):
        weights = TargetWeights.equal_weight(
            identical_growth_data.symbols, identical_growth_data.dates
        )
        result = engine.run_portfolio(weights, identical_growth_data, initial_capital=100_000)

        assert list(result.weights_history.columns) == identical_growth_data.symbols
        assert len(result.weights_history) == identical_growth_data.n_dates

    def test_invalid_strategy_raises_value_error(self, engine, identical_growth_data):
        with pytest.raises(ValueError, match="TargetWeights or PortfolioStrategy"):
            engine.run_portfolio("not-a-strategy", identical_growth_data)


class TestPortfolioReturnCalculation:
    """Aggregate portfolio returns are the weighted sum of asset returns."""

    def test_identical_assets_equal_weight_compounds_exactly(
        self, engine, identical_growth_data
    ):
        """Two assets both at +1%/day -> portfolio compounds at +1%/day."""
        n = identical_growth_data.n_dates
        weights = TargetWeights.equal_weight(
            identical_growth_data.symbols, identical_growth_data.dates
        )
        result = engine.run_portfolio(weights, identical_growth_data, initial_capital=100_000)

        expected_total = 1.01 ** (n - 1) - 1
        # float32 price storage -> small tolerance
        assert result.total_return == pytest.approx(expected_total, rel=1e-4)
        assert result.final_capital == pytest.approx(100_000 * (1 + expected_total), rel=1e-4)
        # Every daily return should be ~1%
        assert np.allclose(result.returns.values, 0.01, atol=1e-5)

    def test_single_asset_full_weight_tracks_price(self, engine, dates):
        """100% in one asset -> portfolio return equals the asset's return."""
        n = len(dates)
        rng = np.random.default_rng(7)
        close = 100 * np.exp(np.cumsum(rng.normal(0.0005, 0.02, n)))
        data = PortfolioData.from_dict({"ONLY": make_ohlcv_df(close, dates)})
        stored_close = data.close()[0]  # float32-stored prices

        weights = TargetWeights.equal_weight(["ONLY"], dates)
        result = engine.run_portfolio(weights, data, initial_capital=100_000)

        expected_total = stored_close[-1] / stored_close[0] - 1
        assert result.total_return == pytest.approx(expected_total, rel=1e-4)

    def test_first_period_return_is_weighted_sum(self, engine, mixed_growth_data):
        """With weights (0.6, 0.4) on +1%/flat assets, day-1 return is 0.6%."""
        n = mixed_growth_data.n_dates
        weights_array = np.tile(np.array([[0.6], [0.4]]), (1, n))
        weights = TargetWeights.from_array(
            weights_array, mixed_growth_data.symbols, mixed_growth_data.dates
        )
        result = engine.run_portfolio(weights, mixed_growth_data, initial_capital=100_000)

        assert result.returns.iloc[0] == pytest.approx(0.6 * 0.01, rel=1e-4)

    def test_zero_weight_asset_does_not_affect_returns(self, engine, mixed_growth_data):
        """Weights (1, 0): portfolio compounds at asset A's +1%/day only."""
        n = mixed_growth_data.n_dates
        weights_array = np.tile(np.array([[1.0], [0.0]]), (1, n))
        weights = TargetWeights.from_array(
            weights_array, mixed_growth_data.symbols, mixed_growth_data.dates
        )
        result = engine.run_portfolio(weights, mixed_growth_data, initial_capital=100_000)

        assert result.total_return == pytest.approx(1.01 ** (n - 1) - 1, rel=1e-4)


class TestEquityCurveConsistency:
    """Equity curve is internally consistent with returns (FR-001/US1)."""

    def test_equity_curve_matches_compounded_returns(self, engine, mixed_growth_data):
        weights = TargetWeights.equal_weight(
            mixed_growth_data.symbols, mixed_growth_data.dates
        )
        capital = 100_000.0
        result = engine.run_portfolio(weights, mixed_growth_data, initial_capital=capital)

        reconstructed = capital * np.cumprod(np.insert(1 + result.returns.values, 0, 1.0))
        assert np.allclose(result.equity_curve.values, reconstructed, rtol=1e-10)

    def test_total_return_matches_equity_curve(self, engine, mixed_growth_data):
        weights = TargetWeights.equal_weight(
            mixed_growth_data.symbols, mixed_growth_data.dates
        )
        result = engine.run_portfolio(weights, mixed_growth_data, initial_capital=100_000)

        assert result.total_return == pytest.approx(
            result.equity_curve.iloc[-1] / result.equity_curve.iloc[0] - 1, rel=1e-10
        )


class TestPerAssetContributions:
    """Per-asset contribution tracking (FR-022, US1 scenario 2)."""

    def test_contributions_sum_to_portfolio_return(self, engine, mixed_growth_data):
        weights = TargetWeights.equal_weight(
            mixed_growth_data.symbols, mixed_growth_data.dates
        )
        result = engine.run_portfolio(weights, mixed_growth_data, initial_capital=100_000)

        contrib_sum = result.asset_contributions.sum(axis=1)
        assert np.allclose(contrib_sum.values, result.returns.values, atol=1e-12)

    def test_zero_weight_asset_contributes_zero(self, engine, mixed_growth_data):
        n = mixed_growth_data.n_dates
        weights_array = np.tile(np.array([[1.0], [0.0]]), (1, n))
        weights = TargetWeights.from_array(
            weights_array, mixed_growth_data.symbols, mixed_growth_data.dates
        )
        result = engine.run_portfolio(weights, mixed_growth_data, initial_capital=100_000)

        assert np.allclose(result.asset_contributions["B"].values, 0.0)
        assert np.allclose(
            result.asset_contributions["A"].values, result.returns.values, atol=1e-12
        )

    def test_contributions_have_asset_columns(self, engine, identical_growth_data):
        weights = TargetWeights.equal_weight(
            identical_growth_data.symbols, identical_growth_data.dates
        )
        result = engine.run_portfolio(weights, identical_growth_data, initial_capital=100_000)

        assert list(result.asset_contributions.columns) == identical_growth_data.symbols
        assert len(result.asset_contributions) == identical_growth_data.n_dates - 1


class TestWeightTracking:
    """Weight history reflects targets and drift."""

    def test_identical_assets_keep_equal_weights(self, engine, identical_growth_data):
        """Identical returns -> no drift -> weights stay exactly 50/50."""
        weights = TargetWeights.equal_weight(
            identical_growth_data.symbols, identical_growth_data.dates
        )
        result = engine.run_portfolio(weights, identical_growth_data, initial_capital=100_000)

        assert np.allclose(result.weights_history.values, 0.5, atol=1e-6)

    def test_weights_rows_sum_to_one(self, engine, mixed_growth_data):
        weights = TargetWeights.equal_weight(
            mixed_growth_data.symbols, mixed_growth_data.dates
        )
        result = engine.run_portfolio(weights, mixed_growth_data, initial_capital=100_000)

        row_sums = result.weights_history.sum(axis=1)
        assert np.allclose(row_sums.values, 1.0, atol=1e-9)


class TestPortfolioMetricsBasics:
    """Basic Sharpe/Sortino/drawdown calculations (FR-017)."""

    def test_monotonic_growth_has_zero_drawdown(self, engine, identical_growth_data):
        weights = TargetWeights.equal_weight(
            identical_growth_data.symbols, identical_growth_data.dates
        )
        result = engine.run_portfolio(weights, identical_growth_data, initial_capital=100_000)

        assert result.max_drawdown == pytest.approx(0.0, abs=1e-9)
        assert result.win_rate == pytest.approx(1.0)
        assert result.sharpe_ratio > 0

    def test_metrics_are_finite_for_random_data(self, engine, dates):
        n = len(dates)
        rng = np.random.default_rng(11)
        data = PortfolioData.from_dict(
            {
                f"S{i}": make_ohlcv_df(
                    100 * np.exp(np.cumsum(rng.normal(0.0003, 0.015, n))), dates
                )
                for i in range(4)
            }
        )
        weights = TargetWeights.equal_weight(data.symbols, data.dates)
        result = engine.run_portfolio(weights, data, initial_capital=100_000)

        assert np.isfinite(result.sharpe_ratio)
        assert np.isfinite(result.sortino_ratio)
        assert np.isfinite(result.annual_return)
        assert result.max_drawdown <= 0.0

    def test_annual_return_consistent_with_total_return(self, engine, identical_growth_data):
        n = identical_growth_data.n_dates
        weights = TargetWeights.equal_weight(
            identical_growth_data.symbols, identical_growth_data.dates
        )
        result = engine.run_portfolio(weights, identical_growth_data, initial_capital=100_000)

        n_years = (n - 1) / 252
        expected_annual = (1 + result.total_return) ** (1 / n_years) - 1
        assert result.annual_return == pytest.approx(expected_annual, rel=1e-9)


class TestPortfolioStrategyPath:
    """run_portfolio() accepts PortfolioStrategy subclasses."""

    def test_equal_weight_strategy_matches_target_weights(self, engine, mixed_growth_data):
        class EqualWeightStrategy(PortfolioStrategy):
            def generate_weights(self, portfolio_data, current_weights=None):
                n = portfolio_data.n_symbols
                return np.ones(n) / n

        strategy_result = engine.run_portfolio(
            EqualWeightStrategy(), mixed_growth_data, initial_capital=100_000
        )
        weights = TargetWeights.equal_weight(
            mixed_growth_data.symbols, mixed_growth_data.dates
        )
        weights_result = engine.run_portfolio(weights, mixed_growth_data, initial_capital=100_000)

        assert strategy_result.total_return == pytest.approx(
            weights_result.total_return, rel=1e-12
        )
        assert np.allclose(
            strategy_result.returns.values, weights_result.returns.values, atol=1e-15
        )

    def test_custom_allocation_strategy(self, engine, mixed_growth_data):
        class FavorFirstStrategy(PortfolioStrategy):
            def generate_weights(self, portfolio_data, current_weights=None):
                w = np.zeros(portfolio_data.n_symbols)
                w[0] = 1.0
                return w

        result = engine.run_portfolio(
            FavorFirstStrategy(), mixed_growth_data, initial_capital=100_000
        )
        n = mixed_growth_data.n_dates
        assert result.total_return == pytest.approx(1.01 ** (n - 1) - 1, rel=1e-4)
