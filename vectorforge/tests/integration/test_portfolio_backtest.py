"""
Integration test: 50-symbol momentum portfolio backtest (T024, US1).

End-to-end flow per US1 acceptance scenarios:
data loading -> alignment -> momentum signal -> top-quintile weights ->
monthly-rebalanced backtest -> aggregate returns, equity curve, and
per-asset contributions (FR-001, FR-002, FR-017, FR-022).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tests.fixtures.portfolio_data import generate_sp500_like_data
from vectorforge import VectorizedBacktester
from vectorforge.engine.base import PortfolioBacktestResult
from vectorforge.portfolio.data import MissingDataPolicy, PortfolioData
from vectorforge.portfolio.rebalancer import RebalanceFrequency, Rebalancer
from vectorforge.portfolio.signals import CrossSectionalSignal, TargetWeights

N_SYMBOLS = 50
N_DAYS = 504  # ~2 years of daily data
LOOKBACK = 126
SKIP_RECENT = 21
INITIAL_CAPITAL = 1_000_000.0


@pytest.fixture(scope="module")
def portfolio_data() -> PortfolioData:
    """50 correlated symbols, aligned and validated."""
    raw = generate_sp500_like_data(n_symbols=N_SYMBOLS, n_days=N_DAYS, seed=42)
    data = PortfolioData.from_dict(raw)
    return data.align(policy=MissingDataPolicy.FORWARD_FILL).validate(
        min_dates=100, max_gap_days=5
    )


@pytest.fixture(scope="module")
def momentum_weights(portfolio_data: PortfolioData) -> TargetWeights:
    signal = CrossSectionalSignal.momentum(lookback=LOOKBACK, skip_recent=SKIP_RECENT)
    result = signal.generate(portfolio_data)
    return result.top_percentile(20).to_weights("equal")


@pytest.fixture(scope="module")
def backtest_result(
    portfolio_data: PortfolioData, momentum_weights: TargetWeights
) -> PortfolioBacktestResult:
    engine = VectorizedBacktester()
    rebalancer = Rebalancer.calendar(RebalanceFrequency.MONTHLY)
    return engine.run_portfolio(
        strategy=momentum_weights,
        data=portfolio_data,
        initial_capital=INITIAL_CAPITAL,
        rebalancer=rebalancer,
    )


class TestFiftySymbolMomentumBacktest:
    """US1 scenario 1: portfolio-level results for a 50-symbol momentum run."""

    def test_data_universe_is_50_symbols(self, portfolio_data):
        assert portfolio_data.n_symbols == N_SYMBOLS
        assert portfolio_data.n_dates == N_DAYS

    def test_backtest_returns_portfolio_result(self, backtest_result):
        assert isinstance(backtest_result, PortfolioBacktestResult)
        assert backtest_result.mode == "vectorized_portfolio"

    def test_equity_curve_aggregated_across_positions(self, portfolio_data, backtest_result):
        equity = backtest_result.equity_curve
        assert len(equity) == portfolio_data.n_dates
        assert equity.iloc[0] == pytest.approx(INITIAL_CAPITAL)
        assert equity.index.equals(portfolio_data.dates)
        assert np.isfinite(equity.values).all()
        assert (equity.values > 0).all()

    def test_equity_curve_consistent_with_returns(self, backtest_result):
        """Equity curve must equal compounded portfolio returns."""
        equity = backtest_result.equity_curve.values
        returns = backtest_result.returns.values
        reconstructed = INITIAL_CAPITAL * np.cumprod(np.insert(1 + returns, 0, 1.0))
        assert np.allclose(equity, reconstructed, rtol=1e-9)

    def test_total_return_matches_equity_endpoints(self, backtest_result):
        equity = backtest_result.equity_curve
        assert backtest_result.total_return == pytest.approx(
            equity.iloc[-1] / equity.iloc[0] - 1, rel=1e-9
        )
        assert backtest_result.final_capital == pytest.approx(equity.iloc[-1], rel=1e-12)

    def test_aggregate_metrics_are_finite(self, backtest_result):
        assert np.isfinite(backtest_result.sharpe_ratio)
        assert np.isfinite(backtest_result.sortino_ratio)
        assert np.isfinite(backtest_result.annual_return)
        assert -1.0 <= backtest_result.max_drawdown <= 0.0
        assert 0.0 <= backtest_result.win_rate <= 1.0

    def test_warmup_period_has_flat_equity(self, backtest_result):
        """No positions can exist before the momentum lookback has elapsed."""
        warmup_equity = backtest_result.equity_curve.iloc[: LOOKBACK - 1]
        assert np.allclose(warmup_equity.values, INITIAL_CAPITAL, rtol=1e-12)


class TestPerAssetContributions:
    """US1 scenario 2: individual asset contributions alongside aggregates."""

    def test_contributions_cover_all_symbols_and_dates(
        self, portfolio_data, backtest_result
    ):
        contrib = backtest_result.asset_contributions
        assert list(contrib.columns) == portfolio_data.symbols
        assert len(contrib) == portfolio_data.n_dates - 1

    def test_contributions_sum_to_portfolio_returns(self, backtest_result):
        """Per-asset P&L contributions must aggregate to portfolio returns."""
        contrib_sum = backtest_result.asset_contributions.sum(axis=1)
        assert np.allclose(
            contrib_sum.values, backtest_result.returns.values, atol=1e-10
        )

    def test_unheld_assets_contribute_nothing(self, backtest_result):
        """At any date, at most 10 of 50 assets (top quintile) contribute."""
        contrib = backtest_result.asset_contributions
        late_rows = contrib.iloc[LOOKBACK + 30 :]
        nonzero_per_day = (late_rows.abs() > 1e-15).sum(axis=1)
        assert (nonzero_per_day <= 10).all()

    def test_asset_returns_match_price_data(self, portfolio_data, backtest_result):
        expected = portfolio_data.returns()  # (n_symbols, n_dates-1)
        actual = backtest_result.asset_returns.values.T
        assert np.allclose(actual, expected, atol=1e-12)


class TestPortfolioWeights:
    """Weight tracking through the momentum backtest."""

    def test_weights_history_shape(self, portfolio_data, backtest_result):
        weights = backtest_result.weights_history
        assert weights.shape == (portfolio_data.n_dates, portfolio_data.n_symbols)

    def test_top_quintile_holds_ten_assets_after_warmup(self, backtest_result):
        """Top 20% of 50 symbols -> exactly 10 held after a rebalance."""
        # Pick a rebalance date comfortably inside the valid signal window
        valid_rebalances = [
            d
            for d in backtest_result.rebalance_dates
            if pd.Timestamp(d) > backtest_result.weights_history.index[LOOKBACK + 5]
            and pd.Timestamp(d) < backtest_result.weights_history.index[-SKIP_RECENT - 5]
        ]
        assert valid_rebalances, "expected rebalances inside the valid signal window"

        row = backtest_result.weights_history.loc[pd.Timestamp(valid_rebalances[0])]
        assert (row > 1e-12).sum() == 10
        assert row.sum() == pytest.approx(1.0, abs=1e-9)

    def test_invested_weights_sum_to_one(self, backtest_result):
        """Once invested, weights stay fully allocated (sum to 1)."""
        weights = backtest_result.weights_history
        invested = weights.sum(axis=1) > 0.5
        assert invested.any()
        sums = weights.loc[invested].sum(axis=1)
        assert np.allclose(sums.values, 1.0, atol=1e-9)


class TestRebalancingBehavior:
    """Monthly rebalancing was actually applied."""

    def test_rebalances_recorded(self, backtest_result):
        assert len(backtest_result.rebalance_dates) > 0
        assert isinstance(backtest_result.turnover_history, pd.Series)

    def test_at_most_one_rebalance_per_month(self, backtest_result):
        months = [(d.year, d.month) for d in backtest_result.rebalance_dates]
        assert len(months) == len(set(months))
