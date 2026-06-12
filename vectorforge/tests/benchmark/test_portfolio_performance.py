"""
T085 — Portfolio backtest performance benchmarks.

Success criteria (specs/001-multi-asset-portfolio/spec.md):

- SC-001: 100-asset portfolio over 10 years of daily data (2520 business
  days) backtests in < 5 seconds via ``VectorizedBacktester.run_portfolio``
  with monthly calendar rebalancing.
- SC-002: Portfolio-level metrics (Sharpe, max drawdown, sector exposure)
  computed via ``PortfolioMetrics`` in < 1 second after backtest completion.

Methodology: plain wall-clock timing with ``time.perf_counter`` on a single
run (a small warm-up backtest first amortizes lazy imports inside
``run_portfolio``). Hardware caveat: shared cloud container, pure NumPy
backend — JAX and Numba are not installed.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from vectorforge import VectorizedBacktester
from vectorforge.portfolio.metrics import PortfolioMetrics
from vectorforge.portfolio.rebalancer import RebalanceFrequency, Rebalancer
from vectorforge.portfolio.signals import TargetWeights

from tests.benchmark.helpers import TEN_YEARS_DAYS, make_portfolio_data, make_sector_map

N_ASSETS = 100

SC001_TARGET_SECONDS = 5.0
SC002_TARGET_SECONDS = 1.0


@pytest.fixture(scope="module")
def engine() -> VectorizedBacktester:
    """Engine with a tiny warm-up run to amortize lazy module imports."""
    engine = VectorizedBacktester()
    small = make_portfolio_data(5, 80, seed=1)
    weights = TargetWeights.equal_weight(symbols=small.symbols, dates=small.dates)
    engine.run_portfolio(
        strategy=weights,
        data=small,
        initial_capital=100_000,
        rebalancer=Rebalancer.calendar(RebalanceFrequency.MONTHLY),
    )
    return engine


@pytest.fixture(scope="module")
def backtest_100(engine):
    """Run the SC-001 scenario once; share the result with the SC-002 test."""
    data = make_portfolio_data(N_ASSETS, TEN_YEARS_DAYS, seed=42)
    weights = TargetWeights.equal_weight(symbols=data.symbols, dates=data.dates)
    rebalancer = Rebalancer.calendar(RebalanceFrequency.MONTHLY)

    start = time.perf_counter()
    result = engine.run_portfolio(
        strategy=weights,
        data=data,
        initial_capital=1_000_000,
        rebalancer=rebalancer,
    )
    elapsed = time.perf_counter() - start
    return data, result, elapsed


class TestSC001BacktestSpeed:
    """SC-001: 100 assets x 10 years daily backtest < 5 s (vectorized)."""

    def test_sc001_100_asset_10_year_backtest_under_5_seconds(self, backtest_100):
        data, result, elapsed = backtest_100

        # Sanity: the backtest actually did the work we claim to measure.
        assert data.n_symbols == N_ASSETS
        assert data.n_dates == TEN_YEARS_DAYS
        # Monthly rebalancing over ~10 years -> roughly 120 rebalances.
        assert 100 <= len(result.rebalance_dates) <= 130
        assert len(result.equity_curve) == TEN_YEARS_DAYS
        assert np.isfinite(result.sharpe_ratio)
        assert np.isfinite(result.total_return)

        print(
            f"\nSC-001: {N_ASSETS} assets x {TEN_YEARS_DAYS} days, monthly rebalance: "
            f"{elapsed:.3f}s (target < {SC001_TARGET_SECONDS:.1f}s)"
        )
        assert elapsed < SC001_TARGET_SECONDS, (
            f"SC-001 FAIL: backtest took {elapsed:.3f}s, target < {SC001_TARGET_SECONDS}s"
        )


class TestSC002MetricsSpeed:
    """SC-002: Sharpe, drawdown, sector exposure computed < 1 s post-backtest."""

    def test_sc002_portfolio_metrics_under_1_second(self, backtest_100):
        data, result, _ = backtest_100
        sector_map = make_sector_map(data.symbols)

        start = time.perf_counter()
        metrics = PortfolioMetrics.from_backtest_result(result, sector_map=sector_map)
        sharpe = metrics.sharpe_ratio()
        drawdown = metrics.max_drawdown()
        sector_exposure = metrics.sector_exposure()
        elapsed = time.perf_counter() - start

        # Sanity checks on outputs.
        assert np.isfinite(sharpe)
        assert -1.0 <= drawdown <= 0.0
        assert sector_exposure.shape == (TEN_YEARS_DAYS, 10)
        # Equal-weight portfolio: sector weights must sum to ~1 each day.
        np.testing.assert_allclose(sector_exposure.sum(axis=1).values, 1.0, atol=1e-4)

        print(
            f"\nSC-002: Sharpe + max drawdown + sector exposure for {N_ASSETS} assets: "
            f"{elapsed:.4f}s (target < {SC002_TARGET_SECONDS:.1f}s)"
        )
        assert elapsed < SC002_TARGET_SECONDS, (
            f"SC-002 FAIL: metrics took {elapsed:.3f}s, target < {SC002_TARGET_SECONDS}s"
        )
