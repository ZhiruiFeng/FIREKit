"""
Integration test: full metrics analysis on a completed backtest (T070, US4).

Runs an equal-weight backtest over a 10-symbol universe with sector
metadata, then exercises the complete PortfolioMetrics suite:
concentration (HHI, top-N), diversification ratio, correlations,
sector exposure, beta, and return attribution
(FR-017, FR-018, FR-019, FR-020, FR-021, FR-022).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tests.fixtures.portfolio_data import get_sample_metadata, get_sample_portfolio_data
from vectorforge import VectorizedBacktester
from vectorforge.engine.base import PortfolioBacktestResult
from vectorforge.portfolio.data import PortfolioData
from vectorforge.portfolio.metrics import (
    ConcentrationMetrics,
    DiversificationMetrics,
    PortfolioMetrics,
    ReturnAttribution,
)
from vectorforge.portfolio.signals import TargetWeights

N_SYMBOLS = 10
N_DAYS = 252


@pytest.fixture(scope="module")
def sector_map() -> dict[str, str]:
    return {symbol: meta["sector"] for symbol, meta in get_sample_metadata().items()}


@pytest.fixture(scope="module")
def backtest_result() -> PortfolioBacktestResult:
    raw = get_sample_portfolio_data(n_days=N_DAYS, seed=42)
    data = PortfolioData.from_dict(raw)
    weights = TargetWeights.equal_weight(data.symbols, data.dates)
    engine = VectorizedBacktester()
    return engine.run_portfolio(weights, data, initial_capital=1_000_000)


@pytest.fixture(scope="module")
def metrics(backtest_result, sector_map) -> PortfolioMetrics:
    return PortfolioMetrics.from_backtest_result(backtest_result, sector_map=sector_map)


class TestMetricsConstruction:
    def test_from_backtest_result_wires_universe(self, metrics, backtest_result):
        assert metrics.symbols == list(backtest_result.weights_history.columns)
        assert len(metrics.symbols) == N_SYMBOLS
        assert metrics.dates.equals(backtest_result.weights_history.index)

    def test_performance_metrics_match_engine(self, metrics, backtest_result):
        assert metrics.total_return() == pytest.approx(backtest_result.total_return, rel=1e-9)
        assert metrics.max_drawdown() == pytest.approx(backtest_result.max_drawdown, rel=1e-6)
        # Engine uses population std (ddof=0), metrics uses sample std (ddof=1)
        assert metrics.sharpe_ratio() == pytest.approx(backtest_result.sharpe_ratio, rel=0.01)


class TestConcentrationAnalysis:
    """FR-021 / US4: HHI and top-N holding percentages."""

    def test_hhi_of_equal_weight_portfolio_is_one_over_n(self, metrics):
        hhi = metrics.herfindahl_index()
        assert isinstance(hhi, pd.Series)
        assert len(hhi) == N_DAYS
        # Initial allocation is exactly equal weight
        assert hhi.iloc[0] == pytest.approx(1.0 / N_SYMBOLS, abs=1e-12)
        # Drift between rebalances keeps HHI near 1/N
        assert float((hhi - 1.0 / N_SYMBOLS).abs().max()) < 0.005

    def test_concentration_metrics_formulas(self, metrics):
        cm = metrics.concentration_metrics()
        assert isinstance(cm, ConcentrationMetrics)
        # avg weights ~ 0.1 each
        assert cm.herfindahl_index == pytest.approx(0.1, abs=0.005)
        assert cm.effective_n == pytest.approx(1.0 / cm.herfindahl_index, rel=1e-9)
        assert cm.effective_n == pytest.approx(N_SYMBOLS, abs=0.5)
        assert cm.top_1_weight == pytest.approx(0.1, abs=0.02)
        assert cm.top_5_weight == pytest.approx(0.5, abs=0.05)
        assert cm.top_10_weight == pytest.approx(1.0, abs=1e-6)
        # Internal consistency
        assert cm.top_1_weight <= cm.top_5_weight <= cm.top_10_weight

    def test_top_n_concentration_time_series(self, metrics):
        top5 = metrics.top_n_concentration(n=5)
        assert len(top5) == N_DAYS
        assert ((top5 > 0.4) & (top5 < 0.7)).all()
        top10 = metrics.top_n_concentration(n=10)
        assert np.allclose(top10.values, 1.0, atol=1e-6)


class TestDiversificationAnalysis:
    """FR-020 / US4 scenario 1: diversification ratio and correlations."""

    def test_diversification_ratio_above_one(self, metrics):
        """Imperfectly correlated assets always yield DR > 1."""
        dr = metrics.diversification_ratio()
        assert dr > 1.0
        assert np.isfinite(dr)

    def test_diversification_metrics_bundle(self, metrics):
        dm = metrics.diversification_metrics()
        assert isinstance(dm, DiversificationMetrics)
        assert dm.diversification_ratio == pytest.approx(
            metrics.diversification_ratio(), rel=1e-12
        )
        assert -1.0 <= dm.min_correlation <= dm.avg_correlation <= dm.max_correlation <= 1.0

    def test_correlation_matrix_full_universe(self, metrics):
        corr = metrics.correlation_matrix()
        assert corr.shape == (N_SYMBOLS, N_SYMBOLS)
        assert np.allclose(np.diag(corr.values), 1.0)
        assert np.allclose(corr.values, corr.values.T, atol=1e-12)

    def test_rolling_correlation_analysis(self, metrics):
        """US4 scenario 3: rolling pairwise correlation over a lookback."""
        rolling = metrics.rolling_correlation(window=60)
        assert isinstance(rolling, pd.Series)
        assert len(rolling) == N_DAYS - 1 - 60 + 1  # asset returns have N-1 rows
        assert (rolling.abs() <= 1.0 + 1e-9).all()


class TestSectorExposureAnalysis:
    """FR-019 / US4 scenario 2: time series of sector allocations."""

    def test_sector_exposure_time_series(self, metrics, sector_map):
        exposure = metrics.sector_exposure()
        assert not exposure.empty
        assert set(exposure.columns) == set(sector_map.values())
        assert len(exposure) == N_DAYS
        # Fully invested: sector weights sum to total portfolio weight
        assert np.allclose(exposure.sum(axis=1).values, 1.0, atol=1e-9)

    def test_sector_weights_match_constituent_counts(self, metrics, sector_map):
        """Equal weight portfolio: sector weight ~ n_constituents / 10."""
        exposure = metrics.sector_exposure()
        counts: dict[str, int] = {}
        for sector in sector_map.values():
            counts[sector] = counts.get(sector, 0) + 1

        for sector, count in counts.items():
            expected = count / N_SYMBOLS
            assert exposure[sector].iloc[0] == pytest.approx(expected, abs=1e-9)
            assert float((exposure[sector] - expected).abs().max()) < 0.03

    def test_sector_exposure_at_date(self, metrics, backtest_result):
        first_date = backtest_result.weights_history.index[0]
        snap = metrics.sector_exposure_at(first_date)
        # Sample metadata has 5 Technology constituents -> largest sector
        sector, weight = snap.top_sector()
        assert sector == "Technology"
        assert weight == pytest.approx(0.5, abs=1e-9)


class TestBetaAnalysis:
    """FR-020: portfolio-level beta."""

    def test_beta_against_own_returns_is_one(self, metrics, backtest_result):
        assert metrics.portfolio_beta(backtest_result.returns) == pytest.approx(1.0, rel=1e-9)

    def test_beta_against_scaled_benchmark(self, metrics, backtest_result):
        benchmark = backtest_result.returns * 2.0
        assert metrics.portfolio_beta(benchmark) == pytest.approx(0.5, rel=1e-9)

    def test_rolling_beta_series(self, metrics, backtest_result):
        rolling = metrics.rolling_beta(backtest_result.returns, window=60)
        assert len(rolling) > 0
        assert np.allclose(rolling.values, 1.0, atol=1e-9)


class TestAttributionAnalysis:
    """FR-022: per-asset contribution to return and risk."""

    def test_contribution_to_return_structure(self, metrics):
        contrib = metrics.contribution_to_return()
        assert set(contrib.columns) == set(metrics.symbols)
        assert len(contrib) == N_DAYS - 2  # shift(1) drops one row of N-1 returns

    def test_return_attribution_complete_and_sorted(self, metrics):
        attributions = metrics.return_attribution()
        assert len(attributions) == N_SYMBOLS
        assert all(isinstance(a, ReturnAttribution) for a in attributions)

        contributions = [a.contribution for a in attributions]
        assert contributions == sorted(contributions, reverse=True)

        # Equal-weight portfolio: every asset's average weight ~ 0.1
        for a in attributions:
            assert a.weight == pytest.approx(0.1, abs=0.02)

        total = metrics.contribution_to_return().sum().sum()
        assert sum(contributions) == pytest.approx(total, rel=1e-9)

    def test_contribution_to_risk(self, metrics):
        risk = metrics.contribution_to_risk()
        assert set(risk.columns) == set(metrics.symbols)
        assert (risk.values >= 0).all()


class TestReportGeneration:
    def test_full_report(self, metrics, backtest_result):
        report = metrics.generate_report()
        assert set(report.keys()) == {
            "performance",
            "concentration",
            "diversification",
            "turnover",
        }
        assert report["performance"]["total_return"] == pytest.approx(
            backtest_result.total_return, rel=1e-9
        )
        assert report["concentration"]["effective_n"] == pytest.approx(10.0, abs=0.5)
        assert report["diversification"]["diversification_ratio"] > 1.0

    def test_report_dataframe(self, metrics):
        df = metrics.to_dataframe()
        assert set(df.columns) == {"category", "metric", "value"}
        assert len(df) >= 15
