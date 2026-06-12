"""
Contract Tests for Portfolio Metrics

These tests define the expected API contract for portfolio analytics.
"""

import numpy as np
import pandas as pd
import pytest

from vectorforge.portfolio.metrics import (
    ConcentrationMetrics,
    DiversificationMetrics,
    PortfolioMetrics,
    SectorExposure,
)


class TestPortfolioMetricsConcentration:
    """Contract tests for concentration metrics."""

    def test_herfindahl_index_single_asset(self):
        """HHI should be 1.0 for single asset portfolio."""
        metrics = self._make_single_asset_metrics()

        hhi = metrics.herfindahl_index()

        # All weight in one asset -> HHI = 1
        assert hhi.iloc[-1] == pytest.approx(1.0, abs=0.01)

    def test_herfindahl_index_equal_weight(self):
        """HHI should be 1/n for equal-weight portfolio."""
        n_assets = 5
        metrics = self._make_equal_weight_metrics(n_assets=n_assets)

        hhi = metrics.herfindahl_index()

        # Equal weight -> HHI = 1/n
        expected = 1.0 / n_assets
        assert hhi.iloc[-1] == pytest.approx(expected, abs=0.01)

    def test_concentration_metrics_returns_dataclass(self):
        """concentration_metrics should return ConcentrationMetrics."""
        metrics = self._make_equal_weight_metrics()

        result = metrics.concentration_metrics()

        assert isinstance(result, ConcentrationMetrics)
        assert hasattr(result, "herfindahl_index")
        assert hasattr(result, "top_1_weight")
        assert hasattr(result, "top_5_weight")
        assert hasattr(result, "effective_n")

    def test_effective_n_equals_actual_n_for_equal_weight(self):
        """Effective N should equal actual N for equal-weight portfolio."""
        n_assets = 4
        metrics = self._make_equal_weight_metrics(n_assets=n_assets)

        result = metrics.concentration_metrics()

        assert result.effective_n == pytest.approx(n_assets, abs=0.5)

    def test_top_n_concentration(self):
        """top_n_concentration should return correct top-N weights."""
        metrics = self._make_unequal_weight_metrics()

        top_3 = metrics.top_n_concentration(n=3)

        # Should be Series indexed by date
        assert isinstance(top_3, pd.Series)
        assert len(top_3) > 0

    @staticmethod
    def _make_single_asset_metrics() -> PortfolioMetrics:
        dates = pd.date_range("2023-01-01", periods=10, freq="B")
        returns = pd.Series([0.01] * 10, index=dates)
        weights = pd.DataFrame({"A": [1.0] * 10}, index=dates)
        asset_returns = pd.DataFrame({"A": [0.01] * 10}, index=dates)
        return PortfolioMetrics(returns, weights, asset_returns)

    @staticmethod
    def _make_equal_weight_metrics(n_assets: int = 5) -> PortfolioMetrics:
        dates = pd.date_range("2023-01-01", periods=10, freq="B")
        returns = pd.Series([0.01] * 10, index=dates)
        weight = 1.0 / n_assets
        weights = pd.DataFrame(
            {f"A{i}": [weight] * 10 for i in range(n_assets)},
            index=dates,
        )
        asset_returns = pd.DataFrame(
            {f"A{i}": [0.01] * 10 for i in range(n_assets)},
            index=dates,
        )
        return PortfolioMetrics(returns, weights, asset_returns)

    @staticmethod
    def _make_unequal_weight_metrics() -> PortfolioMetrics:
        dates = pd.date_range("2023-01-01", periods=10, freq="B")
        returns = pd.Series([0.01] * 10, index=dates)
        weights = pd.DataFrame(
            {
                "A": [0.4] * 10,
                "B": [0.3] * 10,
                "C": [0.2] * 10,
                "D": [0.1] * 10,
            },
            index=dates,
        )
        asset_returns = pd.DataFrame(
            {
                "A": [0.01] * 10,
                "B": [0.01] * 10,
                "C": [0.01] * 10,
                "D": [0.01] * 10,
            },
            index=dates,
        )
        return PortfolioMetrics(returns, weights, asset_returns)


class TestPortfolioMetricsDiversification:
    """Contract tests for diversification metrics."""

    def test_diversification_ratio_returns_float(self):
        """diversification_ratio should return a float."""
        metrics = self._make_metrics()

        dr = metrics.diversification_ratio()

        assert isinstance(dr, float)
        assert np.isfinite(dr)

    def test_diversification_metrics_returns_dataclass(self):
        """diversification_metrics should return DiversificationMetrics."""
        metrics = self._make_metrics()

        result = metrics.diversification_metrics()

        assert isinstance(result, DiversificationMetrics)
        assert hasattr(result, "diversification_ratio")
        assert hasattr(result, "avg_correlation")
        assert hasattr(result, "max_correlation")
        assert hasattr(result, "min_correlation")

    def test_correlation_matrix_shape(self):
        """correlation_matrix should return n x n matrix."""
        metrics = self._make_metrics()

        corr = metrics.correlation_matrix()

        assert isinstance(corr, pd.DataFrame)
        n = len(metrics.symbols)
        assert corr.shape == (n, n)

    def test_correlation_matrix_diagonal_is_one(self):
        """Diagonal of correlation matrix should be 1."""
        metrics = self._make_metrics()

        corr = metrics.correlation_matrix()

        for i in range(len(corr)):
            assert corr.iloc[i, i] == pytest.approx(1.0, abs=0.01)

    def test_rolling_correlation_returns_series(self):
        """rolling_correlation should return Series."""
        metrics = self._make_long_metrics()

        rolling_corr = metrics.rolling_correlation(window=20)

        assert isinstance(rolling_corr, pd.Series)

    @staticmethod
    def _make_metrics() -> PortfolioMetrics:
        dates = pd.date_range("2023-01-01", periods=30, freq="B")
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0.001, 0.01, 30), index=dates)
        weights = pd.DataFrame(
            {
                "A": [0.5] * 30,
                "B": [0.3] * 30,
                "C": [0.2] * 30,
            },
            index=dates,
        )
        asset_returns = pd.DataFrame(
            {
                "A": np.random.normal(0.001, 0.015, 30),
                "B": np.random.normal(0.0008, 0.012, 30),
                "C": np.random.normal(0.0005, 0.01, 30),
            },
            index=dates,
        )
        return PortfolioMetrics(returns, weights, asset_returns)

    @staticmethod
    def _make_long_metrics() -> PortfolioMetrics:
        dates = pd.date_range("2023-01-01", periods=60, freq="B")
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0.001, 0.01, 60), index=dates)
        weights = pd.DataFrame(
            {
                "A": [0.5] * 60,
                "B": [0.5] * 60,
            },
            index=dates,
        )
        asset_returns = pd.DataFrame(
            {
                "A": np.random.normal(0.001, 0.015, 60),
                "B": np.random.normal(0.0008, 0.012, 60),
            },
            index=dates,
        )
        return PortfolioMetrics(returns, weights, asset_returns)


class TestPortfolioMetricsSectorExposure:
    """Contract tests for sector exposure."""

    def test_sector_exposure_returns_dataframe(self):
        """sector_exposure should return DataFrame with sectors as columns."""
        metrics = self._make_metrics_with_sectors()

        exposure = metrics.sector_exposure()

        assert isinstance(exposure, pd.DataFrame)
        assert "Technology" in exposure.columns
        assert "Financials" in exposure.columns

    def test_sector_exposure_sums_to_one(self):
        """Sector weights should sum to 1 (portfolio weight)."""
        metrics = self._make_metrics_with_sectors()

        exposure = metrics.sector_exposure()

        # Each row should sum to approximately 1
        row_sums = exposure.sum(axis=1)
        assert row_sums.iloc[-1] == pytest.approx(1.0, abs=0.01)

    def test_sector_exposure_at_returns_dataclass(self):
        """sector_exposure_at should return SectorExposure."""
        metrics = self._make_metrics_with_sectors()
        date = metrics.dates[5]

        exposure = metrics.sector_exposure_at(date)

        assert isinstance(exposure, SectorExposure)
        assert hasattr(exposure, "weights")
        assert hasattr(exposure, "date")

    def test_sector_exposure_at_top_sector(self):
        """SectorExposure.top_sector should return largest sector."""
        metrics = self._make_metrics_with_sectors()
        date = metrics.dates[5]

        exposure = metrics.sector_exposure_at(date)
        top_sector, top_weight = exposure.top_sector()

        assert isinstance(top_sector, str)
        assert isinstance(top_weight, float)
        assert top_weight > 0

    @staticmethod
    def _make_metrics_with_sectors() -> PortfolioMetrics:
        dates = pd.date_range("2023-01-01", periods=10, freq="B")
        returns = pd.Series([0.01] * 10, index=dates)
        weights = pd.DataFrame(
            {
                "AAPL": [0.3] * 10,
                "GOOG": [0.2] * 10,
                "JPM": [0.3] * 10,
                "GS": [0.2] * 10,
            },
            index=dates,
        )
        asset_returns = pd.DataFrame(
            {
                "AAPL": [0.01] * 10,
                "GOOG": [0.01] * 10,
                "JPM": [0.01] * 10,
                "GS": [0.01] * 10,
            },
            index=dates,
        )
        sector_map = {
            "AAPL": "Technology",
            "GOOG": "Technology",
            "JPM": "Financials",
            "GS": "Financials",
        }
        return PortfolioMetrics(returns, weights, asset_returns, sector_map)


class TestPortfolioMetricsPerformance:
    """Contract tests for performance metrics."""

    def test_sharpe_ratio(self):
        """sharpe_ratio should return annualized Sharpe."""
        metrics = self._make_metrics()

        sharpe = metrics.sharpe_ratio()

        assert isinstance(sharpe, float)
        assert np.isfinite(sharpe)

    def test_sortino_ratio(self):
        """sortino_ratio should return Sortino ratio."""
        metrics = self._make_metrics()

        sortino = metrics.sortino_ratio()

        assert isinstance(sortino, float)
        assert np.isfinite(sortino)

    def test_max_drawdown(self):
        """max_drawdown should return negative number."""
        metrics = self._make_metrics()

        mdd = metrics.max_drawdown()

        assert isinstance(mdd, float)
        assert mdd <= 0

    def test_volatility(self):
        """volatility should return annualized volatility."""
        metrics = self._make_metrics()

        vol = metrics.volatility()

        assert isinstance(vol, float)
        assert vol >= 0

    @staticmethod
    def _make_metrics() -> PortfolioMetrics:
        dates = pd.date_range("2023-01-01", periods=252, freq="B")
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0.0005, 0.01, 252), index=dates)
        weights = pd.DataFrame({"A": [1.0] * 252}, index=dates)
        asset_returns = pd.DataFrame({"A": returns.values}, index=dates)
        return PortfolioMetrics(returns, weights, asset_returns)


class TestPortfolioMetricsAttribution:
    """Contract tests for return attribution."""

    def test_contribution_to_return(self):
        """contribution_to_return should return DataFrame."""
        metrics = self._make_metrics()

        contrib = metrics.contribution_to_return()

        assert isinstance(contrib, pd.DataFrame)
        assert set(contrib.columns) == set(metrics.symbols)

    def test_return_attribution(self):
        """return_attribution should return list of ReturnAttribution."""
        metrics = self._make_metrics()

        attributions = metrics.return_attribution()

        assert isinstance(attributions, list)
        if len(attributions) > 0:
            assert hasattr(attributions[0], "symbol")
            assert hasattr(attributions[0], "weight")
            assert hasattr(attributions[0], "contribution")

    def test_contribution_to_risk(self):
        """contribution_to_risk should return DataFrame."""
        metrics = self._make_metrics()

        risk_contrib = metrics.contribution_to_risk()

        assert isinstance(risk_contrib, pd.DataFrame)

    @staticmethod
    def _make_metrics() -> PortfolioMetrics:
        dates = pd.date_range("2023-01-01", periods=60, freq="B")
        np.random.seed(42)
        weights = pd.DataFrame(
            {
                "A": [0.6] * 60,
                "B": [0.4] * 60,
            },
            index=dates,
        )
        asset_returns = pd.DataFrame(
            {
                "A": np.random.normal(0.001, 0.015, 60),
                "B": np.random.normal(0.0005, 0.01, 60),
            },
            index=dates,
        )
        # Portfolio return = weighted asset returns
        returns = (weights.shift(1).bfill() * asset_returns).sum(axis=1)
        return PortfolioMetrics(returns, weights, asset_returns)


class TestPortfolioMetricsReport:
    """Contract tests for report generation."""

    def test_generate_report_returns_dict(self):
        """generate_report should return comprehensive dict."""
        metrics = self._make_metrics()

        report = metrics.generate_report()

        assert isinstance(report, dict)
        assert "performance" in report
        assert "concentration" in report
        assert "diversification" in report
        assert "turnover" in report

    def test_to_dataframe(self):
        """to_dataframe should convert report to DataFrame."""
        metrics = self._make_metrics()

        df = metrics.to_dataframe()

        assert isinstance(df, pd.DataFrame)
        assert "category" in df.columns
        assert "metric" in df.columns
        assert "value" in df.columns

    @staticmethod
    def _make_metrics() -> PortfolioMetrics:
        dates = pd.date_range("2023-01-01", periods=30, freq="B")
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0.001, 0.01, 30), index=dates)
        weights = pd.DataFrame(
            {
                "A": [0.5] * 30,
                "B": [0.3] * 30,
                "C": [0.2] * 30,
            },
            index=dates,
        )
        asset_returns = pd.DataFrame(
            {
                "A": np.random.normal(0.001, 0.015, 30),
                "B": np.random.normal(0.0008, 0.012, 30),
                "C": np.random.normal(0.0005, 0.01, 30),
            },
            index=dates,
        )
        return PortfolioMetrics(returns, weights, asset_returns)
