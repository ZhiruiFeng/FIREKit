"""
Unit tests for portfolio metrics (T069, US4).

Covers concentration (HHI, top-N), diversification ratio, correlation,
sector exposure, beta, attribution, and report generation
(FR-017, FR-018, FR-019, FR-020, FR-021, FR-022).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vectorforge.portfolio.metrics import (
    ConcentrationMetrics,
    DiversificationMetrics,
    PortfolioMetrics,
    ReturnAttribution,
    SectorExposure,
)

N = 120
DATES = pd.date_range("2023-01-02", periods=N, freq="B")


def equal_weight_metrics(n_assets: int, seed: int = 5) -> PortfolioMetrics:
    """Equal-weight portfolio over n random assets."""
    rng = np.random.default_rng(seed)
    asset_returns = pd.DataFrame(
        rng.normal(0.0005, 0.01, (N, n_assets)),
        index=DATES,
        columns=[f"S{i}" for i in range(n_assets)],
    )
    weights = pd.DataFrame(1.0 / n_assets, index=DATES, columns=asset_returns.columns)
    portfolio_returns = (asset_returns * (1.0 / n_assets)).sum(axis=1)
    return PortfolioMetrics(
        returns=portfolio_returns,
        weights_history=weights,
        asset_returns=asset_returns,
    )


def orthogonal_two_asset_metrics() -> PortfolioMetrics:
    """
    Two equal-volatility assets with exactly zero sample correlation.

    A = +1%, -1%, +1%, -1%, ...
    B = +1%, +1%, -1%, -1%, ...
    Sample means are 0, sample covariance is 0, sample stds are equal,
    so the 50/50 diversification ratio is exactly sqrt(2).
    """
    k = N // 4
    a = np.tile([0.01, -0.01], 2 * k)
    b = np.tile([0.01, 0.01, -0.01, -0.01], k)
    asset_returns = pd.DataFrame({"A": a, "B": b}, index=DATES)
    weights = pd.DataFrame(0.5, index=DATES, columns=["A", "B"])
    portfolio_returns = 0.5 * asset_returns["A"] + 0.5 * asset_returns["B"]
    return PortfolioMetrics(
        returns=portfolio_returns,
        weights_history=weights,
        asset_returns=asset_returns,
    )


class TestPerformanceMetrics:
    """Portfolio-level Sharpe/Sortino/drawdown from aggregate returns (FR-017)."""

    def test_total_return_compounds(self):
        returns = pd.Series(np.full(N, 0.01), index=DATES)
        m = PortfolioMetrics(
            returns=returns,
            weights_history=pd.DataFrame({"A": np.ones(N)}, index=DATES),
            asset_returns=pd.DataFrame({"A": returns}),
        )
        assert m.total_return() == pytest.approx(1.01**N - 1, rel=1e-9)

    def test_max_drawdown_known_value(self):
        # Up 10 days, one -10% day, then flat
        rets = np.zeros(N)
        rets[:10] = 0.01
        rets[10] = -0.10
        returns = pd.Series(rets, index=DATES)
        m = PortfolioMetrics(
            returns=returns,
            weights_history=pd.DataFrame({"A": np.ones(N)}, index=DATES),
            asset_returns=pd.DataFrame({"A": returns}),
        )
        assert m.max_drawdown() == pytest.approx(-0.10, rel=1e-9)

    def test_monotonic_returns_have_zero_drawdown_and_inf_sortino(self):
        returns = pd.Series(np.full(N, 0.005), index=DATES)
        m = PortfolioMetrics(
            returns=returns,
            weights_history=pd.DataFrame({"A": np.ones(N)}, index=DATES),
            asset_returns=pd.DataFrame({"A": returns}),
        )
        assert m.max_drawdown() == pytest.approx(0.0)
        assert m.sortino_ratio() == float("inf")

    def test_sharpe_ratio_formula(self):
        rng = np.random.default_rng(1)
        rets = pd.Series(rng.normal(0.001, 0.01, N), index=DATES)
        m = PortfolioMetrics(
            returns=rets,
            weights_history=pd.DataFrame({"A": np.ones(N)}, index=DATES),
            asset_returns=pd.DataFrame({"A": rets}),
        )
        expected = rets.mean() / rets.std() * np.sqrt(252)
        assert m.sharpe_ratio() == pytest.approx(expected, rel=1e-9)

    def test_volatility_annualizes_daily_std(self):
        rng = np.random.default_rng(2)
        rets = pd.Series(rng.normal(0, 0.02, N), index=DATES)
        m = PortfolioMetrics(
            returns=rets,
            weights_history=pd.DataFrame({"A": np.ones(N)}, index=DATES),
            asset_returns=pd.DataFrame({"A": rets}),
        )
        assert m.volatility() == pytest.approx(rets.std() * np.sqrt(252), rel=1e-9)


class TestConcentrationMetrics:
    """HHI and top-N concentration (FR-021, US4)."""

    @pytest.mark.parametrize("n_assets", [2, 5, 10, 50])
    def test_hhi_of_equal_weights_is_one_over_n(self, n_assets):
        m = equal_weight_metrics(n_assets)
        hhi = m.herfindahl_index()
        assert isinstance(hhi, pd.Series)
        assert np.allclose(hhi.values, 1.0 / n_assets, atol=1e-12)

    def test_hhi_of_single_asset_is_one(self):
        weights = pd.DataFrame({"A": np.ones(N)}, index=DATES)
        rets = pd.Series(np.zeros(N), index=DATES)
        m = PortfolioMetrics(
            returns=rets, weights_history=weights, asset_returns=pd.DataFrame({"A": rets})
        )
        assert np.allclose(m.herfindahl_index().values, 1.0)

    def test_concentration_metrics_for_known_weights(self):
        # Fixed, sorted weights: 0.4, 0.3, 0.2, 0.1
        w = [0.4, 0.3, 0.2, 0.1]
        weights = pd.DataFrame(
            {f"S{i}": np.full(N, wi) for i, wi in enumerate(w)}, index=DATES
        )
        rets = pd.Series(np.zeros(N), index=DATES)
        asset_returns = pd.DataFrame(0.0, index=DATES, columns=weights.columns)
        m = PortfolioMetrics(
            returns=rets, weights_history=weights, asset_returns=asset_returns
        )

        cm = m.concentration_metrics()
        assert isinstance(cm, ConcentrationMetrics)
        expected_hhi = sum(x**2 for x in w)  # 0.30
        assert cm.herfindahl_index == pytest.approx(expected_hhi, abs=1e-12)
        assert cm.top_1_weight == pytest.approx(0.4)
        assert cm.top_5_weight == pytest.approx(1.0)  # only 4 assets
        assert cm.top_10_weight == pytest.approx(1.0)
        assert cm.effective_n == pytest.approx(1.0 / expected_hhi)

    def test_effective_n_equals_n_for_equal_weights(self):
        m = equal_weight_metrics(10)
        cm = m.concentration_metrics()
        assert cm.effective_n == pytest.approx(10.0, rel=1e-9)
        assert cm.top_5_weight == pytest.approx(0.5, rel=1e-9)
        assert cm.top_10_weight == pytest.approx(1.0, rel=1e-9)

    def test_top_n_concentration_series(self):
        w = [0.4, 0.3, 0.2, 0.1]
        weights = pd.DataFrame(
            {f"S{i}": np.full(N, wi) for i, wi in enumerate(w)}, index=DATES
        )
        rets = pd.Series(np.zeros(N), index=DATES)
        m = PortfolioMetrics(
            returns=rets,
            weights_history=weights,
            asset_returns=pd.DataFrame(0.0, index=DATES, columns=weights.columns),
        )

        top2 = m.top_n_concentration(n=2)
        assert isinstance(top2, pd.Series)
        assert np.allclose(top2.values, 0.7, atol=1e-12)


class TestDiversificationMetrics:
    """Diversification ratio and correlations (FR-018, FR-020, US4 scenario 1)."""

    def test_diversification_ratio_uncorrelated_equal_vol_is_sqrt2(self):
        m = orthogonal_two_asset_metrics()
        assert m.diversification_ratio() == pytest.approx(np.sqrt(2), rel=1e-9)

    def test_diversification_ratio_single_asset_is_one(self):
        rets = pd.Series(np.tile([0.01, -0.01], N // 2), index=DATES)
        m = PortfolioMetrics(
            returns=rets,
            weights_history=pd.DataFrame({"A": np.ones(N)}, index=DATES),
            asset_returns=pd.DataFrame({"A": rets}),
        )
        assert m.diversification_ratio() == pytest.approx(1.0, rel=1e-9)

    def test_diversification_metrics_orthogonal_assets(self):
        m = orthogonal_two_asset_metrics()
        dm = m.diversification_metrics()
        assert isinstance(dm, DiversificationMetrics)
        assert dm.diversification_ratio == pytest.approx(np.sqrt(2), rel=1e-9)
        assert dm.avg_correlation == pytest.approx(0.0, abs=1e-12)

    def test_perfectly_negative_correlation(self):
        a = np.tile([0.01, -0.01], N // 2)
        asset_returns = pd.DataFrame({"A": a, "B": -a}, index=DATES)
        weights = pd.DataFrame(0.5, index=DATES, columns=["A", "B"])
        m = PortfolioMetrics(
            returns=pd.Series(np.zeros(N), index=DATES),
            weights_history=weights,
            asset_returns=asset_returns,
        )
        dm = m.diversification_metrics()
        assert dm.avg_correlation == pytest.approx(-1.0)
        assert dm.min_correlation == pytest.approx(-1.0)

    def test_correlation_matrix_properties(self):
        m = equal_weight_metrics(4)
        corr = m.correlation_matrix()
        assert corr.shape == (4, 4)
        assert np.allclose(np.diag(corr.values), 1.0)
        assert np.allclose(corr.values, corr.values.T, atol=1e-12)
        assert (corr.values <= 1.0 + 1e-12).all() and (corr.values >= -1.0 - 1e-12).all()

    def test_rolling_correlation_returns_series_in_range(self):
        """FR-018 / US4 scenario 3: rolling pairwise correlation."""
        m = equal_weight_metrics(3)
        rolling = m.rolling_correlation(window=30)
        assert isinstance(rolling, pd.Series)
        assert len(rolling) == N - 30 + 1
        assert (rolling.abs() <= 1.0 + 1e-9).all()


class TestSectorExposure:
    """Sector exposure time series (FR-019, US4 scenario 2)."""

    def make_metrics(self) -> PortfolioMetrics:
        weights = pd.DataFrame(
            {"AAPL": 0.4, "MSFT": 0.2, "JPM": 0.3, "GS": 0.1}, index=DATES
        )
        asset_returns = pd.DataFrame(0.0, index=DATES, columns=weights.columns)
        return PortfolioMetrics(
            returns=pd.Series(np.zeros(N), index=DATES),
            weights_history=weights,
            asset_returns=asset_returns,
            sector_map={
                "AAPL": "Technology",
                "MSFT": "Technology",
                "JPM": "Financials",
                "GS": "Financials",
            },
        )

    def test_sector_exposure_sums_symbol_weights(self):
        m = self.make_metrics()
        exposure = m.sector_exposure()

        assert set(exposure.columns) == {"Technology", "Financials"}
        assert np.allclose(exposure["Technology"].values, 0.6, atol=1e-12)
        assert np.allclose(exposure["Financials"].values, 0.4, atol=1e-12)
        # Time series over all dates
        assert exposure.index.equals(DATES)

    def test_sector_exposure_rows_sum_to_total_weight(self):
        m = self.make_metrics()
        exposure = m.sector_exposure()
        assert np.allclose(exposure.sum(axis=1).values, 1.0, atol=1e-12)

    def test_sector_exposure_empty_without_map(self):
        m = equal_weight_metrics(3)
        assert m.sector_exposure().empty

    def test_sector_exposure_at_date(self):
        m = self.make_metrics()
        snap = m.sector_exposure_at(DATES[10])
        assert isinstance(snap, SectorExposure)
        assert snap.weights["Technology"] == pytest.approx(0.6)
        assert snap.top_sector() == ("Technology", pytest.approx(0.6))
        assert isinstance(snap.to_series(), pd.Series)


class TestPortfolioBeta:
    """Portfolio beta vs benchmark (FR-020)."""

    def test_beta_of_scaled_benchmark(self):
        rng = np.random.default_rng(9)
        bench = pd.Series(rng.normal(0.0003, 0.01, N), index=DATES)
        port = 1.5 * bench
        m = PortfolioMetrics(
            returns=port,
            weights_history=pd.DataFrame({"A": np.ones(N)}, index=DATES),
            asset_returns=pd.DataFrame({"A": port}),
        )
        assert m.portfolio_beta(bench) == pytest.approx(1.5, rel=1e-9)

    def test_beta_of_self_is_one(self):
        rng = np.random.default_rng(10)
        rets = pd.Series(rng.normal(0.0003, 0.01, N), index=DATES)
        m = PortfolioMetrics(
            returns=rets,
            weights_history=pd.DataFrame({"A": np.ones(N)}, index=DATES),
            asset_returns=pd.DataFrame({"A": rets}),
        )
        assert m.portfolio_beta(rets) == pytest.approx(1.0, rel=1e-9)

    def test_rolling_beta_converges_to_constant_beta(self):
        rng = np.random.default_rng(12)
        bench = pd.Series(rng.normal(0.0003, 0.01, N), index=DATES)
        port = 0.8 * bench
        m = PortfolioMetrics(
            returns=port,
            weights_history=pd.DataFrame({"A": np.ones(N)}, index=DATES),
            asset_returns=pd.DataFrame({"A": port}),
        )
        rolling = m.rolling_beta(bench, window=30)
        assert len(rolling) > 0
        assert np.allclose(rolling.values, 0.8, atol=1e-9)


class TestAttribution:
    """Per-asset contribution to return and risk (FR-022)."""

    def make_metrics(self) -> PortfolioMetrics:
        a = np.full(N, 0.01)  # A returns +1%/day
        b = np.zeros(N)  # B flat
        asset_returns = pd.DataFrame({"A": a, "B": b}, index=DATES)
        weights = pd.DataFrame({"A": 0.6, "B": 0.4}, index=DATES)
        port = 0.6 * asset_returns["A"] + 0.4 * asset_returns["B"]
        return PortfolioMetrics(
            returns=port, weights_history=weights, asset_returns=asset_returns
        )

    def test_contribution_is_weight_times_return(self):
        m = self.make_metrics()
        contrib = m.contribution_to_return()

        # Constant weights -> contribution = w * r each day (first day dropped by shift)
        assert np.allclose(contrib["A"].values, 0.6 * 0.01, atol=1e-15)
        assert np.allclose(contrib["B"].values, 0.0, atol=1e-15)
        assert len(contrib) == N - 1

    def test_return_attribution_sorted_and_consistent(self):
        m = self.make_metrics()
        attributions = m.return_attribution()

        assert len(attributions) == 2
        assert all(isinstance(a, ReturnAttribution) for a in attributions)
        # Sorted descending by contribution: A first
        assert attributions[0].symbol == "A"
        assert attributions[0].contribution > attributions[1].contribution
        assert attributions[0].weight == pytest.approx(0.6)
        assert attributions[0].asset_return == pytest.approx(1.01**N - 1, rel=1e-9)
        assert attributions[1].contribution == pytest.approx(0.0, abs=1e-12)

        total_contrib = m.contribution_to_return().sum().sum()
        assert sum(a.contribution for a in attributions) == pytest.approx(
            total_contrib, rel=1e-9
        )

    def test_contribution_to_risk_weights_times_vol(self):
        m = self.make_metrics()
        risk = m.contribution_to_risk()
        asset_vols = m._asset_returns.std()
        assert np.allclose(risk["A"].values, 0.6 * asset_vols["A"], atol=1e-15)
        assert np.allclose(risk["B"].values, 0.4 * asset_vols["B"], atol=1e-15)


class TestTurnoverAnalysis:
    def test_constant_weights_have_zero_turnover(self):
        m = equal_weight_metrics(4)
        analysis = m.turnover_analysis()
        assert analysis["total_turnover"] == pytest.approx(0.0, abs=1e-12)
        assert analysis["max_turnover"] == pytest.approx(0.0, abs=1e-12)

    def test_single_weight_change_turnover(self):
        weights = pd.DataFrame({"A": 0.5, "B": 0.5}, index=DATES)
        weights.iloc[10:] = [0.7, 0.3]  # one 20% shift
        rets = pd.Series(np.zeros(N), index=DATES)
        m = PortfolioMetrics(
            returns=rets,
            weights_history=weights,
            asset_returns=pd.DataFrame(0.0, index=DATES, columns=["A", "B"]),
        )
        analysis = m.turnover_analysis()
        assert analysis["total_turnover"] == pytest.approx(0.2, abs=1e-12)
        assert analysis["max_turnover"] == pytest.approx(0.2, abs=1e-12)
        assert analysis["max_turnover_date"] == DATES[10]


class TestReporting:
    def test_generate_report_structure(self):
        m = equal_weight_metrics(5)
        report = m.generate_report()

        assert set(report.keys()) == {
            "performance",
            "concentration",
            "diversification",
            "turnover",
        }
        assert report["concentration"]["herfindahl_index"] == pytest.approx(0.2, abs=1e-12)
        assert report["concentration"]["effective_n"] == pytest.approx(5.0, rel=1e-9)
        assert "sharpe_ratio" in report["performance"]
        assert "diversification_ratio" in report["diversification"]

    def test_to_dataframe(self):
        m = equal_weight_metrics(5)
        df = m.to_dataframe()
        assert set(df.columns) == {"category", "metric", "value"}
        assert set(df["category"].unique()) == {
            "performance",
            "concentration",
            "diversification",
            "turnover",
        }


class TestFromBacktestResult:
    def test_factory_wires_result_fields(self):
        from vectorforge import VectorizedBacktester
        from vectorforge.portfolio.data import PortfolioData
        from vectorforge.portfolio.signals import TargetWeights

        dates = pd.date_range("2023-01-02", periods=80, freq="B")
        n = len(dates)
        rng = np.random.default_rng(21)

        def df(close):
            close = np.asarray(close, dtype=float)
            return pd.DataFrame(
                {
                    "open": close,
                    "high": close * 1.01,
                    "low": close * 0.99,
                    "close": close,
                    "volume": np.full(n, 1e6),
                },
                index=dates,
            )

        data = PortfolioData.from_dict(
            {
                "A": df(100 * np.exp(np.cumsum(rng.normal(0.0005, 0.01, n)))),
                "B": df(100 * np.exp(np.cumsum(rng.normal(0.0005, 0.01, n)))),
            }
        )
        weights = TargetWeights.equal_weight(data.symbols, data.dates)
        result = VectorizedBacktester().run_portfolio(weights, data, initial_capital=100_000)

        m = PortfolioMetrics.from_backtest_result(result, sector_map={"A": "X", "B": "Y"})
        assert m.symbols == ["A", "B"]
        assert m.total_return() == pytest.approx(result.total_return, rel=1e-9)
        assert not m.sector_exposure().empty
