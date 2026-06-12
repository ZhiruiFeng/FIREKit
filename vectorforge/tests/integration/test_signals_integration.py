"""
Integration test: momentum signal -> top decile -> equal weight (T036, US2).

Pipeline per US2 acceptance scenarios: generate a cross-sectional momentum
ranking over a 20-asset universe, filter to the top decile, convert to
equal weights, and run a portfolio backtest
(FR-007, FR-008, FR-010).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vectorforge import VectorizedBacktester
from vectorforge.portfolio.data import PortfolioData
from vectorforge.portfolio.signals import CrossSectionalSignal, TargetWeights

N_SYMBOLS = 20
N_DAYS = 160
LOOKBACK = 60
SKIP_RECENT = 5

# Strictly increasing daily growth rates: S00 worst ... S19 best
GROWTH_RATES = [0.0002 * i for i in range(N_SYMBOLS)]
SYMBOLS = [f"S{i:02d}" for i in range(N_SYMBOLS)]
DATES = pd.date_range("2023-01-02", periods=N_DAYS, freq="B")


def make_ohlcv_df(close: np.ndarray, dates: pd.DatetimeIndex) -> pd.DataFrame:
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


@pytest.fixture(scope="module")
def portfolio_data() -> PortfolioData:
    data = {
        sym: make_ohlcv_df(100 * np.cumprod(np.full(N_DAYS, 1 + g)), DATES)
        for sym, g in zip(SYMBOLS, GROWTH_RATES)
    }
    return PortfolioData.from_dict(data)


@pytest.fixture(scope="module")
def signal_result(portfolio_data):
    signal = CrossSectionalSignal.momentum(lookback=LOOKBACK, skip_recent=SKIP_RECENT)
    return signal.generate(portfolio_data)


@pytest.fixture(scope="module")
def top_decile_weights(signal_result) -> TargetWeights:
    return signal_result.top_percentile(10).to_weights("equal")


def valid_columns() -> range:
    """Date columns where the momentum signal is defined."""
    return range(LOOKBACK, N_DAYS - SKIP_RECENT)


class TestMomentumRanking:
    """US2 scenario 1: each stock receives a 0-100 percentile rank."""

    def test_ranks_are_percentiles(self, signal_result):
        for j in [LOOKBACK, LOOKBACK + 20, N_DAYS - SKIP_RECENT - 1]:
            ranks = signal_result.ranks[:, j]
            assert np.nanmin(ranks) == pytest.approx(0.0)
            assert np.nanmax(ranks) == pytest.approx(100.0)
            assert ((ranks >= 0) & (ranks <= 100)).all()

    def test_ranking_matches_growth_order(self, signal_result):
        """Higher constant growth -> strictly higher momentum rank."""
        j = LOOKBACK + 10
        ranks = signal_result.ranks[:, j]
        assert (np.diff(ranks) > 0).all()
        expected = np.arange(N_SYMBOLS) / (N_SYMBOLS - 1) * 100
        assert ranks == pytest.approx(expected)


class TestTopDecileSelection:
    """US2 scenario 2: requesting top 10% returns only the highest decile."""

    def test_top_decile_selects_two_of_twenty(self, signal_result):
        top = signal_result.top_percentile(10)
        for j in valid_columns():
            active = np.isfinite(top.values[:, j])
            assert active.sum() == 2, f"expected 2 active assets at column {j}"

    def test_top_decile_selects_highest_growth_symbols(self, signal_result):
        top = signal_result.top_percentile(10)
        for j in [LOOKBACK, LOOKBACK + 25, N_DAYS - SKIP_RECENT - 1]:
            active_symbols = {
                SYMBOLS[i] for i in range(N_SYMBOLS) if np.isfinite(top.values[i, j])
            }
            assert active_symbols == {"S18", "S19"}


class TestEqualWeightConversion:
    """Equal-weight portfolio from the filtered signal."""

    def test_selected_assets_get_half_weight_each(self, top_decile_weights):
        for j in valid_columns():
            col = top_decile_weights.weights[:, j]
            assert col[18] == pytest.approx(0.5)
            assert col[19] == pytest.approx(0.5)
            assert col[:18].sum() == pytest.approx(0.0)
            assert col.sum() == pytest.approx(1.0)

    def test_warmup_columns_have_zero_weight(self, top_decile_weights):
        assert np.allclose(top_decile_weights.weights[:, :LOOKBACK], 0.0)


class TestEndToEndBacktest:
    """The signal pipeline plugs into the portfolio engine."""

    @pytest.fixture(scope="class")
    def result(self, portfolio_data, top_decile_weights):
        engine = VectorizedBacktester()
        return engine.run_portfolio(
            strategy=top_decile_weights,
            data=portfolio_data,
            initial_capital=100_000,
        )

    def test_backtest_completes_with_positive_return(self, result):
        # Holding the two fastest-growing assets must make money
        assert result.total_return > 0

    def test_portfolio_return_between_top_two_growth_rates(self, result):
        """Daily returns of the invested portfolio sit between the two
        constituents' growth rates (a 50/50 blend of S18 and S19)."""
        invested = result.returns[result.returns.abs() > 1e-12]
        assert len(invested) > 50
        low, high = GROWTH_RATES[18], GROWTH_RATES[19]
        assert (invested >= low - 1e-6).all()
        assert (invested <= high + 1e-6).all()
        # Blended return is approximately the midpoint
        assert invested.mean() == pytest.approx((low + high) / 2, rel=0.05)

    def test_only_top_decile_assets_contribute(self, result):
        contrib = result.asset_contributions
        idle = contrib.drop(columns=["S18", "S19"])
        assert np.allclose(idle.values, 0.0, atol=1e-15)
        assert contrib[["S18", "S19"]].abs().values.sum() > 0

    def test_equity_flat_during_warmup_then_grows(self, result):
        equity = result.equity_curve
        assert np.allclose(equity.iloc[: LOOKBACK - 1].values, 100_000, rtol=1e-12)
        assert equity.iloc[-1] > equity.iloc[LOOKBACK]
