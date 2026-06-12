"""
Unit tests for cross-sectional signals (T035, US2).

Covers RankMethod, SignalResult filtering, CrossSectionalSignal factories
(momentum, mean reversion, volatility, relative strength, custom),
sector-neutral ranking, and weight conversion
(FR-007, FR-008, FR-009, FR-010, FR-011).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vectorforge.portfolio.data import PortfolioData, SymbolMetadata
from vectorforge.portfolio.signals import (
    CrossSectionalSignal,
    MomentumSignal,
    RankMethod,
    SignalResult,
    TargetWeights,
)


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


N_DAYS = 60
DATES = pd.date_range("2023-01-02", periods=N_DAYS, freq="B")
# Distinct constant daily growth rates: S0 worst ... S3 best
GROWTH_RATES = [0.000, 0.002, 0.004, 0.006]


@pytest.fixture
def growth_data() -> PortfolioData:
    data = {
        f"S{i}": make_ohlcv_df(100 * np.cumprod(np.full(N_DAYS, 1 + g)), DATES)
        for i, g in enumerate(GROWTH_RATES)
    }
    return PortfolioData.from_dict(data)


@pytest.fixture
def growth_data_with_sectors() -> PortfolioData:
    data = {
        f"S{i}": make_ohlcv_df(100 * np.cumprod(np.full(N_DAYS, 1 + g)), DATES)
        for i, g in enumerate(GROWTH_RATES)
    }
    metadata = {
        "S0": SymbolMetadata("S0", sector="Alpha"),
        "S1": SymbolMetadata("S1", sector="Alpha"),
        "S2": SymbolMetadata("S2", sector="Beta"),
        "S3": SymbolMetadata("S3", sector="Beta"),
    }
    return PortfolioData.from_dict(data, metadata=metadata)


# A column index where momentum(lookback=20, skip_recent=5) is valid
VALID_COL = 30


class TestRankMethod:
    def test_enum_values(self):
        assert RankMethod.PERCENTILE.value == "percentile"
        assert RankMethod.FRACTIONAL.value == "fractional"
        assert RankMethod.ORDINAL.value == "ordinal"


class TestMomentumSignal:
    """Momentum ranking identifies relative winners (FR-007, US2 scenario 1)."""

    def test_momentum_factory_configuration(self):
        signal = CrossSectionalSignal.momentum(lookback=126, skip_recent=10)
        assert isinstance(signal, MomentumSignal)
        assert signal.lookback == 126
        assert signal.skip_recent == 10
        assert signal.rank_method == RankMethod.PERCENTILE

    def test_percentile_ranks_span_0_to_100(self, growth_data):
        signal = CrossSectionalSignal.momentum(lookback=20, skip_recent=5)
        result = signal.generate(growth_data)

        ranks = result.ranks[:, VALID_COL]
        # 4 assets with strictly increasing momentum -> evenly spaced percentiles
        assert ranks == pytest.approx([0.0, 100 / 3, 200 / 3, 100.0])

    def test_momentum_values_match_price_change(self, growth_data):
        lookback, skip = 20, 5
        signal = CrossSectionalSignal.momentum(lookback=lookback, skip_recent=skip)
        result = signal.generate(growth_data)

        close = growth_data.close()
        j = VALID_COL
        expected = (close[:, j - skip] - close[:, j - lookback]) / close[:, j - lookback]
        assert result.values[:, j] == pytest.approx(expected, rel=1e-6)

    def test_signal_is_nan_during_warmup(self, growth_data):
        signal = CrossSectionalSignal.momentum(lookback=20, skip_recent=5)
        result = signal.generate(growth_data)

        # No signal before lookback periods have elapsed
        assert np.isnan(result.values[:, :20]).all()

    def test_fractional_rank_method(self, growth_data):
        signal = CrossSectionalSignal.momentum(
            lookback=20, skip_recent=5, rank_method=RankMethod.FRACTIONAL
        )
        result = signal.generate(growth_data)
        ranks = result.ranks[:, VALID_COL]
        assert ranks == pytest.approx([0.25, 0.5, 0.75, 1.0])

    def test_ordinal_rank_method(self, growth_data):
        signal = CrossSectionalSignal.momentum(
            lookback=20, skip_recent=5, rank_method=RankMethod.ORDINAL
        )
        result = signal.generate(growth_data)
        ranks = result.ranks[:, VALID_COL]
        assert ranks == pytest.approx([1.0, 2.0, 3.0, 4.0])

    def test_result_carries_symbols_and_dates(self, growth_data):
        signal = CrossSectionalSignal.momentum(lookback=20, skip_recent=5)
        result = signal.generate(growth_data)

        assert isinstance(result, SignalResult)
        assert result.symbols == growth_data.symbols
        assert result.dates.equals(growth_data.dates)
        assert result.values.shape == (growth_data.n_symbols, growth_data.n_dates)
        assert result.ranks.shape == result.values.shape


class TestSignalResultFiltering:
    """Rank-threshold filtering (FR-008, US2 scenario 2)."""

    def test_top_percentile_keeps_best_asset(self, growth_data):
        signal = CrossSectionalSignal.momentum(lookback=20, skip_recent=5)
        top = signal.generate(growth_data).top_percentile(25)

        col = top.values[:, VALID_COL]
        # Only S3 (rank 100) survives the top-25% filter on 4 assets
        assert np.isnan(col[:3]).all()
        assert not np.isnan(col[3])

    def test_top_percentile_50_keeps_two_assets(self, growth_data):
        signal = CrossSectionalSignal.momentum(lookback=20, skip_recent=5)
        top = signal.generate(growth_data).top_percentile(50)

        col = top.values[:, VALID_COL]
        assert np.isnan(col[:2]).all()
        assert np.isfinite(col[2:]).all()

    def test_bottom_percentile_keeps_worst_asset(self, growth_data):
        signal = CrossSectionalSignal.momentum(lookback=20, skip_recent=5)
        bottom = signal.generate(growth_data).bottom_percentile(25)

        col = bottom.values[:, VALID_COL]
        assert not np.isnan(col[0])
        assert np.isnan(col[1:]).all()

    def test_between_percentile(self, growth_data):
        signal = CrossSectionalSignal.momentum(lookback=20, skip_recent=5)
        mid = signal.generate(growth_data).between_percentile(30, 70)

        col = mid.values[:, VALID_COL]
        # Ranks are [0, 33.3, 66.7, 100] -> middle two survive
        assert np.isnan(col[0])
        assert np.isfinite(col[1])
        assert np.isfinite(col[2])
        assert np.isnan(col[3])

    def test_filtering_preserves_shape(self, growth_data):
        signal = CrossSectionalSignal.momentum(lookback=20, skip_recent=5)
        result = signal.generate(growth_data)
        top = result.top_percentile(50)

        assert top.values.shape == result.values.shape
        assert top.symbols == result.symbols


class TestSectorNeutralRanking:
    """Group-neutral signal generation (FR-009, US2 scenario 3)."""

    def test_ranks_computed_within_sector(self, growth_data_with_sectors):
        signal = CrossSectionalSignal.momentum(lookback=20, skip_recent=5)
        result = signal.generate(growth_data_with_sectors, group_field="sector")

        ranks = result.ranks[:, VALID_COL]
        # Within Alpha: S0 loser (0), S1 winner (100)
        # Within Beta:  S2 loser (0), S3 winner (100)
        assert ranks == pytest.approx([0.0, 100.0, 0.0, 100.0])

    def test_single_stock_sector_gets_median_rank(self):
        data = {
            f"S{i}": make_ohlcv_df(100 * np.cumprod(np.full(N_DAYS, 1 + g)), DATES)
            for i, g in enumerate([0.001, 0.003, 0.005])
        }
        metadata = {
            "S0": SymbolMetadata("S0", sector="Solo"),
            "S1": SymbolMetadata("S1", sector="Pair"),
            "S2": SymbolMetadata("S2", sector="Pair"),
        }
        pdata = PortfolioData.from_dict(data, metadata=metadata)

        signal = CrossSectionalSignal.momentum(lookback=20, skip_recent=5)
        result = signal.generate(pdata, group_field="sector")

        ranks = result.ranks[:, VALID_COL]
        assert ranks[0] == pytest.approx(50.0)  # single stock in sector
        assert ranks[1] == pytest.approx(0.0)
        assert ranks[2] == pytest.approx(100.0)


class TestMeanReversionSignal:
    def test_recent_loser_ranks_highest(self):
        n = N_DAYS
        data = {
            "LOSER": make_ohlcv_df(100 * np.cumprod(np.full(n, 0.995)), DATES),
            "FLAT": make_ohlcv_df(np.full(n, 100.0), DATES),
            "WINNER": make_ohlcv_df(100 * np.cumprod(np.full(n, 1.005)), DATES),
        }
        pdata = PortfolioData.from_dict(data)

        signal = CrossSectionalSignal.mean_reversion(lookback=20)
        result = signal.generate(pdata)

        ranks = result.ranks[:, 30]
        assert ranks == pytest.approx([100.0, 50.0, 0.0])


class TestVolatilitySignal:
    def test_inverse_volatility_ranks_low_vol_highest(self):
        n = N_DAYS
        low_vol = 100 * np.cumprod(np.full(n, 1.001))
        high_vol = 100 * np.cumprod(1 + 0.02 * np.array([1, -1] * (n // 2)))
        pdata = PortfolioData.from_dict(
            {
                "LOWVOL": make_ohlcv_df(low_vol, DATES),
                "HIVOL": make_ohlcv_df(high_vol, DATES),
            }
        )

        signal = CrossSectionalSignal.volatility(lookback=10, inverse=True)
        result = signal.generate(pdata)

        ranks = result.ranks[:, 30]
        assert ranks[0] == pytest.approx(100.0)
        assert ranks[1] == pytest.approx(0.0)

    def test_non_inverse_ranks_high_vol_highest(self):
        n = N_DAYS
        low_vol = 100 * np.cumprod(np.full(n, 1.001))
        high_vol = 100 * np.cumprod(1 + 0.02 * np.array([1, -1] * (n // 2)))
        pdata = PortfolioData.from_dict(
            {
                "LOWVOL": make_ohlcv_df(low_vol, DATES),
                "HIVOL": make_ohlcv_df(high_vol, DATES),
            }
        )

        signal = CrossSectionalSignal.volatility(lookback=10, inverse=False)
        result = signal.generate(pdata)

        ranks = result.ranks[:, 30]
        assert ranks[0] == pytest.approx(0.0)
        assert ranks[1] == pytest.approx(100.0)


class TestRelativeStrengthSignal:
    """Relative strength vs universe average (FR-011)."""

    def test_signal_is_return_minus_universe_average(self, growth_data):
        lookback = 20
        signal = CrossSectionalSignal.relative_strength(lookback=lookback)
        result = signal.generate(growth_data)

        j = VALID_COL
        close = growth_data.close()
        rets = (close[:, j] - close[:, j - lookback]) / close[:, j - lookback]
        expected = rets - np.nanmean(rets)
        assert result.values[:, j] == pytest.approx(expected, rel=1e-5)
        # Average asset signal centers near zero
        assert np.nanmean(result.values[:, j]) == pytest.approx(0.0, abs=1e-9)

    def test_benchmark_relative_strength(self, growth_data):
        lookback = 20
        signal = CrossSectionalSignal.relative_strength(lookback=lookback, benchmark_idx=0)
        result = signal.generate(growth_data)

        j = VALID_COL
        # S0 measured against itself -> exactly zero
        assert result.values[0, j] == pytest.approx(0.0, abs=1e-9)
        # Higher-growth assets have positive relative strength
        assert (result.values[1:, j] > 0).all()


class TestCustomSignal:
    def test_custom_function_signal(self, growth_data):
        signal = CrossSectionalSignal.custom(
            func=lambda close: close, lookback=1, name="price_level"
        )
        result = signal.generate(growth_data)

        # At the last date, ranking by price level matches growth ordering
        ranks = result.ranks[:, -1]
        assert ranks == pytest.approx([0.0, 100 / 3, 200 / 3, 100.0])


class TestToWeights:
    """Conversion of signals to target weights (FR-010, US2)."""

    def test_equal_weights_sum_to_one(self, growth_data):
        signal = CrossSectionalSignal.momentum(lookback=20, skip_recent=5)
        weights = signal.generate(growth_data).top_percentile(50).to_weights("equal")

        assert isinstance(weights, TargetWeights)
        col = weights.weights[:, VALID_COL]
        assert col == pytest.approx([0.0, 0.0, 0.5, 0.5])
        assert col.sum() == pytest.approx(1.0)

    def test_equal_weights_zero_when_no_signal(self, growth_data):
        signal = CrossSectionalSignal.momentum(lookback=20, skip_recent=5)
        weights = signal.generate(growth_data).to_weights("equal")

        # Warmup columns have no active assets -> zero weights
        assert np.allclose(weights.weights[:, :20], 0.0)

    def test_signal_weighted_proportional_to_shifted_signal(self):
        values = np.array([[1.0], [2.0], [3.0]])
        ranks = np.array([[0.0], [50.0], [100.0]])
        result = SignalResult(
            values=values,
            ranks=ranks,
            symbols=["A", "B", "C"],
            dates=pd.DatetimeIndex([pd.Timestamp("2023-01-02")]),
        )
        weights = result.to_weights("signal_weighted")

        # Values shifted to [0, 1, 2] -> weights [0, 1/3, 2/3]
        assert weights.weights[:, 0] == pytest.approx([0.0, 1 / 3, 2 / 3])

    def test_rank_weighted(self):
        values = np.array([[1.0], [2.0], [3.0]])
        ranks = np.array([[0.0], [50.0], [100.0]])
        result = SignalResult(
            values=values,
            ranks=ranks,
            symbols=["A", "B", "C"],
            dates=pd.DatetimeIndex([pd.Timestamp("2023-01-02")]),
        )
        weights = result.to_weights("rank_weighted")

        assert weights.weights[:, 0] == pytest.approx([0.0, 1 / 3, 2 / 3])


class TestTargetWeights:
    def test_equal_weight_uniform(self):
        weights = TargetWeights.equal_weight(["A", "B", "C", "D"], DATES)
        assert weights.n_symbols == 4
        assert weights.n_dates == N_DAYS
        assert np.allclose(weights.weights, 0.25)

    def test_equal_weight_with_active_mask(self):
        mask = np.zeros((3, N_DAYS), dtype=bool)
        mask[0, :] = True
        mask[1, :] = True  # only A and B active
        weights = TargetWeights.equal_weight(["A", "B", "C"], DATES, active_mask=mask)

        assert np.allclose(weights.weights[0], 0.5)
        assert np.allclose(weights.weights[1], 0.5)
        assert np.allclose(weights.weights[2], 0.0)

    def test_market_cap_weight(self):
        """FR-010: capitalization-weighted portfolio construction."""
        data = {
            "BIG": make_ohlcv_df(np.full(N_DAYS, 100.0), DATES),
            "SMALL": make_ohlcv_df(np.full(N_DAYS, 50.0), DATES),
        }
        metadata = {
            "BIG": SymbolMetadata("BIG", market_cap=3e9),
            "SMALL": SymbolMetadata("SMALL", market_cap=1e9),
        }
        pdata = PortfolioData.from_dict(data, metadata=metadata)

        weights = TargetWeights.market_cap_weight(pdata, DATES)
        assert np.allclose(weights.weights[0], 0.75)
        assert np.allclose(weights.weights[1], 0.25)

    def test_get_weights_at(self):
        weights = TargetWeights.equal_weight(["A", "B"], DATES)
        w = weights.get_weights_at(DATES[10])
        assert w == {"A": pytest.approx(0.5), "B": pytest.approx(0.5)}

    def test_to_dataframe(self):
        weights = TargetWeights.equal_weight(["A", "B"], DATES)
        df = weights.to_dataframe()
        assert df.shape == (N_DAYS, 2)
        assert list(df.columns) == ["A", "B"]
        assert df.index.equals(DATES)
