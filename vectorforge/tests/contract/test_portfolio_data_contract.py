"""
Contract Tests for PortfolioData

These tests define the expected API contract for PortfolioData.
They must FAIL before implementation and PASS after.
"""


import numpy as np
import pandas as pd
import pytest

from vectorforge.portfolio.data import (
    MissingDataPolicy,
    PortfolioData,
    SymbolMetadata,
)

# ============================================================================
# T005: Contract test for PortfolioData basic construction
# ============================================================================


class TestPortfolioDataConstruction:
    """Contract tests for PortfolioData construction."""

    def test_from_dict_creates_portfolio(self):
        """PortfolioData.from_dict() should create instance from dict of DataFrames."""
        # Given: Dictionary of OHLCV DataFrames
        dates = pd.date_range("2023-01-01", periods=10, freq="B")
        aapl = pd.DataFrame(
            {
                "open": np.random.uniform(100, 110, 10),
                "high": np.random.uniform(110, 120, 10),
                "low": np.random.uniform(90, 100, 10),
                "close": np.random.uniform(100, 110, 10),
                "volume": np.random.uniform(1e6, 2e6, 10),
            },
            index=dates,
        )
        goog = pd.DataFrame(
            {
                "open": np.random.uniform(130, 140, 10),
                "high": np.random.uniform(140, 150, 10),
                "low": np.random.uniform(120, 130, 10),
                "close": np.random.uniform(130, 140, 10),
                "volume": np.random.uniform(5e5, 1e6, 10),
            },
            index=dates,
        )

        # When: Creating PortfolioData
        portfolio = PortfolioData.from_dict({"AAPL": aapl, "GOOG": goog})

        # Then: Should have correct properties
        assert portfolio.n_symbols == 2
        assert portfolio.n_dates == 10
        assert "AAPL" in portfolio.symbols
        assert "GOOG" in portfolio.symbols

    def test_symbols_property_returns_list(self):
        """symbols property should return list of symbol strings."""
        dates = pd.date_range("2023-01-01", periods=5, freq="B")
        data = self._make_ohlcv_df(dates)
        portfolio = PortfolioData.from_dict({"AAPL": data, "GOOG": data.copy()})

        symbols = portfolio.symbols

        assert isinstance(symbols, list)
        assert all(isinstance(s, str) for s in symbols)
        assert set(symbols) == {"AAPL", "GOOG"}

    def test_dates_property_returns_datetimeindex(self):
        """dates property should return pandas DatetimeIndex."""
        dates = pd.date_range("2023-01-01", periods=5, freq="B")
        data = self._make_ohlcv_df(dates)
        portfolio = PortfolioData.from_dict({"AAPL": data})

        result = portfolio.dates

        assert isinstance(result, pd.DatetimeIndex)
        assert len(result) == 5

    def test_shape_property(self):
        """shape property should return (n_symbols, n_dates, n_fields)."""
        dates = pd.date_range("2023-01-01", periods=10, freq="B")
        data = self._make_ohlcv_df(dates)
        portfolio = PortfolioData.from_dict(
            {
                "AAPL": data,
                "GOOG": data.copy(),
                "MSFT": data.copy(),
            }
        )

        shape = portfolio.shape

        assert shape == (3, 10, 5)  # 3 symbols, 10 dates, 5 fields (OHLCV)

    def test_close_returns_2d_array(self):
        """close() should return 2D array of shape (n_symbols, n_dates)."""
        dates = pd.date_range("2023-01-01", periods=5, freq="B")
        data = self._make_ohlcv_df(dates, close_val=100.0)
        portfolio = PortfolioData.from_dict({"AAPL": data, "GOOG": data.copy()})

        close = portfolio.close()

        assert isinstance(close, np.ndarray)
        assert close.shape == (2, 5)
        assert np.allclose(close, 100.0)

    def test_open_high_low_volume_accessors(self):
        """OHLCV accessor methods should return correct data."""
        dates = pd.date_range("2023-01-01", periods=5, freq="B")
        data = pd.DataFrame(
            {
                "open": [100.0] * 5,
                "high": [110.0] * 5,
                "low": [90.0] * 5,
                "close": [105.0] * 5,
                "volume": [1e6] * 5,
            },
            index=dates,
        )
        portfolio = PortfolioData.from_dict({"SYM": data})

        assert np.allclose(portfolio.open()[0], 100.0)
        assert np.allclose(portfolio.high()[0], 110.0)
        assert np.allclose(portfolio.low()[0], 90.0)
        assert np.allclose(portfolio.volume()[0], 1e6)

    def test_returns_calculation(self):
        """returns() should calculate period returns correctly."""
        dates = pd.date_range("2023-01-01", periods=5, freq="B")
        # Prices: 100, 110, 121, 133.1, 146.41 (10% daily returns)
        prices = [100, 110, 121, 133.1, 146.41]
        data = pd.DataFrame(
            {
                "open": prices,
                "high": [p * 1.01 for p in prices],
                "low": [p * 0.99 for p in prices],
                "close": prices,
                "volume": [1e6] * 5,
            },
            index=dates,
        )
        portfolio = PortfolioData.from_dict({"SYM": data})

        returns = portfolio.returns()

        assert returns.shape == (1, 4)  # 5 prices -> 4 returns
        assert np.allclose(returns[0], 0.1, atol=0.01)  # ~10% returns

    def test_from_dict_rejects_empty_dict(self):
        """from_dict() should reject empty dictionary."""
        with pytest.raises(ValueError, match="empty"):
            PortfolioData.from_dict({})

    def test_from_dict_requires_datetimeindex(self):
        """from_dict() should require DatetimeIndex."""
        data = pd.DataFrame(
            {
                "open": [100],
                "high": [110],
                "low": [90],
                "close": [100],
                "volume": [1e6],
            }
        )  # No index specified, defaults to RangeIndex

        with pytest.raises(ValueError, match="DatetimeIndex"):
            PortfolioData.from_dict({"SYM": data})

    @staticmethod
    def _make_ohlcv_df(dates, close_val=100.0):
        """Helper to create OHLCV DataFrame."""
        n = len(dates)
        return pd.DataFrame(
            {
                "open": [close_val] * n,
                "high": [close_val * 1.1] * n,
                "low": [close_val * 0.9] * n,
                "close": [close_val] * n,
                "volume": [1e6] * n,
            },
            index=dates,
        )


# ============================================================================
# T006: Contract test for PortfolioData alignment methods
# ============================================================================


class TestPortfolioDataAlignment:
    """Contract tests for PortfolioData alignment."""

    def test_align_forward_fill_missing_values(self):
        """align() with FORWARD_FILL should fill missing dates."""
        dates1 = pd.date_range("2023-01-02", periods=5, freq="B")
        dates2 = pd.date_range("2023-01-03", periods=5, freq="B")  # Offset by 1 day

        data1 = self._make_ohlcv_df(dates1, close_val=100.0)
        data2 = self._make_ohlcv_df(dates2, close_val=200.0)

        portfolio = PortfolioData.from_dict({"SYM1": data1, "SYM2": data2})
        aligned = portfolio.align(policy=MissingDataPolicy.FORWARD_FILL)

        # Should have no NaN in close prices after alignment
        close = aligned.close()
        assert not np.isnan(close).any()

    def test_align_with_date_filter(self):
        """align() should filter to specified date range."""
        dates = pd.date_range("2023-01-01", periods=20, freq="B")
        data = self._make_ohlcv_df(dates)
        portfolio = PortfolioData.from_dict({"SYM": data})

        aligned = portfolio.align(
            start_date="2023-01-10",
            end_date="2023-01-20",
        )

        assert aligned.n_dates < 20
        assert aligned.dates[0] >= pd.Timestamp("2023-01-10")
        assert aligned.dates[-1] <= pd.Timestamp("2023-01-20")

    def test_align_returns_new_instance(self):
        """align() should return new PortfolioData, not modify in place."""
        dates = pd.date_range("2023-01-01", periods=10, freq="B")
        data = self._make_ohlcv_df(dates)
        portfolio = PortfolioData.from_dict({"SYM": data})

        aligned = portfolio.align()

        assert aligned is not portfolio
        assert portfolio.n_dates == 10  # Original unchanged

    def test_align_drop_policy(self):
        """align() with DROP policy should mark missing as non-tradeable."""
        dates1 = pd.date_range("2023-01-02", periods=5, freq="B")
        dates2 = pd.date_range("2023-01-03", periods=5, freq="B")

        data1 = self._make_ohlcv_df(dates1, close_val=100.0)
        data2 = self._make_ohlcv_df(dates2, close_val=200.0)

        portfolio = PortfolioData.from_dict({"SYM1": data1, "SYM2": data2})
        aligned = portfolio.align(policy=MissingDataPolicy.DROP)

        # Tradeable mask should reflect missing data
        mask = aligned.tradeable_mask()
        # SYM2 should have first date marked as non-tradeable
        assert not mask[1, 0]  # SYM2 (index 1) at first date

    def test_align_interpolate_policy(self):
        """align() with INTERPOLATE policy should linearly interpolate."""
        dates = pd.date_range("2023-01-02", periods=5, freq="B")
        # Create data with a gap in the middle
        data = pd.DataFrame(
            {
                "open": [100.0, np.nan, np.nan, 130.0, 140.0],
                "high": [110.0, np.nan, np.nan, 140.0, 150.0],
                "low": [90.0, np.nan, np.nan, 120.0, 130.0],
                "close": [100.0, np.nan, np.nan, 130.0, 140.0],
                "volume": [1e6, np.nan, np.nan, 1e6, 1e6],
            },
            index=dates,
        )

        portfolio = PortfolioData.from_dict({"SYM": data})
        aligned = portfolio.align(policy=MissingDataPolicy.INTERPOLATE)

        close = aligned.close()[0]
        # Values should be interpolated (100 -> 130 over 3 steps)
        assert np.isfinite(close).all()
        assert close[1] > 100 and close[1] < 130

    @staticmethod
    def _make_ohlcv_df(dates, close_val=100.0):
        n = len(dates)
        return pd.DataFrame(
            {
                "open": [close_val] * n,
                "high": [close_val * 1.1] * n,
                "low": [close_val * 0.9] * n,
                "close": [close_val] * n,
                "volume": [1e6] * n,
            },
            index=dates,
        )


# ============================================================================
# T007: Contract test for PortfolioData validation rules
# ============================================================================


class TestPortfolioDataValidation:
    """Contract tests for PortfolioData validation."""

    def test_validate_returns_self_when_valid(self):
        """validate() should return self when data is valid."""
        dates = pd.date_range("2023-01-01", periods=30, freq="B")
        data = self._make_ohlcv_df(dates)
        portfolio = PortfolioData.from_dict({"SYM": data}).align()

        result = portfolio.validate()

        assert result is portfolio

    def test_validate_rejects_insufficient_dates(self):
        """validate() should reject data with too few dates."""
        dates = pd.date_range("2023-01-01", periods=10, freq="B")
        data = self._make_ohlcv_df(dates)
        portfolio = PortfolioData.from_dict({"SYM": data}).align()

        with pytest.raises(ValueError, match="minimum required"):
            portfolio.validate(min_dates=20)

    def test_validate_rejects_nan_values(self):
        """validate() should reject data with NaN when allow_nan=False."""
        dates = pd.date_range("2023-01-01", periods=30, freq="B")
        data = pd.DataFrame(
            {
                "open": [100.0] * 30,
                "high": [110.0] * 30,
                "low": [90.0] * 30,
                "close": [100.0] * 15 + [np.nan] * 15,  # NaN in second half
                "volume": [1e6] * 30,
            },
            index=dates,
        )
        portfolio = PortfolioData.from_dict({"SYM": data})

        with pytest.raises(ValueError, match="NaN"):
            portfolio.validate(allow_nan=False)

    def test_validate_rejects_large_gaps(self):
        """validate() should reject data with gaps larger than max_gap_days."""
        # Create dates with a 10-day gap
        dates1 = pd.date_range("2023-01-02", periods=10, freq="B")
        dates2 = pd.date_range("2023-01-25", periods=10, freq="B")  # 10+ day gap
        dates = dates1.append(dates2)

        data = self._make_ohlcv_df(dates)
        portfolio = PortfolioData.from_dict({"SYM": data}).align()

        with pytest.raises(ValueError, match="gap"):
            portfolio.validate(max_gap_days=5)

    def test_validate_allows_nan_when_specified(self):
        """validate() should allow NaN when allow_nan=True."""
        dates = pd.date_range("2023-01-01", periods=30, freq="B")
        data = pd.DataFrame(
            {
                "open": [100.0] * 30,
                "high": [110.0] * 30,
                "low": [90.0] * 30,
                "close": [100.0] * 15 + [np.nan] * 15,
                "volume": [1e6] * 30,
            },
            index=dates,
        )
        portfolio = PortfolioData.from_dict({"SYM": data})

        # Should not raise
        result = portfolio.validate(allow_nan=True, max_gap_days=30)
        assert result is portfolio

    def test_validate_rejects_symbol_with_no_valid_data(self):
        """validate() should reject symbols with no valid prices."""
        dates = pd.date_range("2023-01-01", periods=30, freq="B")
        good_data = self._make_ohlcv_df(dates)
        bad_data = pd.DataFrame(
            {
                "open": [np.nan] * 30,
                "high": [np.nan] * 30,
                "low": [np.nan] * 30,
                "close": [np.nan] * 30,
                "volume": [np.nan] * 30,
            },
            index=dates,
        )

        portfolio = PortfolioData.from_dict({"GOOD": good_data, "BAD": bad_data})

        with pytest.raises(ValueError, match="no valid prices"):
            portfolio.validate(allow_nan=True)

    @staticmethod
    def _make_ohlcv_df(dates, close_val=100.0):
        n = len(dates)
        return pd.DataFrame(
            {
                "open": [close_val] * n,
                "high": [close_val * 1.1] * n,
                "low": [close_val * 0.9] * n,
                "close": [close_val] * n,
                "volume": [1e6] * n,
            },
            index=dates,
        )


# ============================================================================
# Additional contract tests for PortfolioData
# ============================================================================


class TestPortfolioDataMetadata:
    """Contract tests for PortfolioData metadata handling."""

    def test_get_metadata_returns_none_for_missing(self):
        """get_metadata() should return None for symbol without metadata."""
        dates = pd.date_range("2023-01-01", periods=10, freq="B")
        data = self._make_ohlcv_df(dates)
        portfolio = PortfolioData.from_dict({"SYM": data})

        result = portfolio.get_metadata("SYM")

        assert result is None

    def test_set_metadata_returns_new_instance(self):
        """set_metadata() should return new PortfolioData with metadata."""
        dates = pd.date_range("2023-01-01", periods=10, freq="B")
        data = self._make_ohlcv_df(dates)
        portfolio = PortfolioData.from_dict({"SYM": data})

        metadata = SymbolMetadata(symbol="SYM", sector="Technology", market_cap=1e12)
        new_portfolio = portfolio.set_metadata("SYM", metadata)

        assert new_portfolio is not portfolio
        assert new_portfolio.get_metadata("SYM") == metadata
        assert portfolio.get_metadata("SYM") is None  # Original unchanged

    def test_set_metadata_rejects_unknown_symbol(self):
        """set_metadata() should reject symbols not in portfolio."""
        dates = pd.date_range("2023-01-01", periods=10, freq="B")
        data = self._make_ohlcv_df(dates)
        portfolio = PortfolioData.from_dict({"SYM": data})

        metadata = SymbolMetadata(symbol="UNKNOWN", sector="Technology")

        with pytest.raises(ValueError, match="not in portfolio"):
            portfolio.set_metadata("UNKNOWN", metadata)

    def test_symbol_metadata_validates_market_cap(self):
        """SymbolMetadata should reject non-positive market_cap."""
        with pytest.raises(ValueError, match="market_cap"):
            SymbolMetadata(symbol="SYM", market_cap=-1000)

        with pytest.raises(ValueError, match="market_cap"):
            SymbolMetadata(symbol="SYM", market_cap=0)

    @staticmethod
    def _make_ohlcv_df(dates, close_val=100.0):
        n = len(dates)
        return pd.DataFrame(
            {
                "open": [close_val] * n,
                "high": [close_val * 1.1] * n,
                "low": [close_val * 0.9] * n,
                "close": [close_val] * n,
                "volume": [1e6] * n,
            },
            index=dates,
        )


class TestPortfolioDataFiltering:
    """Contract tests for PortfolioData filtering."""

    def test_filter_symbols_returns_subset(self):
        """filter_symbols() should return PortfolioData with only specified symbols."""
        dates = pd.date_range("2023-01-01", periods=10, freq="B")
        data = self._make_ohlcv_df(dates)
        portfolio = PortfolioData.from_dict(
            {
                "AAPL": data,
                "GOOG": data.copy(),
                "MSFT": data.copy(),
            }
        )

        filtered = portfolio.filter_symbols(["AAPL", "GOOG"])

        assert filtered.n_symbols == 2
        assert set(filtered.symbols) == {"AAPL", "GOOG"}
        assert portfolio.n_symbols == 3  # Original unchanged

    def test_filter_dates_returns_date_range(self):
        """filter_dates() should return PortfolioData with only specified date range."""
        dates = pd.date_range("2023-01-01", periods=30, freq="B")
        data = self._make_ohlcv_df(dates)
        portfolio = PortfolioData.from_dict({"SYM": data})

        filtered = portfolio.filter_dates("2023-01-10", "2023-01-20")

        assert filtered.n_dates < 30
        assert filtered.dates[0] >= pd.Timestamp("2023-01-10")
        assert filtered.dates[-1] <= pd.Timestamp("2023-01-20")

    def test_get_symbol_data_returns_dataframe(self):
        """get_symbol_data() should return OHLCV DataFrame for single symbol."""
        dates = pd.date_range("2023-01-01", periods=10, freq="B")
        data = self._make_ohlcv_df(dates, close_val=100.0)
        portfolio = PortfolioData.from_dict({"SYM": data})

        result = portfolio.get_symbol_data("SYM")

        assert isinstance(result, pd.DataFrame)
        assert set(result.columns) == {"open", "high", "low", "close", "volume"}
        assert len(result) == 10

    @staticmethod
    def _make_ohlcv_df(dates, close_val=100.0):
        n = len(dates)
        return pd.DataFrame(
            {
                "open": [close_val] * n,
                "high": [close_val * 1.1] * n,
                "low": [close_val * 0.9] * n,
                "close": [close_val] * n,
                "volume": [1e6] * n,
            },
            index=dates,
        )
