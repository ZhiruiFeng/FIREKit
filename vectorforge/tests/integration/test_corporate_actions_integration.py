"""
Integration test: backtest with splits and dividends (T081, US5).

Covers the US5 acceptance scenarios end to end:
- a 2:1 split doubles position quantity while keeping value unchanged,
  and split-adjusted data produces identical P&L to pre-adjusted data
  (SC-006, FR-004);
- a cash dividend credits cash (no reinvestment) and DRIP buys
  fractional shares (FR-005).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vectorforge import VectorizedBacktester
from vectorforge.portfolio.corporate_actions import Dividend, Split
from vectorforge.portfolio.data import PortfolioData
from vectorforge.portfolio.signals import TargetWeights

N_DAYS = 80
DATES = pd.date_range("2023-01-02", periods=N_DAYS, freq="B")
SPLIT_IDX = 40
SPLIT_DATE = DATES[SPLIT_IDX].date()


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
def engine() -> VectorizedBacktester:
    return VectorizedBacktester()


class TestBacktestWithTwoForOneSplit:
    """US5 scenario 1: 2:1 split on a held position."""

    @pytest.fixture(scope="class")
    def true_prices(self) -> np.ndarray:
        """Economically continuous price path (already split-adjusted)."""
        return 100.0 + np.arange(N_DAYS, dtype=float)

    @pytest.fixture(scope="class")
    def unadjusted_data(self, true_prices) -> PortfolioData:
        """Raw exchange prices: double before the split's ex-date."""
        unadjusted = true_prices.copy()
        unadjusted[:SPLIT_IDX] *= 2.0
        return PortfolioData.from_dict({"SPLITCO": make_ohlcv_df(unadjusted, DATES)})

    @pytest.fixture(scope="class")
    def adjusted_data(self, unadjusted_data) -> PortfolioData:
        split = Split(symbol="SPLITCO", ex_date=SPLIT_DATE, ratio=2.0)
        return unadjusted_data.apply_corporate_actions([split])

    def test_position_quantity_doubles_value_unchanged(self):
        """100 shares @ $80 become 200 shares @ $40: value constant."""
        split = Split(symbol="SPLITCO", ex_date=SPLIT_DATE, ratio=2.0)
        shares, price = 100, 80.0

        new_shares = split.adjust_shares(shares)
        new_price = split.adjust_price(price)

        assert new_shares == 200
        assert new_price == pytest.approx(40.0)
        assert new_shares * new_price == pytest.approx(shares * price)

    def test_adjustment_recovers_continuous_prices(self, adjusted_data, true_prices):
        close = adjusted_data.close()[0]
        assert np.allclose(close, true_prices, rtol=1e-6)

    def test_no_artificial_return_at_split_date(self, adjusted_data):
        """The split itself must not show up as a -50% return."""
        returns = adjusted_data.returns()[0]
        split_day_return = returns[SPLIT_IDX - 1]
        assert abs(split_day_return) < 0.05  # ~+0.7% price move, not -50%

    def test_backtest_pnl_identical_to_preadjusted_data(
        self, engine, adjusted_data, true_prices
    ):
        """SC-006: corporate action adjustment reproduces pre-adjusted P&L."""
        native_data = PortfolioData.from_dict({"SPLITCO": make_ohlcv_df(true_prices, DATES)})
        weights = TargetWeights.equal_weight(["SPLITCO"], DATES)

        result_adjusted = engine.run_portfolio(weights, adjusted_data, initial_capital=100_000)
        result_native = engine.run_portfolio(weights, native_data, initial_capital=100_000)

        assert result_adjusted.total_return == pytest.approx(
            result_native.total_return, rel=1e-9, abs=1e-12
        )
        assert np.allclose(
            result_adjusted.returns.values, result_native.returns.values, atol=1e-12
        )
        assert np.allclose(
            result_adjusted.equity_curve.values,
            result_native.equity_curve.values,
            rtol=1e-9,
        )

    def test_unadjusted_backtest_shows_spurious_loss(self, engine, unadjusted_data):
        """Sanity check: without adjustment the split looks like a -50% crash."""
        weights = TargetWeights.equal_weight(["SPLITCO"], DATES)
        result = engine.run_portfolio(weights, unadjusted_data, initial_capital=100_000)
        split_day_return = result.returns.iloc[SPLIT_IDX - 1]
        assert split_day_return < -0.45


class TestBacktestWithDividends:
    """US5 scenarios 2 and 3: cash payout and dividend reinvestment."""

    DIV_AMOUNT = 1.0
    FLAT_PRICE = 100.0

    @pytest.fixture(scope="class")
    def dividend(self) -> Dividend:
        return Dividend(symbol="DIVCO", ex_date=SPLIT_DATE, amount=self.DIV_AMOUNT)

    @pytest.fixture(scope="class")
    def flat_data(self) -> PortfolioData:
        return PortfolioData.from_dict(
            {"DIVCO": make_ohlcv_df(np.full(N_DAYS, self.FLAT_PRICE), DATES)}
        )

    def test_cash_payout_on_100_shares(self, dividend):
        """$1.00 dividend on 100 shares credits $100 cash (no reinvestment)."""
        assert dividend.cash_payout(100) == pytest.approx(100.0)

    def test_drip_buys_fractional_shares(self, dividend):
        """With reinvestment enabled, the $100 payout buys fractional shares
        at the current price and total position value grows accordingly."""
        shares = 100
        price = 80.0
        new_shares = dividend.reinvestment_shares(shares, price)

        assert new_shares == pytest.approx(1.25)  # fractional shares allowed
        total_value = (shares + new_shares) * price
        assert total_value == pytest.approx(shares * price + dividend.cash_payout(shares))

    def test_dividend_adjustment_factor_applied_to_history(self, flat_data, dividend):
        adjusted = flat_data.apply_corporate_actions([dividend])
        close = adjusted.close()[0]

        factor = (self.FLAT_PRICE - self.DIV_AMOUNT) / self.FLAT_PRICE  # 0.99
        assert np.allclose(close[:SPLIT_IDX], self.FLAT_PRICE * factor, rtol=1e-5)
        assert np.allclose(close[SPLIT_IDX:], self.FLAT_PRICE, rtol=1e-5)

    def test_backtest_on_adjusted_data_captures_dividend_return(
        self, engine, flat_data, dividend
    ):
        """A flat-price stock paying a $1 dividend yields ~1% total return
        when backtested on dividend-adjusted prices."""
        adjusted = flat_data.apply_corporate_actions([dividend])
        weights = TargetWeights.equal_weight(["DIVCO"], DATES)
        result = engine.run_portfolio(weights, adjusted, initial_capital=100_000)

        expected = self.FLAT_PRICE / (self.FLAT_PRICE - self.DIV_AMOUNT) - 1  # ~1.01%
        assert result.total_return == pytest.approx(expected, rel=1e-3)

        # The whole return arrives on the ex-date
        ex_return = result.returns.iloc[SPLIT_IDX - 1]
        assert ex_return == pytest.approx(expected, rel=1e-3)
        other_days = result.returns.drop(result.returns.index[SPLIT_IDX - 1])
        assert np.allclose(other_days.values, 0.0, atol=1e-6)

    def test_unadjusted_flat_stock_has_zero_return(self, engine, flat_data):
        """Without the dividend adjustment the flat stock earns nothing."""
        weights = TargetWeights.equal_weight(["DIVCO"], DATES)
        result = engine.run_portfolio(weights, flat_data, initial_capital=100_000)
        assert result.total_return == pytest.approx(0.0, abs=1e-6)


class TestCombinedActionsInMultiAssetBacktest:
    """Splits and dividends together in a multi-asset portfolio."""

    @pytest.fixture(scope="class")
    def result_pair(self, engine):
        """Backtests on (adjusted unadjusted data, natively continuous data)."""
        rng = np.random.default_rng(17)
        base_a = 100 * np.exp(np.cumsum(rng.normal(0.0005, 0.01, N_DAYS)))
        base_b = np.full(N_DAYS, 50.0)

        # SPLITCO has a 2:1 split: raw prices double before ex-date
        raw_a = base_a.copy()
        raw_a[:SPLIT_IDX] *= 2.0

        actions = [
            Split(symbol="SPLITCO", ex_date=SPLIT_DATE, ratio=2.0),
            Dividend(symbol="DIVCO", ex_date=DATES[60].date(), amount=0.5),
        ]

        raw_data = PortfolioData.from_dict(
            {"SPLITCO": make_ohlcv_df(raw_a, DATES), "DIVCO": make_ohlcv_df(base_b, DATES)}
        )
        adjusted = raw_data.apply_corporate_actions(actions)

        weights = TargetWeights.equal_weight(adjusted.symbols, adjusted.dates)
        result = engine.run_portfolio(weights, adjusted, initial_capital=200_000)

        # Reference: continuous split series + manually dividend-adjusted flat
        factor = (50.0 - 0.5) / 50.0
        ref_b = base_b.copy()
        ref_b[:60] *= factor
        ref_data = PortfolioData.from_dict(
            {"SPLITCO": make_ohlcv_df(base_a, DATES), "DIVCO": make_ohlcv_df(ref_b, DATES)}
        )
        ref_result = engine.run_portfolio(weights, ref_data, initial_capital=200_000)
        return result, ref_result

    def test_combined_adjustments_match_reference(self, result_pair):
        result, ref_result = result_pair
        assert np.allclose(result.returns.values, ref_result.returns.values, atol=1e-6)
        assert result.total_return == pytest.approx(ref_result.total_return, abs=1e-5)

    def test_portfolio_result_is_well_formed(self, result_pair):
        result, _ = result_pair
        assert result.equity_curve.iloc[0] == pytest.approx(200_000)
        assert np.isfinite(result.equity_curve.values).all()
        contrib_sum = result.asset_contributions.sum(axis=1)
        assert np.allclose(contrib_sum.values, result.returns.values, atol=1e-10)
