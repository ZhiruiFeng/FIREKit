"""
Unit tests for corporate actions (T080, US5).

Covers Split, Dividend, Merger, Spinoff dataclasses, the
CorporateActionsRegistry, and apply_actions() price adjustment
(FR-004, FR-005).
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from vectorforge.portfolio.corporate_actions import (
    CorporateActionsRegistry,
    CorporateActionType,
    Dividend,
    Merger,
    Spinoff,
    Split,
    apply_actions,
)
from vectorforge.portfolio.data import PortfolioData


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


N = 40
DATES = pd.date_range("2023-01-02", periods=N, freq="B")
EX_IDX = 20
EX_DATE = DATES[EX_IDX].date()


@pytest.fixture
def flat_portfolio() -> PortfolioData:
    """Two symbols with constant prices (X=100, Y=200)."""
    return PortfolioData.from_dict(
        {
            "X": make_ohlcv_df(np.full(N, 100.0), DATES),
            "Y": make_ohlcv_df(np.full(N, 200.0), DATES),
        }
    )


class TestSplit:
    """Split adjustments (FR-004, US5 scenario 1)."""

    def test_action_type(self):
        split = Split(symbol="X", ex_date=EX_DATE, ratio=2.0)
        assert split.action_type == CorporateActionType.SPLIT

    def test_two_for_one_split_halves_price(self):
        split = Split(symbol="X", ex_date=EX_DATE, ratio=2.0)
        assert split.adjust_price(100.0) == pytest.approx(50.0)

    def test_two_for_one_split_doubles_shares(self):
        split = Split(symbol="X", ex_date=EX_DATE, ratio=2.0)
        assert split.adjust_shares(100) == 200
        assert split.adjust_shares(100.0) == pytest.approx(200.0)

    def test_split_preserves_position_value(self):
        """2:1 split: quantity doubles, price halves, value unchanged."""
        split = Split(symbol="X", ex_date=EX_DATE, ratio=2.0)
        shares, price = 100, 80.0
        new_shares = split.adjust_shares(shares)
        new_price = split.adjust_price(price)
        assert new_shares == 200
        assert new_price == pytest.approx(40.0)
        assert new_shares * new_price == pytest.approx(shares * price)

    def test_reverse_split(self):
        split = Split(symbol="X", ex_date=EX_DATE, ratio=0.1)  # 1-for-10
        assert split.adjust_shares(100) == 10
        assert split.adjust_price(10.0) == pytest.approx(100.0)
        assert split.adjust_shares(100) * split.adjust_price(10.0) == pytest.approx(100 * 10.0)

    def test_volume_adjustment(self):
        split = Split(symbol="X", ex_date=EX_DATE, ratio=4.0)
        assert split.adjust_volume(1e6) == pytest.approx(4e6)


class TestDividend:
    """Dividend handling (FR-005, US5 scenarios 2 and 3)."""

    def test_action_type(self):
        div = Dividend(symbol="X", ex_date=EX_DATE, amount=1.0)
        assert div.action_type == CorporateActionType.DIVIDEND

    def test_cash_payout_for_100_shares(self):
        """$1.00 dividend on 100 shares credits $100 cash."""
        div = Dividend(symbol="X", ex_date=EX_DATE, amount=1.0)
        assert div.cash_payout(100) == pytest.approx(100.0)

    def test_reinvestment_buys_fractional_shares(self):
        """DRIP buys fractional shares at the current price."""
        div = Dividend(symbol="X", ex_date=EX_DATE, amount=1.0)
        assert div.reinvestment_shares(100, price=50.0) == pytest.approx(2.0)
        assert div.reinvestment_shares(100, price=80.0) == pytest.approx(1.25)

    def test_reinvestment_with_zero_price_is_zero(self):
        div = Dividend(symbol="X", ex_date=EX_DATE, amount=1.0)
        assert div.reinvestment_shares(100, price=0.0) == 0.0

    def test_adjustment_factor(self):
        div = Dividend(symbol="X", ex_date=EX_DATE, amount=1.0)
        assert div.adjustment_factor(100.0) == pytest.approx(0.99)

    def test_adjustment_factor_invalid_price_is_one(self):
        div = Dividend(symbol="X", ex_date=EX_DATE, amount=1.0)
        assert div.adjustment_factor(0.0) == 1.0
        assert div.adjustment_factor(-5.0) == 1.0


class TestMerger:
    def test_convert_shares(self):
        merger = Merger(
            symbol="TGT", ex_date=EX_DATE, acquirer="ACQ", ratio=0.5, cash_per_share=10.0
        )
        acquirer_shares, cash = merger.convert_shares(100)
        assert acquirer_shares == pytest.approx(50.0)
        assert cash == pytest.approx(1000.0)
        assert merger.action_type == CorporateActionType.MERGER


class TestSpinoff:
    def test_spinoff_shares(self):
        spinoff = Spinoff(symbol="PARENT", ex_date=EX_DATE, spinoff_symbol="CHILD", ratio=0.25)
        assert spinoff.spinoff_shares(100) == pytest.approx(25.0)
        assert spinoff.action_type == CorporateActionType.SPINOFF


class TestCorporateActionsRegistry:
    @pytest.fixture
    def registry(self) -> CorporateActionsRegistry:
        actions = [
            Split(symbol="X", ex_date=date(2023, 2, 1), ratio=2.0),
            Dividend(symbol="X", ex_date=date(2023, 1, 15), amount=0.5),
            Dividend(symbol="Y", ex_date=date(2023, 3, 1), amount=1.0),
            Merger(symbol="Z", ex_date=date(2023, 4, 1), acquirer="X", ratio=1.0),
        ]
        return CorporateActionsRegistry.from_list(actions)

    def test_len_and_iter(self, registry):
        assert len(registry) == 4
        assert len(list(registry)) == 4

    def test_actions_for_symbol_sorted_by_date(self, registry):
        actions = registry.get_actions_for_symbol("X")
        assert len(actions) == 2
        assert actions[0].ex_date <= actions[1].ex_date
        assert isinstance(actions[0], Dividend)
        assert isinstance(actions[1], Split)

    def test_unknown_symbol_returns_empty(self, registry):
        assert registry.get_actions_for_symbol("NOPE") == []

    def test_filter_date_range(self, registry):
        filtered = registry.filter_date_range(date(2023, 1, 1), date(2023, 2, 28))
        assert len(filtered) == 2
        assert all(a.ex_date <= date(2023, 2, 28) for a in filtered)

    def test_get_actions_between(self, registry):
        actions = registry.get_actions_between(date(2023, 3, 1), date(2023, 4, 30))
        assert len(actions) == 2

    def test_typed_getters(self, registry):
        assert len(registry.get_splits()) == 1
        assert len(registry.get_dividends()) == 2
        assert len(registry.get_dividends("X")) == 1
        assert len(registry.get_mergers()) == 1
        assert len(registry.get_spinoffs()) == 0


class TestApplyActions:
    """Historical price adjustment via apply_actions (FR-004)."""

    def test_split_adjusts_prices_before_ex_date_only(self, flat_portfolio):
        split = Split(symbol="X", ex_date=EX_DATE, ratio=2.0)
        adjusted = apply_actions(flat_portfolio, [split])

        close = adjusted.close()
        assert np.allclose(close[0, :EX_IDX], 50.0, rtol=1e-5)
        assert np.allclose(close[0, EX_IDX:], 100.0, rtol=1e-5)

    def test_split_adjusts_all_ohlc_fields(self, flat_portfolio):
        split = Split(symbol="X", ex_date=EX_DATE, ratio=2.0)
        adjusted = apply_actions(flat_portfolio, [split])

        original = flat_portfolio
        for getter in ("open", "high", "low", "close"):
            adj_vals = getattr(adjusted, getter)()[0]
            orig_vals = getattr(original, getter)()[0]
            assert np.allclose(adj_vals[:EX_IDX], orig_vals[:EX_IDX] / 2.0, rtol=1e-5)
            assert np.allclose(adj_vals[EX_IDX:], orig_vals[EX_IDX:], rtol=1e-5)

    def test_split_multiplies_historical_volume(self, flat_portfolio):
        split = Split(symbol="X", ex_date=EX_DATE, ratio=2.0)
        adjusted = apply_actions(flat_portfolio, [split])

        volume = adjusted.volume()
        assert np.allclose(volume[0, :EX_IDX], 2e6, rtol=1e-5)
        assert np.allclose(volume[0, EX_IDX:], 1e6, rtol=1e-5)

    def test_split_does_not_touch_other_symbols(self, flat_portfolio):
        split = Split(symbol="X", ex_date=EX_DATE, ratio=2.0)
        adjusted = apply_actions(flat_portfolio, [split])
        assert np.allclose(adjusted.close()[1], 200.0, rtol=1e-5)

    def test_dividend_adjusts_historical_prices(self, flat_portfolio):
        div = Dividend(symbol="X", ex_date=EX_DATE, amount=1.0)
        adjusted = apply_actions(flat_portfolio, [div])

        close = adjusted.close()
        # Pre-ex prices scaled by (100 - 1) / 100 = 0.99
        assert np.allclose(close[0, :EX_IDX], 99.0, rtol=1e-5)
        assert np.allclose(close[0, EX_IDX:], 100.0, rtol=1e-5)

    def test_empty_actions_returns_same_data(self, flat_portfolio):
        assert apply_actions(flat_portfolio, []) is flat_portfolio

    def test_action_for_unknown_symbol_is_ignored(self, flat_portfolio):
        split = Split(symbol="UNKNOWN", ex_date=EX_DATE, ratio=2.0)
        adjusted = apply_actions(flat_portfolio, [split])
        assert np.allclose(adjusted.close(), flat_portfolio.close(), rtol=1e-6)

    def test_apply_corporate_actions_method_delegates(self, flat_portfolio):
        split = Split(symbol="X", ex_date=EX_DATE, ratio=2.0)
        adjusted = flat_portfolio.apply_corporate_actions([split])
        assert np.allclose(adjusted.close()[0, :EX_IDX], 50.0, rtol=1e-5)

    def test_original_data_unmodified(self, flat_portfolio):
        split = Split(symbol="X", ex_date=EX_DATE, ratio=2.0)
        apply_actions(flat_portfolio, [split])
        assert np.allclose(flat_portfolio.close()[0], 100.0, rtol=1e-6)

    def test_multiple_actions_applied_chronologically(self, flat_portfolio):
        early_div = Dividend(symbol="X", ex_date=DATES[10].date(), amount=1.0)
        later_split = Split(symbol="X", ex_date=EX_DATE, ratio=2.0)
        adjusted = apply_actions(flat_portfolio, [later_split, early_div])

        close = adjusted.close()
        # Before dividend ex-date: dividend factor then split divisor
        assert np.allclose(close[0, :10], 99.0 / 2.0, rtol=1e-5)
        # Between dividend and split: only split divisor
        assert np.allclose(close[0, 10:EX_IDX], 50.0, rtol=1e-5)
        # After split ex-date: unchanged
        assert np.allclose(close[0, EX_IDX:], 100.0, rtol=1e-5)

    def test_split_after_data_window_adjusts_everything(self, flat_portfolio):
        split = Split(symbol="X", ex_date=date(2024, 1, 1), ratio=2.0)
        adjusted = apply_actions(flat_portfolio, [split])
        assert np.allclose(adjusted.close()[0], 50.0, rtol=1e-5)
