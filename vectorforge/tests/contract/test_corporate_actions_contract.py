"""
Contract Tests for Corporate Actions

These tests define the expected API contract for corporate action handling.
They must FAIL before implementation and PASS after.
"""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from vectorforge.portfolio.data import PortfolioData

# ============================================================================
# T071: Contract test for CorporateAction dataclass
# ============================================================================


class TestCorporateActionDataclass:
    """Contract tests for CorporateAction dataclass."""

    def test_split_action_has_required_fields(self):
        """Split action should have symbol, date, and ratio."""
        from vectorforge.portfolio.corporate_actions import Split

        split = Split(
            symbol="AAPL",
            ex_date=date(2020, 8, 31),
            ratio=4.0,  # 4-for-1 split
        )

        assert split.symbol == "AAPL"
        assert split.ex_date == date(2020, 8, 31)
        assert split.ratio == 4.0

    def test_dividend_action_has_required_fields(self):
        """Dividend action should have symbol, date, and amount."""
        from vectorforge.portfolio.corporate_actions import Dividend

        dividend = Dividend(
            symbol="AAPL",
            ex_date=date(2023, 5, 12),
            amount=0.24,  # $0.24 per share
        )

        assert dividend.symbol == "AAPL"
        assert dividend.ex_date == date(2023, 5, 12)
        assert dividend.amount == 0.24

    def test_corporate_action_types_enum(self):
        """CorporateActionType enum should have standard types."""
        from vectorforge.portfolio.corporate_actions import CorporateActionType

        assert hasattr(CorporateActionType, "SPLIT")
        assert hasattr(CorporateActionType, "DIVIDEND")
        assert hasattr(CorporateActionType, "MERGER")
        assert hasattr(CorporateActionType, "SPINOFF")

    def test_split_adjust_price(self):
        """Split should be able to adjust a price."""
        from vectorforge.portfolio.corporate_actions import Split

        split = Split(
            symbol="AAPL",
            ex_date=date(2020, 8, 31),
            ratio=4.0,
        )

        # Before split: $500, after 4-for-1: $125
        adjusted = split.adjust_price(500.0)
        assert adjusted == pytest.approx(125.0, rel=0.01)

    def test_split_adjust_shares(self):
        """Split should be able to adjust share count."""
        from vectorforge.portfolio.corporate_actions import Split

        split = Split(
            symbol="AAPL",
            ex_date=date(2020, 8, 31),
            ratio=4.0,
        )

        # Before split: 100 shares, after 4-for-1: 400 shares
        adjusted = split.adjust_shares(100)
        assert adjusted == 400

    def test_reverse_split(self):
        """Reverse split should work with ratio < 1."""
        from vectorforge.portfolio.corporate_actions import Split

        # 1-for-10 reverse split
        reverse_split = Split(
            symbol="XYZ",
            ex_date=date(2023, 1, 15),
            ratio=0.1,
        )

        # Before: $5, after 1-for-10: $50
        adjusted_price = reverse_split.adjust_price(5.0)
        assert adjusted_price == pytest.approx(50.0, rel=0.01)

        # Before: 1000 shares, after 1-for-10: 100 shares
        adjusted_shares = reverse_split.adjust_shares(1000)
        assert adjusted_shares == 100


# ============================================================================
# T072: Contract test for PortfolioData.apply_corporate_actions()
# ============================================================================


class TestPortfolioDataApplyCorporateActions:
    """Contract tests for PortfolioData.apply_corporate_actions()."""

    def test_apply_split_adjusts_historical_prices(self):
        """apply_corporate_actions should adjust prices before split date."""
        from vectorforge.portfolio.corporate_actions import Split

        # Create data with a clear price pattern
        dates = pd.date_range("2020-08-01", periods=40, freq="B")
        pre_split_price = 500.0  # Before split
        post_split_price = 125.0  # After 4-for-1 split

        # Split date: 2020-08-31 (roughly middle of our dates)
        split_date = date(2020, 8, 31)
        split_idx = dates.get_indexer([pd.Timestamp(split_date)], method="nearest")[0]

        prices = np.array([pre_split_price] * split_idx + [post_split_price] * (40 - split_idx))
        aapl = pd.DataFrame(
            {
                "open": prices * 0.99,
                "high": prices * 1.01,
                "low": prices * 0.98,
                "close": prices,
                "volume": [1e6] * 40,
            },
            index=dates,
        )

        portfolio = PortfolioData.from_dict({"AAPL": aapl})
        split = Split(symbol="AAPL", ex_date=split_date, ratio=4.0)

        # When: applying the split
        adjusted = portfolio.apply_corporate_actions([split])

        # Then: Historical prices should be adjusted down
        # All close prices should now be around $125
        # close() returns 2D array (n_symbols, n_dates), AAPL is index 0
        close_prices = adjusted.close()[0]  # First symbol (AAPL)
        assert close_prices[0] == pytest.approx(125.0, rel=0.05)
        assert close_prices[-1] == pytest.approx(125.0, rel=0.05)

    def test_apply_split_adjusts_volume(self):
        """apply_corporate_actions should adjust volume inversely."""
        from vectorforge.portfolio.corporate_actions import Split

        dates = pd.date_range("2020-08-01", periods=10, freq="B")
        pre_vol = 1_000_000

        aapl = pd.DataFrame(
            {
                "open": [500.0] * 10,
                "high": [510.0] * 10,
                "low": [490.0] * 10,
                "close": [500.0] * 10,
                "volume": [pre_vol] * 10,
            },
            index=dates,
        )

        portfolio = PortfolioData.from_dict({"AAPL": aapl})
        split = Split(symbol="AAPL", ex_date=date(2020, 8, 15), ratio=4.0)

        adjusted = portfolio.apply_corporate_actions([split])

        # Volume before split should be multiplied by 4
        volume = adjusted.volume()[0]  # First symbol (AAPL)
        # Pre-split volume adjusted: 1M * 4 = 4M
        assert volume[0] == pytest.approx(4_000_000, rel=0.01)

    def test_apply_multiple_splits(self):
        """Should handle multiple splits for same symbol."""
        from vectorforge.portfolio.corporate_actions import Split

        dates = pd.date_range("2020-01-01", periods=100, freq="B")

        aapl = pd.DataFrame(
            {
                "open": [100.0] * 100,
                "high": [101.0] * 100,
                "low": [99.0] * 100,
                "close": [100.0] * 100,
                "volume": [1e6] * 100,
            },
            index=dates,
        )

        portfolio = PortfolioData.from_dict({"AAPL": aapl})

        # Two splits: 2-for-1 then 5-for-1 (10x total)
        splits = [
            Split(symbol="AAPL", ex_date=date(2020, 2, 1), ratio=2.0),
            Split(symbol="AAPL", ex_date=date(2020, 4, 1), ratio=5.0),
        ]

        adjusted = portfolio.apply_corporate_actions(splits)
        close = adjusted.close()[0]  # First symbol (AAPL)

        # All prices should be adjusted to $10 (100 / 10)
        assert close[0] == pytest.approx(10.0, rel=0.1)

    def test_apply_split_only_affects_correct_symbol(self):
        """Split should only affect the specified symbol."""
        from vectorforge.portfolio.corporate_actions import Split

        dates = pd.date_range("2020-08-01", periods=10, freq="B")

        aapl = pd.DataFrame(
            {
                "open": [500.0] * 10,
                "high": [510.0] * 10,
                "low": [490.0] * 10,
                "close": [500.0] * 10,
                "volume": [1e6] * 10,
            },
            index=dates,
        )

        goog = pd.DataFrame(
            {
                "open": [1500.0] * 10,
                "high": [1510.0] * 10,
                "low": [1490.0] * 10,
                "close": [1500.0] * 10,
                "volume": [5e5] * 10,
            },
            index=dates,
        )

        portfolio = PortfolioData.from_dict({"AAPL": aapl, "GOOG": goog})
        split = Split(symbol="AAPL", ex_date=date(2020, 8, 5), ratio=4.0)

        adjusted = portfolio.apply_corporate_actions([split])

        # GOOG should be unchanged - it's the second symbol (index 1)
        goog_idx = adjusted.symbols.index("GOOG")
        goog_close = adjusted.close()[goog_idx]
        assert goog_close[0] == pytest.approx(1500.0, rel=0.01)

    def test_apply_corporate_actions_returns_new_instance(self):
        """apply_corporate_actions should return new PortfolioData, not mutate."""
        from vectorforge.portfolio.corporate_actions import Split

        dates = pd.date_range("2020-08-01", periods=10, freq="B")
        aapl = pd.DataFrame(
            {
                "open": [500.0] * 10,
                "high": [510.0] * 10,
                "low": [490.0] * 10,
                "close": [500.0] * 10,
                "volume": [1e6] * 10,
            },
            index=dates,
        )

        portfolio = PortfolioData.from_dict({"AAPL": aapl})
        original_close = portfolio.close()[0][0]  # First symbol, first date

        split = Split(symbol="AAPL", ex_date=date(2020, 8, 5), ratio=4.0)
        adjusted = portfolio.apply_corporate_actions([split])

        # Original should be unchanged
        assert portfolio.close()[0][0] == original_close
        # Adjusted should be different
        assert adjusted.close()[0][0] != original_close


class TestDividendHandling:
    """Contract tests for dividend handling."""

    def test_dividend_cash_payout(self):
        """Dividend should calculate cash payout."""
        from vectorforge.portfolio.corporate_actions import Dividend

        dividend = Dividend(
            symbol="AAPL",
            ex_date=date(2023, 5, 12),
            amount=0.24,
        )

        # 100 shares * $0.24 = $24.00 cash payout
        payout = dividend.cash_payout(shares=100)
        assert payout == pytest.approx(24.0, rel=0.01)

    def test_dividend_reinvestment_shares(self):
        """Dividend should calculate reinvestment shares."""
        from vectorforge.portfolio.corporate_actions import Dividend

        dividend = Dividend(
            symbol="AAPL",
            ex_date=date(2023, 5, 12),
            amount=0.24,
        )

        # 100 shares * $0.24 = $24.00, at $170/share = 0.141 shares
        new_shares = dividend.reinvestment_shares(shares=100, price=170.0)
        assert new_shares == pytest.approx(0.141, rel=0.01)

    def test_dividend_affects_adjusted_close(self):
        """Dividends should be reflected in adjusted close calculations."""
        from vectorforge.portfolio.corporate_actions import Dividend

        dates = pd.date_range("2023-05-01", periods=20, freq="B")
        ex_date = date(2023, 5, 12)  # Roughly middle

        aapl = pd.DataFrame(
            {
                "open": [170.0] * 20,
                "high": [171.0] * 20,
                "low": [169.0] * 20,
                "close": [170.0] * 20,
                "volume": [1e6] * 20,
            },
            index=dates,
        )

        portfolio = PortfolioData.from_dict({"AAPL": aapl})
        dividend = Dividend(symbol="AAPL", ex_date=ex_date, amount=0.24)

        adjusted = portfolio.apply_corporate_actions([dividend])

        # After dividend, historical prices should be slightly lower
        # to account for dividend value (maintains total return)
        close_prices = adjusted.close()[0]  # First symbol (AAPL)
        pre_div_close = close_prices[0]
        post_div_close = close_prices[-1]

        # Pre-dividend adjusted close should be lower by approx (1 - 0.24/170)
        adjustment_factor = (170.0 - 0.24) / 170.0
        assert pre_div_close == pytest.approx(170.0 * adjustment_factor, rel=0.01)
        # Post-dividend close unchanged
        assert post_div_close == pytest.approx(170.0, rel=0.01)


class TestCorporateActionsRegistry:
    """Contract tests for CorporateActionsRegistry."""

    def test_registry_can_load_from_list(self):
        """Registry should be able to load actions from a list."""
        from vectorforge.portfolio.corporate_actions import (
            CorporateActionsRegistry,
            Dividend,
            Split,
        )

        actions = [
            Split(symbol="AAPL", ex_date=date(2020, 8, 31), ratio=4.0),
            Dividend(symbol="AAPL", ex_date=date(2023, 5, 12), amount=0.24),
        ]

        registry = CorporateActionsRegistry.from_list(actions)

        assert len(registry) == 2
        assert registry.get_actions_for_symbol("AAPL") == actions

    def test_registry_filter_by_date_range(self):
        """Registry should filter actions by date range."""
        from vectorforge.portfolio.corporate_actions import (
            CorporateActionsRegistry,
            Split,
        )

        actions = [
            Split(symbol="AAPL", ex_date=date(2020, 1, 15), ratio=2.0),
            Split(symbol="AAPL", ex_date=date(2020, 8, 31), ratio=4.0),
            Split(symbol="AAPL", ex_date=date(2021, 6, 15), ratio=2.0),
        ]

        registry = CorporateActionsRegistry.from_list(actions)

        # Filter to 2020 only
        filtered = registry.filter_date_range(
            start=date(2020, 1, 1),
            end=date(2020, 12, 31),
        )

        assert len(filtered) == 2

    def test_registry_get_actions_between_dates(self):
        """Registry should return actions between two dates."""
        from vectorforge.portfolio.corporate_actions import (
            CorporateActionsRegistry,
            Split,
        )

        actions = [
            Split(symbol="AAPL", ex_date=date(2020, 1, 15), ratio=2.0),
            Split(symbol="AAPL", ex_date=date(2020, 8, 31), ratio=4.0),
        ]

        registry = CorporateActionsRegistry.from_list(actions)

        # Get actions in August 2020
        aug_actions = registry.get_actions_between(
            date(2020, 8, 1),
            date(2020, 8, 31),
        )

        assert len(aug_actions) == 1
        assert aug_actions[0].ratio == 4.0


class TestMergerAndSpinoff:
    """Contract tests for merger and spinoff handling."""

    def test_merger_action(self):
        """Merger action should have acquirer, target, and ratio."""
        from vectorforge.portfolio.corporate_actions import Merger

        merger = Merger(
            symbol="TARGET",
            ex_date=date(2023, 6, 1),
            acquirer="ACQUIRER",
            ratio=0.5,  # 0.5 acquirer shares per target share
            cash_per_share=10.0,  # Plus $10 cash
        )

        assert merger.symbol == "TARGET"
        assert merger.acquirer == "ACQUIRER"
        assert merger.ratio == 0.5
        assert merger.cash_per_share == 10.0

    def test_spinoff_action(self):
        """Spinoff action should have parent, spinoff symbol, and ratio."""
        from vectorforge.portfolio.corporate_actions import Spinoff

        spinoff = Spinoff(
            symbol="PARENT",
            ex_date=date(2023, 7, 1),
            spinoff_symbol="NEWCO",
            ratio=0.25,  # 0.25 NEWCO shares per PARENT share
        )

        assert spinoff.symbol == "PARENT"
        assert spinoff.spinoff_symbol == "NEWCO"
        assert spinoff.ratio == 0.25
