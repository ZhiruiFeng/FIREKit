"""Slippage and commission model arithmetic."""

import pytest

from executioncore.costs import (
    FixedBpsSlippage,
    NoSlippage,
    PercentageCommission,
    PerShareCommission,
    TieredCommission,
    VolumeImpactSlippage,
)
from executioncore.market import Bar
from executioncore.orders import OrderSide

BAR = Bar(index=0, open=100.0, high=101.0, low=99.0, close=100.5, volume=10_000.0)


class TestSlippage:
    def test_no_slippage_identity(self):
        assert NoSlippage().fill_price(100.0, OrderSide.BUY, 500, BAR) == 100.0

    def test_fixed_bps_buy_pays_up(self):
        model = FixedBpsSlippage(2.0)
        assert model.fill_price(100.0, OrderSide.BUY, 500, BAR) == pytest.approx(100.02)

    def test_fixed_bps_sell_receives_less(self):
        model = FixedBpsSlippage(2.0)
        assert model.fill_price(100.0, OrderSide.SELL, 500, BAR) == pytest.approx(99.98)

    def test_volume_impact_square_root(self):
        # 25 bps at full-bar participation; 2500/10000 -> sqrt(0.25) = 0.5 -> 12.5 bps
        model = VolumeImpactSlippage(impact_bps=25.0)
        assert model.slippage_fraction(OrderSide.BUY, 2_500, BAR) == pytest.approx(12.5e-4)
        assert model.fill_price(100.0, OrderSide.BUY, 2_500, BAR) == pytest.approx(100.125)

    def test_volume_impact_scales_with_size(self):
        model = VolumeImpactSlippage(impact_bps=25.0)
        small = model.slippage_fraction(OrderSide.BUY, 100, BAR)
        large = model.slippage_fraction(OrderSide.BUY, 10_000, BAR)
        assert large == pytest.approx(25e-4)
        assert small < large

    def test_negative_bps_rejected(self):
        with pytest.raises(ValueError):
            FixedBpsSlippage(-1.0)


class TestCommission:
    def test_per_share(self):
        assert PerShareCommission(0.005).commission(1_000, 50.0) == pytest.approx(5.0)

    def test_per_share_minimum_applies(self):
        assert PerShareCommission(0.005, minimum=1.0).commission(10, 50.0) == pytest.approx(1.0)

    def test_percentage_of_notional(self):
        assert PercentageCommission(0.001).commission(200, 50.0) == pytest.approx(10.0)

    def test_tiered_rates_by_fill_size(self):
        tiers = [(300.0, 0.0065), (3_000.0, 0.005), (float("inf"), 0.0035)]
        model = TieredCommission(tiers)
        assert model.commission(200, 50.0) == pytest.approx(200 * 0.0065)
        assert model.commission(300, 50.0) == pytest.approx(300 * 0.0065)  # boundary inclusive
        assert model.commission(1_000, 50.0) == pytest.approx(1_000 * 0.005)
        assert model.commission(10_000, 50.0) == pytest.approx(10_000 * 0.0035)

    def test_tiered_requires_sorted_thresholds(self):
        with pytest.raises(ValueError):
            TieredCommission([(1_000.0, 0.005), (300.0, 0.0065)])
