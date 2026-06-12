"""Tests for the factor library (exact formula checks on the tiny panel)."""

import numpy as np
import pytest

from alphalab.factors import (
    Alpha012,
    Alpha101,
    AmihudIlliquidity,
    High52WeekDistance,
    LowVolatility,
    Momentum,
    OvernightGap,
    PriceRange,
    ShortTermReversal,
    VolumeMomentum,
    default_factors,
)
from alphalab.panel import Panel


def test_default_factors_count_and_unique_names() -> None:
    factors = default_factors()
    assert len(factors) >= 15
    names = [f.name for f in factors]
    assert len(names) == len(set(names))
    categories = {f.category for f in factors}
    assert {"momentum", "reversal", "volatility", "volume", "alpha101"} <= categories
    assert sum(1 for f in factors if f.category == "alpha101") == 5


def test_momentum_exact(tiny_panel: Panel) -> None:
    values = Momentum(window=5).compute(tiny_panel)
    assert values.iloc[5]["AAA"] == pytest.approx(0.5)  # 10 -> 15
    assert values.iloc[5]["BBB"] == pytest.approx(-0.25)  # 20 -> 15
    assert np.isnan(values.iloc[4]["AAA"])  # not enough history


def test_momentum_skip_exact(tiny_panel: Panel) -> None:
    values = Momentum(window=3, skip=2).compute(tiny_panel)
    # close.shift(2)/close.shift(5) - 1 at t=5: close[3]/close[0] - 1
    assert values.iloc[5]["AAA"] == pytest.approx(13.0 / 10.0 - 1.0)
    assert Momentum(231, skip=21).name == "momentum_231d_skip21d"


def test_reversal_is_negative_momentum(tiny_panel: Panel) -> None:
    rev = ShortTermReversal(5).compute(tiny_panel)
    mom = Momentum(5).compute(tiny_panel)
    assert rev.iloc[5]["AAA"] == pytest.approx(-mom.iloc[5]["AAA"])


def test_low_volatility_sign_and_value(tiny_panel: Panel) -> None:
    values = LowVolatility(window=3).compute(tiny_panel)
    rets = tiny_panel.close.pct_change()
    expected = -rets["AAA"].iloc[3:6].std()
    assert values.iloc[5]["AAA"] == pytest.approx(expected)
    assert (values.dropna(how="all").stack() <= 0).all()


def test_price_range_exact(tiny_panel: Panel) -> None:
    values = PriceRange(window=3).compute(tiny_panel)
    # AAA at t=2: max(high[0:3]) = 13, min(low[0:3]) = 8.5, close = 12
    assert values.iloc[2]["AAA"] == pytest.approx(-(13.0 - 8.5) / 12.0)


def test_volume_momentum_exact(tiny_panel: Panel) -> None:
    values = VolumeMomentum(fast=2, slow=4).compute(tiny_panel)
    fast = tiny_panel.volume["AAA"].iloc[4:6].mean()
    slow = tiny_panel.volume["AAA"].iloc[2:6].mean()
    assert values.iloc[5]["AAA"] == pytest.approx(fast / slow - 1.0)
    with pytest.raises(ValueError):
        VolumeMomentum(fast=20, slow=5)


def test_amihud_exact(tiny_panel: Panel) -> None:
    values = AmihudIlliquidity(window=2).compute(tiny_panel)
    rets = tiny_panel.close["AAA"].pct_change()
    dollar = tiny_panel.close["AAA"] * tiny_panel.volume["AAA"]
    expected = (rets.abs() / dollar * 1e6).iloc[4:6].mean()
    assert values.iloc[5]["AAA"] == pytest.approx(expected)


def test_high_52w_distance_bounds(tiny_panel: Panel) -> None:
    values = High52WeekDistance(window=4).compute(tiny_panel)
    valid = values.dropna(how="all").stack()
    assert (valid <= 0).all()  # close can never exceed trailing high
    # AAA at t=5: close 15, trailing high max(high[2:6]) = 16
    assert values.iloc[5]["AAA"] == pytest.approx(15.0 / 16.0 - 1.0)


def test_overnight_gap_exact(tiny_panel: Panel) -> None:
    values = OvernightGap(window=1).compute(tiny_panel)
    expected = tiny_panel.open.iloc[3]["AAA"] / tiny_panel.close.iloc[2]["AAA"] - 1.0
    assert values.iloc[3]["AAA"] == pytest.approx(expected)


def test_alpha101_exact(tiny_panel: Panel) -> None:
    values = Alpha101().compute(tiny_panel)
    o = tiny_panel.open.iloc[3]["BBB"]
    h = tiny_panel.high.iloc[3]["BBB"]
    low = tiny_panel.low.iloc[3]["BBB"]
    c = tiny_panel.close.iloc[3]["BBB"]
    assert values.iloc[3]["BBB"] == pytest.approx((c - o) / ((h - low) + 0.001))


def test_alpha012_exact(tiny_panel: Panel) -> None:
    values = Alpha012().compute(tiny_panel)
    # AAA t=1: volume delta +10 -> sign +1; close delta +1 -> value -1
    assert values.iloc[1]["AAA"] == pytest.approx(-1.0)
    # BBB t=1: volume delta -10 -> sign -1; close delta -1 -> -(-1)*-1 = -1... compute:
    # sign(-10) * (-(-1.0)) = -1 * 1.0 = -1.0
    assert values.iloc[1]["BBB"] == pytest.approx(-1.0)


def test_all_factors_compute_on_synthetic(synthetic_panel: Panel) -> None:
    # rolling correlations of cross-sectional ranks are undefined whenever a
    # symbol's rank is constant in the window, so those factors are sparser
    min_coverage = {"alpha_003": 0.10}
    for factor in default_factors():
        values = factor.compute(synthetic_panel)
        assert values.shape == synthetic_panel.shape, factor.name
        assert not np.isinf(values.to_numpy()).any(), factor.name
        # every factor must produce real values for the last quarter of the sample
        tail = values.iloc[-80:]
        coverage = tail.notna().mean().mean()
        assert coverage > min_coverage.get(factor.name, 0.5), factor.name


def test_factor_repr_and_call(tiny_panel: Panel) -> None:
    factor = Momentum(5)
    assert "momentum_5d" in repr(factor)
    a = factor(tiny_panel)
    b = factor.compute(tiny_panel)
    assert a.equals(b)


def test_momentum_validates_args() -> None:
    with pytest.raises(ValueError):
        Momentum(0)
    with pytest.raises(ValueError):
        Momentum(5, skip=-1)
