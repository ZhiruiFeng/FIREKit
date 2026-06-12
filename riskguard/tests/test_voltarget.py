"""Tests for volatility targeting."""

import numpy as np
import pandas as pd
import pytest

from riskguard import VolatilityTargeter, annualized_volatility, realized_vol


@pytest.fixture
def noisy_returns():
    rng = np.random.default_rng(42)
    n = 2000
    # Two vol regimes: 10% then 30% annualized.
    vol = np.where(np.arange(n) < n // 2, 0.10, 0.30) / np.sqrt(252)
    return pd.Series(vol * rng.standard_normal(n), index=pd.RangeIndex(n))


def test_realized_vol_constant_alternating_series():
    # +1% / -1% alternating: rolling std (ddof=1, window=2) == sqrt(2)*0.01
    r = pd.Series([0.01, -0.01] * 10)
    rv = realized_vol(r, window=2)
    expected = np.sqrt(2) * 0.01 * np.sqrt(252)
    assert rv.iloc[1] == pytest.approx(expected)
    assert rv.iloc[-1] == pytest.approx(expected)


def test_realized_vol_window_validation():
    with pytest.raises(ValueError):
        realized_vol(pd.Series([0.01, 0.02]), window=1)


def test_exposure_is_target_over_realized_vol():
    r = pd.Series([0.01, -0.01] * 50)
    targeter = VolatilityTargeter(target_vol=0.10, window=2, max_leverage=10.0)
    exposure = targeter.compute_exposure(r)
    realized = np.sqrt(2) * 0.01 * np.sqrt(252)
    # After the window fills (and the 1-period lag), exposure = target / vol.
    assert exposure.iloc[5] == pytest.approx(0.10 / realized)


def test_exposure_is_lagged_no_lookahead():
    rng = np.random.default_rng(1)
    r = pd.Series(0.01 * rng.standard_normal(100))
    targeter = VolatilityTargeter(target_vol=0.10, window=10, max_leverage=5.0)
    exposure = targeter.compute_exposure(r)
    vol = realized_vol(r, window=10)
    # exposure[t] must depend on vol[t-1], not vol[t]
    t = 50
    assert exposure.iloc[t] == pytest.approx(min(0.10 / vol.iloc[t - 1], 5.0))


def test_exposure_clipped_at_max_leverage():
    r = pd.Series([0.0001, -0.0001] * 50)  # tiny vol -> huge raw exposure
    targeter = VolatilityTargeter(target_vol=0.20, window=2, max_leverage=2.0)
    exposure = targeter.compute_exposure(r)
    assert exposure.iloc[10] == pytest.approx(2.0)
    assert (exposure <= 2.0).all()


def test_initial_exposure_fills_warmup():
    r = pd.Series([0.01, -0.01] * 10)
    targeter = VolatilityTargeter(target_vol=0.10, window=5, initial_exposure=1.0)
    exposure = targeter.compute_exposure(r)
    # First `window` periods have no lagged vol estimate.
    assert (exposure.iloc[:5] == 1.0).all()


def test_apply_scales_returns():
    rng = np.random.default_rng(2)
    r = pd.Series(0.01 * rng.standard_normal(300))
    targeter = VolatilityTargeter(target_vol=0.10, window=20)
    result = targeter.apply(r)
    pd.testing.assert_series_equal(result.scaled_returns, result.exposure * r)


def test_vol_targeting_brings_vol_near_target(noisy_returns):
    targeter = VolatilityTargeter(target_vol=0.10, window=20, max_leverage=3.0)
    result = targeter.apply(noisy_returns)
    # Drop warmup, then realized vol of scaled series should be near 10%.
    scaled = result.scaled_returns.iloc[50:]
    assert annualized_volatility(scaled) == pytest.approx(0.10, rel=0.15)


def test_vol_targeting_cuts_high_vol_regime(noisy_returns):
    targeter = VolatilityTargeter(target_vol=0.10, window=20, max_leverage=3.0)
    exposure = targeter.compute_exposure(noisy_returns)
    low_regime = exposure.iloc[200:900].mean()
    high_regime = exposure.iloc[1200:1900].mean()
    assert high_regime < 0.5 * low_regime


def test_constructor_validation():
    with pytest.raises(ValueError):
        VolatilityTargeter(target_vol=0.0)
    with pytest.raises(ValueError):
        VolatilityTargeter(max_leverage=0.0)
    with pytest.raises(ValueError):
        VolatilityTargeter(min_leverage=3.0, max_leverage=2.0)
