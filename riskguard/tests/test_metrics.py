"""Tests for VaR / CVaR risk metrics."""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from riskguard import (
    annualized_volatility,
    cvar_cornish_fisher,
    cvar_gaussian,
    cvar_historical,
    expected_shortfall,
    rolling_cvar,
    rolling_var,
    sharpe_ratio,
    value_at_risk,
    var_cornish_fisher,
    var_gaussian,
    var_historical,
)


@pytest.fixture
def normal_returns():
    rng = np.random.default_rng(123)
    return pd.Series(0.0005 + 0.01 * rng.standard_normal(5000))


def test_var_historical_matches_quantile():
    arr = np.linspace(-0.10, 0.10, 201)  # symmetric grid, step 0.001
    # 5% quantile index = 0.05 * 200 = 10 -> value -0.09 exactly
    assert var_historical(arr, 0.95) == pytest.approx(0.09)


def test_cvar_historical_exceeds_var(normal_returns):
    var = var_historical(normal_returns, 0.95)
    cvar = cvar_historical(normal_returns, 0.95)
    assert cvar > var


def test_cvar_historical_known_value():
    arr = np.linspace(-0.10, 0.10, 201)
    # Tail at/below the 5% quantile (-0.09): values -0.10 .. -0.09, mean -0.095
    assert cvar_historical(arr, 0.95) == pytest.approx(0.095)


def test_var_gaussian_formula():
    rng = np.random.default_rng(7)
    arr = 0.001 + 0.02 * rng.standard_normal(1000)
    mu, sd = arr.mean(), arr.std(ddof=1)
    expected = -(mu + stats.norm.ppf(0.05) * sd)
    assert var_gaussian(arr, 0.95) == pytest.approx(expected)


def test_cvar_gaussian_formula():
    rng = np.random.default_rng(8)
    arr = 0.0 + 0.015 * rng.standard_normal(1000)
    mu, sd = arr.mean(), arr.std(ddof=1)
    z = stats.norm.ppf(0.05)
    expected = -mu + sd * stats.norm.pdf(z) / 0.05
    assert cvar_gaussian(arr, 0.95) == pytest.approx(expected)


def test_cornish_fisher_reduces_to_gaussian_for_symmetric_data():
    # Perfectly symmetric, light-tailed sample -> skew == 0; for an exactly
    # symmetric grid the CF adjustment uses sample kurtosis, so compare with
    # near-normal random data instead.
    rng = np.random.default_rng(9)
    arr = rng.standard_normal(200_000) * 0.01
    assert var_cornish_fisher(arr, 0.95) == pytest.approx(var_gaussian(arr, 0.95), rel=5e-3)


def test_cornish_fisher_var_penalizes_negative_skew():
    rng = np.random.default_rng(10)
    base = rng.standard_normal(20_000) * 0.01
    # Inject occasional large losses -> negative skew, fat left tail.
    crash = np.where(rng.uniform(size=base.size) < 0.01, -0.05, 0.0)
    skewed = base + crash
    assert var_cornish_fisher(skewed, 0.99) > var_gaussian(skewed, 0.99)


def test_cvar_cornish_fisher_matches_gaussian_es_when_normal():
    rng = np.random.default_rng(11)
    arr = rng.standard_normal(300_000) * 0.01
    assert cvar_cornish_fisher(arr, 0.95) == pytest.approx(cvar_gaussian(arr, 0.95), rel=1e-2)


def test_cvar_at_least_var_all_methods(normal_returns):
    for method in ("historical", "gaussian", "cornish_fisher"):
        var = value_at_risk(normal_returns, 0.95, method)
        cvar = expected_shortfall(normal_returns, 0.95, method)
        assert cvar >= var, method


def test_var_increases_with_confidence(normal_returns):
    for method in ("historical", "gaussian", "cornish_fisher"):
        assert value_at_risk(normal_returns, 0.99, method) > value_at_risk(
            normal_returns, 0.95, method
        )


def test_dispatch_rejects_unknown_method(normal_returns):
    with pytest.raises(ValueError):
        value_at_risk(normal_returns, 0.95, "bogus")
    with pytest.raises(ValueError):
        expected_shortfall(normal_returns, 0.95, "bogus")


def test_confidence_validation(normal_returns):
    with pytest.raises(ValueError):
        var_historical(normal_returns, 1.0)
    with pytest.raises(ValueError):
        var_historical(normal_returns, 0.0)


def test_too_few_observations():
    with pytest.raises(ValueError):
        var_historical(np.array([0.01]))


def test_rolling_var_shape_and_window(normal_returns):
    rv = rolling_var(normal_returns, window=100, confidence=0.95)
    assert len(rv) == len(normal_returns)
    assert rv.iloc[:99].isna().all()
    assert rv.iloc[99:].notna().all()
    # Spot check one window against the scalar function.
    window_vals = normal_returns.iloc[0:100]
    assert rv.iloc[99] == pytest.approx(var_historical(window_vals, 0.95))


def test_rolling_cvar_spot_check(normal_returns):
    rc = rolling_cvar(normal_returns, window=100, confidence=0.95, method="gaussian")
    window_vals = normal_returns.iloc[100:200]
    assert rc.iloc[199] == pytest.approx(cvar_gaussian(window_vals, 0.95))


def test_rolling_rejects_small_window(normal_returns):
    with pytest.raises(ValueError):
        rolling_var(normal_returns, window=5)


def test_annualized_volatility():
    rng = np.random.default_rng(12)
    arr = rng.standard_normal(10_000) * 0.01
    expected = arr.std(ddof=1) * np.sqrt(252)
    assert annualized_volatility(arr) == pytest.approx(expected)


def test_sharpe_ratio_zero_for_zero_mean():
    arr = np.array([0.01, -0.01] * 100)
    assert sharpe_ratio(arr) == pytest.approx(0.0)


def test_sharpe_ratio_known_value():
    rng = np.random.default_rng(13)
    arr = 0.0004 + 0.01 * rng.standard_normal(5000)
    expected = arr.mean() / arr.std(ddof=1) * np.sqrt(252)
    assert sharpe_ratio(arr) == pytest.approx(expected)
