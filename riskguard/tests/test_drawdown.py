"""Tests for drawdown tracking and the circuit breaker."""

import numpy as np
import pandas as pd
import pytest

from riskguard import (
    DrawdownCircuitBreaker,
    drawdown_curve,
    equity_from_returns,
    max_drawdown,
)


def test_drawdown_curve_known_values():
    equity = pd.Series([100.0, 110.0, 99.0, 121.0, 60.5])
    dd = drawdown_curve(equity)
    assert dd.iloc[0] == pytest.approx(0.0)
    assert dd.iloc[1] == pytest.approx(0.0)
    assert dd.iloc[2] == pytest.approx(0.10)  # 99 vs peak 110
    assert dd.iloc[3] == pytest.approx(0.0)  # new peak
    assert dd.iloc[4] == pytest.approx(0.50)  # 60.5 vs peak 121


def test_max_drawdown_known_value():
    equity = pd.Series([100.0, 120.0, 90.0, 130.0, 117.0])
    assert max_drawdown(equity) == pytest.approx(0.25)  # 90 vs 120


def test_max_drawdown_monotonic_equity_is_zero():
    equity = pd.Series([100.0, 101.0, 102.0, 105.0])
    assert max_drawdown(equity) == pytest.approx(0.0)


def test_equity_from_returns_compounds():
    r = pd.Series([0.10, -0.05, 0.02])
    eq = equity_from_returns(r, initial=100.0)
    assert eq.iloc[-1] == pytest.approx(100.0 * 1.10 * 0.95 * 1.02)


def test_breaker_step_triggers_on_threshold():
    breaker = DrawdownCircuitBreaker(max_drawdown=0.15, reentry_drawdown=0.05)
    assert breaker.step(100.0) == 1.0
    assert breaker.step(90.0) == 1.0  # 10% DD, below threshold
    assert breaker.step(84.0) == 0.0  # 16% DD -> triggered
    assert breaker.n_triggers == 1


def test_breaker_step_reenters_on_recovery():
    breaker = DrawdownCircuitBreaker(max_drawdown=0.15, reentry_drawdown=0.05)
    breaker.step(100.0)
    assert breaker.step(80.0) == 0.0  # 20% DD -> off
    assert breaker.step(90.0) == 0.0  # 10% DD, still below re-entry
    assert breaker.step(96.0) == 1.0  # 4% DD -> back on


def test_breaker_reduced_exposure_mode():
    breaker = DrawdownCircuitBreaker(
        max_drawdown=0.10, reentry_drawdown=0.02, reduced_exposure=0.5
    )
    breaker.step(100.0)
    assert breaker.step(85.0) == 0.5


def test_breaker_apply_no_lookahead():
    # Crash happens at t=2; exposure can only react from t=3 onwards.
    r = pd.Series([0.0, 0.0, -0.30, -0.10, 0.0])
    breaker = DrawdownCircuitBreaker(max_drawdown=0.15, reentry_drawdown=0.05)
    result = breaker.apply(r)
    assert result.exposure.iloc[2] == 1.0  # takes the crash hit
    assert result.exposure.iloc[3] == 0.0  # cut next period
    assert result.filtered_returns.iloc[3] == 0.0


def test_breaker_apply_reduces_max_drawdown():
    rng = np.random.default_rng(3)
    n = 1000
    mu = np.full(n, 0.0008)
    mu[400:450] = -0.02  # crash
    r = pd.Series(mu + 0.01 * rng.standard_normal(n))
    breaker = DrawdownCircuitBreaker(max_drawdown=0.10, reentry_drawdown=0.03)
    result = breaker.apply(r)
    raw_dd = max_drawdown(equity_from_returns(r))
    prot_dd = max_drawdown(result.protected_equity)
    assert prot_dd < raw_dd
    assert result.n_triggers >= 1


def test_breaker_apply_recovers_exposure():
    n = 60
    r = pd.Series([0.0] * 10 + [-0.05] * 5 + [0.03] * 20 + [0.001] * (n - 35))
    breaker = DrawdownCircuitBreaker(max_drawdown=0.15, reentry_drawdown=0.05)
    result = breaker.apply(r)
    assert result.exposure.iloc[16] == 0.0  # off after crash
    assert result.exposure.iloc[-1] == 1.0  # back on after recovery


def test_breaker_apply_is_stateless_across_calls():
    r = pd.Series([0.0, -0.20, 0.0, 0.0])
    breaker = DrawdownCircuitBreaker(max_drawdown=0.15, reentry_drawdown=0.05)
    first = breaker.apply(r)
    second = breaker.apply(r)
    pd.testing.assert_series_equal(first.exposure, second.exposure)
    assert first.n_triggers == second.n_triggers == 1


def test_breaker_never_triggers_in_bull_market():
    rng = np.random.default_rng(4)
    r = pd.Series(0.002 + 0.002 * rng.standard_normal(500))
    breaker = DrawdownCircuitBreaker(max_drawdown=0.10, reentry_drawdown=0.03)
    result = breaker.apply(r)
    assert result.n_triggers == 0
    assert (result.exposure == 1.0).all()
    pd.testing.assert_series_equal(result.filtered_returns, r * 1.0)


def test_breaker_constructor_validation():
    with pytest.raises(ValueError):
        DrawdownCircuitBreaker(max_drawdown=0.0)
    with pytest.raises(ValueError):
        DrawdownCircuitBreaker(max_drawdown=0.1, reentry_drawdown=0.2)
    with pytest.raises(ValueError):
        DrawdownCircuitBreaker(reduced_exposure=1.0)


def test_breaker_step_rejects_nonpositive_equity():
    breaker = DrawdownCircuitBreaker()
    with pytest.raises(ValueError):
        breaker.step(0.0)
