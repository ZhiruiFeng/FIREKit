"""Tests for Kelly criterion sizing."""

import numpy as np
import pytest

from riskguard import (
    fractional_kelly,
    kelly_binary,
    kelly_from_moments,
    kelly_multi_asset,
)


def test_kelly_binary_known_value():
    # p=0.6, b=2 -> (0.6*2 - 0.4) / 2 = 0.4
    assert kelly_binary(0.6, 2.0) == pytest.approx(0.4)


def test_kelly_binary_even_odds_coin_flip_is_zero():
    assert kelly_binary(0.5, 1.0) == pytest.approx(0.0)


def test_kelly_binary_negative_edge_is_negative():
    assert kelly_binary(0.4, 1.0) < 0.0


def test_kelly_binary_certain_win_is_one():
    assert kelly_binary(1.0, 1.0) == pytest.approx(1.0)


def test_kelly_binary_classic_example():
    # p=0.55, b=1 -> 2p - 1 = 0.10
    assert kelly_binary(0.55, 1.0) == pytest.approx(0.10)


def test_kelly_binary_validates_inputs():
    with pytest.raises(ValueError):
        kelly_binary(1.5, 1.0)
    with pytest.raises(ValueError):
        kelly_binary(-0.1, 1.0)
    with pytest.raises(ValueError):
        kelly_binary(0.5, 0.0)


def test_kelly_from_moments_known_value():
    # mu=0.10, sigma=0.20 -> 0.10 / 0.04 = 2.5
    assert kelly_from_moments(0.10, 0.20**2) == pytest.approx(2.5)


def test_kelly_from_moments_negative_mu():
    assert kelly_from_moments(-0.05, 0.04) == pytest.approx(-1.25)


def test_kelly_from_moments_rejects_bad_variance():
    with pytest.raises(ValueError):
        kelly_from_moments(0.1, 0.0)


def test_fractional_kelly_scales_linearly():
    assert fractional_kelly(0.4, 0.5) == pytest.approx(0.2)
    assert fractional_kelly(0.4, 0.25) == pytest.approx(0.1)
    assert fractional_kelly(0.4, 1.0) == pytest.approx(0.4)


def test_fractional_kelly_rejects_negative_fraction():
    with pytest.raises(ValueError):
        fractional_kelly(0.4, -0.5)


def test_kelly_multi_asset_diagonal_cov_matches_univariate():
    mu = np.array([0.08, 0.045])
    cov = np.diag([0.04, 0.09])
    w = kelly_multi_asset(mu, cov)
    # f_i = mu_i / sigma_i^2 for independent assets
    assert w == pytest.approx([2.0, 0.5])


def test_kelly_multi_asset_fraction_applied():
    mu = np.array([0.08, 0.045])
    cov = np.diag([0.04, 0.09])
    w = kelly_multi_asset(mu, cov, fraction=0.5)
    assert w == pytest.approx([1.0, 0.25])


def test_kelly_multi_asset_clipping():
    mu = np.array([0.08, 0.045])
    cov = np.diag([0.04, 0.09])
    w = kelly_multi_asset(mu, cov, max_abs_weight=1.0)
    assert w == pytest.approx([1.0, 0.5])


def test_kelly_multi_asset_solves_linear_system():
    rng = np.random.default_rng(0)
    a = rng.standard_normal((4, 4))
    cov = a @ a.T + 4.0 * np.eye(4)
    mu = rng.standard_normal(4) * 0.05
    w = kelly_multi_asset(mu, cov)
    np.testing.assert_allclose(cov @ w, mu, atol=1e-10)


def test_kelly_multi_asset_singular_cov_uses_pinv():
    cov = np.array([[0.04, 0.04], [0.04, 0.04]])  # rank 1
    mu = np.array([0.02, 0.02])
    w = kelly_multi_asset(mu, cov)
    assert np.all(np.isfinite(w))


def test_kelly_multi_asset_shape_validation():
    with pytest.raises(ValueError):
        kelly_multi_asset(np.array([0.1, 0.2]), np.eye(3))
    with pytest.raises(ValueError):
        kelly_multi_asset(np.array([0.1]), np.array([0.1, 0.2]))
