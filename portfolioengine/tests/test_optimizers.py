"""Tests for the portfolio optimizers."""

import numpy as np
import pytest

from portfolioengine import (
    Constraints,
    EqualWeight,
    HierarchicalRiskParity,
    InverseVolatility,
    MaxSharpe,
    MinimumVariance,
    RiskParity,
    portfolio_volatility,
    risk_contributions,
)


def two_asset_cov(s1=0.20, s2=0.30, rho=0.20):
    c12 = rho * s1 * s2
    return np.array([[s1**2, c12], [c12, s2**2]])


@pytest.fixture
def random_cov():
    rng = np.random.default_rng(11)
    a = rng.standard_normal((8, 8)) * 0.1
    cov = a @ a.T + 0.05 * np.eye(8)
    return cov


def test_equal_weight_is_one_over_n(random_cov):
    w = EqualWeight().allocate(None, random_cov)
    assert w == pytest.approx(np.full(8, 1.0 / 8))


def test_inverse_volatility_proportional_to_inverse_sigma():
    cov = np.diag([0.04, 0.16])  # vols 0.2 and 0.4 -> weights 2:1
    w = InverseVolatility().allocate(None, cov)
    assert w == pytest.approx([2.0 / 3.0, 1.0 / 3.0])


def test_inverse_volatility_respects_max_weight():
    cov = np.diag([0.0001, 0.04, 0.04])  # tiny vol would dominate
    w = InverseVolatility(Constraints(max_weight=0.5)).allocate(None, cov)
    assert w.max() <= 0.5 + 1e-12
    assert w.sum() == pytest.approx(1.0)


def test_min_variance_two_asset_closed_form():
    s1, s2, rho = 0.20, 0.30, 0.20
    cov = two_asset_cov(s1, s2, rho)
    c12 = rho * s1 * s2
    w1_expected = (s2**2 - c12) / (s1**2 + s2**2 - 2 * c12)
    w = MinimumVariance().allocate(None, cov)
    assert w[0] == pytest.approx(w1_expected, abs=1e-6)
    assert w.sum() == pytest.approx(1.0)


def test_min_variance_has_lowest_vol_among_methods(random_cov):
    w_mv = MinimumVariance().allocate(None, random_cov)
    vol_mv = portfolio_volatility(w_mv, random_cov)
    for opt in (EqualWeight(), InverseVolatility(), RiskParity()):
        w = opt.allocate(None, random_cov)
        assert vol_mv <= portfolio_volatility(w, random_cov) + 1e-10


def test_min_variance_long_only_and_capped(random_cov):
    w = MinimumVariance(Constraints(max_weight=0.25)).allocate(None, random_cov)
    assert (w >= -1e-9).all()
    assert w.max() <= 0.25 + 1e-6
    assert w.sum() == pytest.approx(1.0)


def test_max_sharpe_two_asset_uncorrelated_closed_form():
    # Unconstrained tangency (rf=0): w proportional to cov^{-1} mu.
    mu = np.array([0.10, 0.06])
    cov = np.diag([0.04, 0.02])
    raw = np.linalg.solve(cov, mu)
    expected = raw / raw.sum()
    w = MaxSharpe().allocate(mu, cov)
    assert w == pytest.approx(expected, abs=1e-5)


def test_max_sharpe_beats_other_methods_in_sample(random_cov):
    rng = np.random.default_rng(5)
    mu = rng.uniform(0.02, 0.12, size=8)
    w_ms = MaxSharpe().allocate(mu, random_cov)
    sharpe_ms = (w_ms @ mu) / portfolio_volatility(w_ms, random_cov)
    for opt in (EqualWeight(), MinimumVariance(), RiskParity()):
        w = opt.allocate(mu, random_cov)
        sharpe = (w @ mu) / portfolio_volatility(w, random_cov)
        assert sharpe_ms >= sharpe - 1e-8


def test_max_sharpe_requires_mu(random_cov):
    with pytest.raises(ValueError, match="expected returns"):
        MaxSharpe().allocate(None, random_cov)


def test_max_sharpe_respects_constraints(random_cov):
    rng = np.random.default_rng(6)
    mu = rng.uniform(0.02, 0.12, size=8)
    w = MaxSharpe(Constraints(max_weight=0.20)).allocate(mu, random_cov)
    assert (w >= -1e-9).all()
    assert w.max() <= 0.20 + 1e-6
    assert w.sum() == pytest.approx(1.0)


def test_risk_parity_contributions_equal_within_1e6(random_cov):
    w = RiskParity().allocate(None, random_cov)
    rc = risk_contributions(w, random_cov)
    assert np.abs(rc - 1.0 / 8).max() < 1e-6
    assert w.sum() == pytest.approx(1.0)
    assert (w > 0).all()


def test_risk_parity_diagonal_cov_is_inverse_volatility():
    cov = np.diag([0.04, 0.09, 0.16])
    w_rp = RiskParity().allocate(None, cov)
    w_iv = InverseVolatility().allocate(None, cov)
    assert w_rp == pytest.approx(w_iv, abs=1e-8)


def test_risk_parity_custom_budgets(random_cov):
    budgets = np.array([3.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    w = RiskParity(budgets=budgets).allocate(None, random_cov)
    rc = risk_contributions(w, random_cov)
    assert rc == pytest.approx(budgets / budgets.sum(), abs=1e-6)


def test_risk_parity_two_assets_known_solution():
    # For 2 assets, ERC weights are inverse-vol regardless of correlation.
    cov = two_asset_cov(0.2, 0.4, 0.5)
    w = RiskParity().allocate(None, cov)
    assert w == pytest.approx([2.0 / 3.0, 1.0 / 3.0], abs=1e-8)


def test_hrp_weights_sum_to_one_and_positive(random_cov):
    w = HierarchicalRiskParity().allocate(None, random_cov)
    assert w.sum() == pytest.approx(1.0)
    assert (w > 0).all()


def test_hrp_two_assets_inverse_variance_split():
    cov = np.diag([0.04, 0.16])
    w = HierarchicalRiskParity().allocate(None, cov)
    # alpha = 1 - v1 / (v1 + v2) = 1 - 0.04/0.20 = 0.8
    assert w == pytest.approx([0.8, 0.2], abs=1e-12)


def test_hrp_is_deterministic(random_cov):
    a = HierarchicalRiskParity().allocate(None, random_cov)
    b = HierarchicalRiskParity().allocate(None, random_cov)
    np.testing.assert_array_equal(a, b)


def test_hrp_single_asset():
    w = HierarchicalRiskParity().allocate(None, np.array([[0.04]]))
    assert w == pytest.approx([1.0])


def test_hrp_overweights_diversifier():
    # Two highly correlated risky assets + one uncorrelated low-vol asset.
    corr = np.array([[1.0, 0.9, 0.0], [0.9, 1.0, 0.0], [0.0, 0.0, 1.0]])
    vols = np.array([0.2, 0.2, 0.2])
    cov = corr * np.outer(vols, vols)
    w = HierarchicalRiskParity().allocate(None, cov)
    assert w[2] > w[0]
    assert w[2] > w[1]


def test_hrp_respects_max_weight(random_cov):
    w = HierarchicalRiskParity(Constraints(max_weight=0.2)).allocate(None, random_cov)
    assert w.max() <= 0.2 + 1e-12
    assert w.sum() == pytest.approx(1.0)


def test_optimizers_reject_dimension_mismatch(random_cov):
    with pytest.raises(ValueError):
        MaxSharpe().allocate(np.array([0.1, 0.2]), random_cov)


def test_optimizers_reject_nan_cov():
    cov = np.array([[0.04, np.nan], [np.nan, 0.09]])
    with pytest.raises(ValueError):
        MinimumVariance().allocate(None, cov)


def test_risk_contributions_sum_to_one(random_cov):
    w = EqualWeight().allocate(None, random_cov)
    rc = risk_contributions(w, random_cov)
    assert rc.sum() == pytest.approx(1.0)


def test_sector_caps_enforced_by_slsqp_optimizers():
    rng = np.random.default_rng(21)
    n = 6
    a = rng.standard_normal((n, n)) * 0.1
    cov = a @ a.T + 0.05 * np.eye(n)
    mu = rng.uniform(0.04, 0.12, size=n)
    sectors = ("tech", "tech", "tech", "fin", "fin", "energy")
    cons = Constraints(sector_caps={"tech": 0.30}, sectors=sectors)
    for opt in (MinimumVariance(cons), MaxSharpe(cons)):
        w = opt.allocate(mu, cov)
        tech = sum(w[i] for i, s in enumerate(sectors) if s == "tech")
        assert tech <= 0.30 + 1e-6, opt.name
        assert w.sum() == pytest.approx(1.0)
