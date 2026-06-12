"""Tests for the efficient frontier, backtest harness, constraints, synthetic data."""

import numpy as np
import pandas as pd
import pytest

from portfolioengine import (
    Constraints,
    EqualWeight,
    HierarchicalRiskParity,
    MaxSharpe,
    MinimumVariance,
    RiskParity,
    apply_weight_cap,
    efficient_frontier,
    make_universe,
    portfolio_volatility,
    run_backtest,
    sample_cov,
)


@pytest.fixture(scope="module")
def universe():
    return make_universe(n_assets=10, n_days=600, n_sectors=3, seed=42)


@pytest.fixture(scope="module")
def mu_cov(universe):
    returns = universe.returns
    mu = returns.to_numpy().mean(axis=0) * 252.0
    cov = sample_cov(returns) * 252.0
    return mu, cov


# --- Efficient frontier ----------------------------------------------------


def test_frontier_returns_are_increasing(mu_cov):
    mu, cov = mu_cov
    points = efficient_frontier(mu, cov, n_points=15)
    rets = [p.expected_return for p in points]
    assert all(b >= a - 1e-9 for a, b in zip(rets, rets[1:], strict=False))


def test_frontier_vol_increasing_along_efficient_branch(mu_cov):
    mu, cov = mu_cov
    points = efficient_frontier(mu, cov, n_points=15)
    vols = [p.volatility for p in points]
    assert all(b >= a - 1e-7 for a, b in zip(vols, vols[1:], strict=False))


def test_frontier_starts_at_min_variance(mu_cov):
    mu, cov = mu_cov
    points = efficient_frontier(mu, cov, n_points=10)
    w_mv = MinimumVariance().allocate(None, cov)
    assert points[0].volatility == pytest.approx(
        portfolio_volatility(w_mv, cov), rel=1e-4
    )


def test_frontier_weights_valid(mu_cov):
    mu, cov = mu_cov
    cons = Constraints(max_weight=0.30)
    points = efficient_frontier(mu, cov, n_points=10, constraints=cons)
    assert len(points) >= 8
    for p in points:
        assert p.weights.sum() == pytest.approx(1.0)
        assert (p.weights >= -1e-8).all()
        assert p.weights.max() <= 0.30 + 1e-6


def test_frontier_max_sharpe_point_dominates(mu_cov):
    mu, cov = mu_cov
    points = efficient_frontier(mu, cov, n_points=20)
    w_ms = MaxSharpe().allocate(mu, cov)
    sharpe_ms = (w_ms @ mu) / portfolio_volatility(w_ms, cov)
    assert all(p.sharpe <= sharpe_ms + 1e-6 for p in points)


def test_frontier_validates_n_points(mu_cov):
    mu, cov = mu_cov
    with pytest.raises(ValueError):
        efficient_frontier(mu, cov, n_points=1)


# --- Backtest harness --------------------------------------------------------


def test_backtest_runs_all_methods(universe):
    optimizers = {
        "EW": EqualWeight(),
        "RP": RiskParity(),
        "HRP": HierarchicalRiskParity(),
    }
    results = run_backtest(universe.returns, optimizers, lookback=126, rebalance_every=21)
    assert set(results) == {"EW", "RP", "HRP"}
    n_expected = len(universe.returns) - 126
    for r in results.values():
        assert len(r.returns) == n_expected
        assert len(r.equity) == n_expected
        assert not r.returns.isna().any()
        assert r.annual_volatility > 0
        assert 0 <= r.max_drawdown < 1
        assert r.avg_turnover >= 0


def test_backtest_equal_weight_matches_manual(universe):
    returns = universe.returns
    results = run_backtest(returns, {"EW": EqualWeight()}, lookback=126, rebalance_every=21)
    manual = returns.iloc[126:].mean(axis=1)
    np.testing.assert_allclose(
        results["EW"].returns.to_numpy(), manual.to_numpy(), atol=1e-12
    )


def test_backtest_equal_weight_has_zero_turnover(universe):
    results = run_backtest(
        universe.returns, {"EW": EqualWeight()}, lookback=126, rebalance_every=21
    )
    assert results["EW"].avg_turnover == pytest.approx(0.0)


def test_backtest_costs_reduce_equity(universe):
    optimizers = {"MS": MaxSharpe()}
    free = run_backtest(universe.returns, optimizers, lookback=126, cost_bps=0.0)
    costly = run_backtest(universe.returns, optimizers, lookback=126, cost_bps=20.0)
    assert costly["MS"].equity.iloc[-1] < free["MS"].equity.iloc[-1]


def test_backtest_weights_recorded_each_rebalance(universe):
    results = run_backtest(
        universe.returns, {"RP": RiskParity()}, lookback=126, rebalance_every=21
    )
    weights = results["RP"].weights
    expected_rebalances = len(range(126, len(universe.returns), 21))
    assert len(weights) == expected_rebalances
    np.testing.assert_allclose(weights.sum(axis=1).to_numpy(), 1.0, atol=1e-9)


def test_backtest_is_deterministic(universe):
    a = run_backtest(universe.returns, {"HRP": HierarchicalRiskParity()}, lookback=126)
    b = run_backtest(universe.returns, {"HRP": HierarchicalRiskParity()}, lookback=126)
    pd.testing.assert_series_equal(a["HRP"].equity, b["HRP"].equity)


def test_backtest_validation(universe):
    with pytest.raises(ValueError):
        run_backtest(universe.returns, {"EW": EqualWeight()}, lookback=1)
    with pytest.raises(ValueError):
        run_backtest(universe.returns.iloc[:50], {"EW": EqualWeight()}, lookback=126)
    bad = universe.returns.copy()
    bad.iloc[0, 0] = np.nan
    with pytest.raises(ValueError):
        run_backtest(bad, {"EW": EqualWeight()}, lookback=126)


# --- Constraints helpers ------------------------------------------------------


def test_apply_weight_cap_redistributes_proportionally():
    w = np.array([0.6, 0.3, 0.1])
    capped = apply_weight_cap(w, 0.5)
    assert capped.sum() == pytest.approx(1.0)
    assert capped[0] == pytest.approx(0.5)
    # Remaining 0.5 split 3:1 between the uncapped assets.
    assert capped[1] == pytest.approx(0.375)
    assert capped[2] == pytest.approx(0.125)


def test_apply_weight_cap_cascading():
    w = np.array([0.70, 0.28, 0.01, 0.01])
    capped = apply_weight_cap(w, 0.30)
    assert capped.sum() == pytest.approx(1.0)
    assert capped.max() <= 0.30 + 1e-12


def test_apply_weight_cap_infeasible():
    with pytest.raises(ValueError):
        apply_weight_cap(np.array([0.5, 0.5]), 0.3)


def test_constraints_validation():
    with pytest.raises(ValueError):
        Constraints(max_weight=0.0)
    with pytest.raises(ValueError):
        Constraints(sector_caps={"tech": 0.3})  # missing sectors
    cons = Constraints(max_weight=0.05)
    with pytest.raises(ValueError):
        cons.validate_dimension(10)  # 0.05 * 10 < 1


# --- Synthetic universe ----------------------------------------------------------


def test_universe_shape_and_determinism():
    a = make_universe(n_assets=12, n_days=300, n_sectors=4, seed=1)
    b = make_universe(n_assets=12, n_days=300, n_sectors=4, seed=1)
    assert a.returns.shape == (300, 12)
    pd.testing.assert_frame_equal(a.returns, b.returns)
    assert len(set(a.sectors.values())) == 4


def test_universe_block_correlation_structure():
    u = make_universe(n_assets=20, n_days=4000, n_sectors=4, seed=2)
    corr = u.returns.corr().to_numpy()
    labels = u.sector_labels
    same, cross = [], []
    for i in range(20):
        for j in range(i + 1, 20):
            (same if labels[i] == labels[j] else cross).append(corr[i, j])
    assert np.mean(same) > 0.4
    assert np.mean(cross) < 0.25
    assert np.mean(same) > np.mean(cross) + 0.2


def test_universe_cov_is_positive_definite():
    u = make_universe(seed=3)
    eigvals = np.linalg.eigvalsh(u.annual_cov)
    assert eigvals.min() > 0
