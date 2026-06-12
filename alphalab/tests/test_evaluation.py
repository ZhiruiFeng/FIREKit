"""Tests for IC, quantile portfolios, turnover, and factor correlation."""

import numpy as np
import pandas as pd
import pytest

from alphalab.evaluation import (
    ICStats,
    evaluate_factor,
    factor_correlation,
    factor_turnover,
    forward_returns,
    ic_series,
    quantile_returns,
    quantile_spread,
)


def make_wide(values: np.ndarray, n_symbols: int | None = None) -> pd.DataFrame:
    n_days, n_sym = values.shape
    dates = pd.bdate_range("2023-01-02", periods=n_days)
    cols = [f"S{i}" for i in range(n_sym)]
    return pd.DataFrame(values, index=dates, columns=cols)


def test_forward_returns_exact() -> None:
    close = make_wide(np.array([[10.0, 100.0], [11.0, 90.0], [12.1, 81.0]]))
    fwd = forward_returns(close, 1)
    assert fwd.iloc[0, 0] == pytest.approx(0.1)
    assert fwd.iloc[1, 1] == pytest.approx(-0.1)
    assert fwd.iloc[2].isna().all()  # no future data
    with pytest.raises(ValueError):
        forward_returns(close, 0)


def test_ic_is_one_for_perfect_factor() -> None:
    rng = np.random.default_rng(0)
    close = make_wide(np.exp(rng.normal(0, 0.01, size=(40, 10)).cumsum(axis=0)) * 100)
    fwd = forward_returns(close, 1)
    ic = ic_series(fwd.copy(), fwd, method="pearson")
    rank_ic = ic_series(fwd.copy(), fwd, method="spearman")
    assert np.allclose(ic, 1.0)
    assert np.allclose(rank_ic, 1.0)


def test_rank_ic_invariant_to_monotonic_transform() -> None:
    rng = np.random.default_rng(1)
    close = make_wide(np.exp(rng.normal(0, 0.01, size=(30, 12)).cumsum(axis=0)) * 50)
    fwd = forward_returns(close, 1)
    transformed = np.exp(5 * fwd)  # strictly increasing transform
    rank_ic = ic_series(transformed, fwd, method="spearman")
    assert np.allclose(rank_ic, 1.0)


def test_ic_negative_for_inverted_factor() -> None:
    rng = np.random.default_rng(2)
    close = make_wide(np.exp(rng.normal(0, 0.01, size=(30, 10)).cumsum(axis=0)) * 50)
    fwd = forward_returns(close, 1)
    ic = ic_series(-fwd, fwd, method="spearman")
    assert np.allclose(ic, -1.0)


def test_ic_requires_min_cross_section() -> None:
    rng = np.random.default_rng(3)
    close = make_wide(np.exp(rng.normal(0, 0.01, size=(20, 3)).cumsum(axis=0)) * 50)
    fwd = forward_returns(close, 1)
    ic = ic_series(fwd.copy(), fwd)  # only 3 names < MIN_CROSS_SECTION
    assert ic.empty


def test_ic_series_rejects_bad_method() -> None:
    df = make_wide(np.ones((10, 6)))
    with pytest.raises(ValueError):
        ic_series(df, df, method="kendall")


def test_ic_stats_exact() -> None:
    ic = pd.Series([0.1, 0.2, 0.3, 0.2, 0.2])
    stats = ICStats.from_series(ic)
    assert stats.mean == pytest.approx(0.2)
    assert stats.std == pytest.approx(np.std([0.1, 0.2, 0.3, 0.2, 0.2], ddof=1))
    assert stats.ir == pytest.approx(stats.mean / stats.std)
    assert stats.t_stat == pytest.approx(stats.ir * np.sqrt(5))
    assert stats.pct_positive == 1.0
    assert stats.n_obs == 5


def test_ic_stats_empty() -> None:
    stats = ICStats.from_series(pd.Series(dtype=float))
    assert stats.n_obs == 0
    assert np.isnan(stats.mean)


def test_quantile_returns_monotonic_for_perfect_factor() -> None:
    # factor exactly equals forward return -> Q5 must beat Q1 every day
    n_days, n_sym = 12, 10
    fwd_vals = np.tile(np.arange(n_sym, dtype=float), (n_days, 1)) / 100.0
    fwd = make_wide(fwd_vals)
    factor = fwd.copy()
    qret = quantile_returns(factor, fwd, n_quantiles=5)
    means = qret.mean()
    # exact values: Q1 = mean(0.00, 0.01) = 0.005 ... Q5 = 0.085
    assert means["Q1"] == pytest.approx(0.005)
    assert means["Q5"] == pytest.approx(0.085)
    assert (means.diff().dropna() > 0).all()
    spread = quantile_spread(qret)
    assert spread.mean() == pytest.approx(0.08)


def test_quantile_returns_validates_args() -> None:
    df = make_wide(np.ones((5, 6)))
    with pytest.raises(ValueError):
        quantile_returns(df, df, n_quantiles=1)


def test_turnover_zero_for_static_factor() -> None:
    static = make_wide(np.tile(np.arange(10, dtype=float), (15, 1)))
    assert factor_turnover(static) == pytest.approx(0.0)


def test_turnover_one_for_full_rotation() -> None:
    # top-20% bucket (2 of 10) alternates between disjoint name sets each day
    a = np.array([9, 8, 0, 0, 0, 0, 0, 0, 1, 2], dtype=float)
    b = np.array([0, 0, 9, 8, 0, 0, 0, 0, 1, 2], dtype=float)
    values = np.vstack([a if t % 2 == 0 else b for t in range(10)])
    assert factor_turnover(make_wide(values), top_pct=0.2) == pytest.approx(1.0)


def test_turnover_validates_args() -> None:
    df = make_wide(np.ones((5, 6)))
    with pytest.raises(ValueError):
        factor_turnover(df, top_pct=1.5)


def test_factor_correlation_matrix_properties() -> None:
    rng = np.random.default_rng(5)
    base = make_wide(rng.normal(size=(30, 10)))
    factors = {
        "f": base,
        "f_clone": base * 3.0 + 1.0,  # rank-identical
        "f_anti": -base,
        "noise": make_wide(rng.normal(size=(30, 10))),
    }
    corr = factor_correlation(factors)
    assert corr.shape == (4, 4)
    assert np.allclose(np.diag(corr), 1.0)
    assert corr.loc["f", "f_clone"] == pytest.approx(1.0)
    assert corr.loc["f", "f_anti"] == pytest.approx(-1.0)
    assert abs(corr.loc["f", "noise"]) < 0.2
    pd.testing.assert_frame_equal(corr, corr.T)


def test_evaluate_factor_perfect_signal() -> None:
    rng = np.random.default_rng(6)
    close = make_wide(np.exp(rng.normal(0, 0.01, size=(60, 10)).cumsum(axis=0)) * 100)
    fwd = forward_returns(close, 1)
    result = evaluate_factor("perfect", fwd.copy(), fwd, category="test", horizon=1)
    assert result.name == "perfect"
    assert result.ic.mean == pytest.approx(1.0)
    assert result.rank_ic.mean == pytest.approx(1.0)
    assert result.spread_mean > 0
    assert (result.quantile_mean_returns.diff().dropna() > 0).all()
    summary = result.summary()
    assert summary["factor"] == "perfect"
    assert summary["ic_mean"] == pytest.approx(1.0)
    assert 0.0 <= summary["turnover"] <= 1.0
