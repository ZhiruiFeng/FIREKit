"""Tests for covariance estimators."""

import numpy as np
import pandas as pd
import pytest

from portfolioengine import (
    annualize_cov,
    ewma_cov,
    ledoit_wolf_cov,
    sample_cov,
    to_correlation,
)


@pytest.fixture
def returns_df():
    rng = np.random.default_rng(3)
    data = rng.standard_normal((500, 4)) * 0.01
    return pd.DataFrame(data, columns=list("ABCD"))


def test_sample_cov_matches_numpy(returns_df):
    cov = sample_cov(returns_df)
    expected = np.cov(returns_df.to_numpy(), rowvar=False, ddof=1)
    np.testing.assert_allclose(cov, expected)


def test_sample_cov_accepts_ndarray(returns_df):
    np.testing.assert_allclose(sample_cov(returns_df.to_numpy()), sample_cov(returns_df))


def test_ledoit_wolf_is_symmetric_psd(returns_df):
    cov = ledoit_wolf_cov(returns_df)
    np.testing.assert_allclose(cov, cov.T, atol=1e-15)
    eigvals = np.linalg.eigvalsh(cov)
    assert eigvals.min() > 0.0


def test_ledoit_wolf_shrinks_off_diagonals(returns_df):
    sample = sample_cov(returns_df)
    lw = ledoit_wolf_cov(returns_df)
    off = ~np.eye(4, dtype=bool)
    # Shrinkage pulls off-diagonal mass toward the structured target.
    assert np.abs(lw[off]).mean() < np.abs(sample[off]).mean()


def test_ewma_cov_weights_recent_observations_more():
    # Vol regime change: late observations are much noisier.
    rng = np.random.default_rng(4)
    early = rng.standard_normal((400, 2)) * 0.005
    late = rng.standard_normal((100, 2)) * 0.03
    data = np.vstack([early, late])
    ewma = ewma_cov(data, lam=0.94)
    sample = sample_cov(data)
    assert ewma[0, 0] > sample[0, 0]


def test_ewma_cov_constant_returns_is_zero():
    data = np.full((100, 3), 0.01)
    cov = ewma_cov(data)
    np.testing.assert_allclose(cov, 0.0, atol=1e-18)


def test_ewma_cov_lambda_validation(returns_df):
    with pytest.raises(ValueError):
        ewma_cov(returns_df, lam=1.0)
    with pytest.raises(ValueError):
        ewma_cov(returns_df, lam=0.0)


def test_ewma_cov_is_symmetric_psd(returns_df):
    cov = ewma_cov(returns_df)
    np.testing.assert_allclose(cov, cov.T, atol=1e-18)
    eigvals = np.linalg.eigvalsh(cov)
    assert eigvals.min() > -1e-12


def test_estimators_reject_nan():
    data = np.array([[0.01, 0.02], [np.nan, 0.01], [0.0, 0.0]])
    for fn in (sample_cov, ledoit_wolf_cov, ewma_cov):
        with pytest.raises(ValueError):
            fn(data)


def test_estimators_reject_too_few_rows():
    with pytest.raises(ValueError):
        sample_cov(np.array([[0.01, 0.02]]))


def test_to_correlation_unit_diagonal(returns_df):
    corr = to_correlation(sample_cov(returns_df))
    np.testing.assert_allclose(np.diag(corr), 1.0)
    assert np.abs(corr).max() <= 1.0


def test_to_correlation_known_value():
    cov = np.array([[0.04, 0.012], [0.012, 0.09]])
    corr = to_correlation(cov)
    assert corr[0, 1] == pytest.approx(0.012 / (0.2 * 0.3))


def test_annualize_cov_scales_by_252():
    cov = np.eye(2) * 1e-4
    np.testing.assert_allclose(annualize_cov(cov), np.eye(2) * 252e-4)
