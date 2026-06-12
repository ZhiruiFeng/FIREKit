"""Tests for the aggregated RiskReport."""

import math

import numpy as np
import pandas as pd
import pytest

from riskguard import ExposureLimits, build_risk_report


@pytest.fixture
def asset_returns():
    rng = np.random.default_rng(99)
    n = 750
    dates = pd.bdate_range("2022-01-03", periods=n)
    data = {
        "EQ": 0.0005 + 0.012 * rng.standard_normal(n),
        "BND": 0.0001 + 0.003 * rng.standard_normal(n),
        "CMD": 0.0002 + 0.009 * rng.standard_normal(n),
    }
    return pd.DataFrame(data, index=dates)


@pytest.fixture
def weights():
    return {"EQ": 0.5, "BND": 0.3, "CMD": 0.2}


def test_report_basic_fields_finite(asset_returns, weights):
    report = build_risk_report(asset_returns, weights)
    assert math.isfinite(report.annual_return)
    assert report.annual_volatility > 0
    assert math.isfinite(report.sharpe)
    assert 0 <= report.max_drawdown < 1


def test_report_var_methods_and_confidences(asset_returns, weights):
    report = build_risk_report(asset_returns, weights)
    for method in ("historical", "gaussian", "cornish_fisher"):
        assert "0.95" in report.var[method]
        assert "0.99" in report.var[method]
        assert report.cvar[method]["0.95"] >= report.var[method]["0.95"]
        assert report.var[method]["0.99"] > report.var[method]["0.95"]


def test_report_portfolio_returns_are_weighted_sum(asset_returns, weights):
    report = build_risk_report(asset_returns, weights)
    manual = (
        0.5 * asset_returns["EQ"] + 0.3 * asset_returns["BND"] + 0.2 * asset_returns["CMD"]
    )
    np.testing.assert_allclose(report.portfolio_returns.to_numpy(), manual.to_numpy())


def test_report_vol_target_exposure(asset_returns, weights):
    report = build_risk_report(asset_returns, weights, target_vol=0.10)
    assert report.vol_target_exposure is not None
    assert report.vol_target_exposure == pytest.approx(
        0.10 / report.current_realized_vol
    )


def test_report_without_target_vol_has_no_exposure(asset_returns, weights):
    report = build_risk_report(asset_returns, weights)
    assert report.vol_target_exposure is None


def test_report_kelly_weights_capped(asset_returns, weights):
    report = build_risk_report(asset_returns, weights, kelly_cap=0.8)
    assert set(report.kelly_weights) == {"EQ", "BND", "CMD"}
    assert all(abs(v) <= 0.8 + 1e-12 for v in report.kelly_weights.values())


def test_report_limit_violations_flow_through(asset_returns):
    limits = ExposureLimits(max_position=0.10, max_gross=1.0, max_net=1.0)
    report = build_risk_report(
        asset_returns, {"EQ": 0.5, "BND": 0.3, "CMD": 0.2}, limits=limits
    )
    assert not report.limit_result.compliant
    assert all(abs(w) <= 0.10 + 1e-9 for w in report.limit_result.weights.values())


def test_report_rejects_unknown_assets(asset_returns):
    with pytest.raises(ValueError, match="unknown assets"):
        build_risk_report(asset_returns, {"NOPE": 1.0})


def test_report_rejects_nan_returns(asset_returns, weights):
    bad = asset_returns.copy()
    bad.iloc[5, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        build_risk_report(bad, weights)


def test_report_to_dict_is_json_friendly(asset_returns, weights):
    import json

    report = build_risk_report(asset_returns, weights, target_vol=0.12)
    payload = report.to_dict()
    text = json.dumps(payload, allow_nan=False)  # must not raise
    assert "annual_volatility" in payload
    assert "kelly_weights" in payload
    assert len(text) > 100
    assert not any(
        isinstance(v, float) and math.isnan(v)
        for v in payload["kelly_weights"].values()
    )
