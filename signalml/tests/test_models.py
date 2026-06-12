"""Tests for the model zoo and SignalModel interface."""

from __future__ import annotations

import numpy as np
import pytest

from signalml.models import (
    GradientBoostingSignalModel,
    NotFittedError,
    RandomForestSignalModel,
    RidgeSignalModel,
    default_model_zoo,
)

ALL_MODELS = [GradientBoostingSignalModel, RandomForestSignalModel, RidgeSignalModel]


@pytest.mark.parametrize("model_cls", ALL_MODELS)
def test_fit_predict_preserves_index(model_cls, toy_xy):
    X, y = toy_xy
    model = model_cls()
    model.fit(X, y)
    pred = model.predict(X)
    assert pred.index.equals(X.index)
    assert pred.name == model.name
    assert not pred.isna().any()


@pytest.mark.parametrize("model_cls", ALL_MODELS)
def test_models_learn_linear_signal(model_cls, toy_xy):
    X, y = toy_xy
    model = model_cls()
    model.fit(X, y)
    pred = model.predict(X)
    assert float(np.corrcoef(pred.to_numpy(), y.to_numpy())[0, 1]) > 0.8


@pytest.mark.parametrize("model_cls", ALL_MODELS)
def test_feature_importances_normalized(model_cls, toy_xy):
    X, y = toy_xy
    model = model_cls().fit(X, y)
    imp = model.feature_importances()
    assert list(imp.index) == ["f0", "f1", "f2"]
    assert (imp.to_numpy() >= 0).all()
    assert imp.sum() == pytest.approx(1.0)
    # f0 (coef 2) should dominate the irrelevant f2 (coef 0).
    assert imp["f0"] > imp["f2"]


@pytest.mark.parametrize("model_cls", ALL_MODELS)
def test_predict_before_fit_raises(model_cls, toy_xy):
    X, _ = toy_xy
    model = model_cls()
    with pytest.raises(NotFittedError):
        model.predict(X)
    with pytest.raises(NotFittedError):
        model.feature_importances()


def test_ridge_recovers_coefficients(toy_xy):
    X, y = toy_xy
    model = RidgeSignalModel(alpha=1e-6).fit(X, y)
    assert model.coef_[0] == pytest.approx(2.0, abs=0.05)
    assert model.coef_[1] == pytest.approx(-1.0, abs=0.05)
    assert model.coef_[2] == pytest.approx(0.0, abs=0.05)


def test_mismatched_columns_raise(toy_xy):
    X, y = toy_xy
    model = RidgeSignalModel().fit(X, y)
    X_bad = X.rename(columns={"f0": "g0"})
    with pytest.raises(ValueError):
        model.predict(X_bad)


def test_mismatched_lengths_raise(toy_xy):
    X, y = toy_xy
    with pytest.raises(ValueError):
        RidgeSignalModel().fit(X, y.iloc[:-3])


def test_default_model_zoo_names():
    zoo = default_model_zoo()
    assert set(zoo) == {"hist_gbdt", "random_forest", "ridge"}
    for name, cls in zoo.items():
        assert cls.name == name


@pytest.mark.parametrize("model_cls", ALL_MODELS)
def test_fit_is_deterministic(model_cls, toy_xy):
    X, y = toy_xy
    p1 = model_cls().fit(X, y).predict(X)
    p2 = model_cls().fit(X, y).predict(X)
    assert np.allclose(p1.to_numpy(), p2.to_numpy())
