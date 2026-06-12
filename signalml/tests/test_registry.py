"""Tests for the model registry."""

from __future__ import annotations

import numpy as np
import pytest

from signalml.models import RidgeSignalModel
from signalml.registry import ModelRegistry


@pytest.fixture()
def fitted_model(toy_xy):
    X, y = toy_xy
    return RidgeSignalModel().fit(X, y), X


def test_save_load_roundtrip(tmp_path, fitted_model):
    model, X = fitted_model
    registry = ModelRegistry(tmp_path)
    path = registry.save("ridge_v1", model, metadata={"horizon": 5})
    assert path.exists()
    loaded = registry.load("ridge_v1")
    assert isinstance(loaded, RidgeSignalModel)
    assert np.allclose(loaded.predict(X).to_numpy(), model.predict(X).to_numpy())


def test_metadata_persisted(tmp_path, fitted_model):
    model, _ = fitted_model
    registry = ModelRegistry(tmp_path)
    registry.save("m", model, metadata={"horizon": 5, "note": "test"})
    meta = registry.metadata("m")
    assert meta["metadata"] == {"horizon": 5, "note": "test"}
    assert meta["model_class"] == "RidgeSignalModel"
    assert meta["feature_names"] == ["f0", "f1", "f2"]
    assert "saved_at" in meta


def test_list_models_sorted(tmp_path, fitted_model):
    model, _ = fitted_model
    registry = ModelRegistry(tmp_path)
    for name in ["zeta", "alpha", "mid"]:
        registry.save(name, model)
    assert registry.list_models() == ["alpha", "mid", "zeta"]


def test_load_missing_raises(tmp_path):
    registry = ModelRegistry(tmp_path)
    with pytest.raises(FileNotFoundError):
        registry.load("ghost")
    with pytest.raises(FileNotFoundError):
        registry.metadata("ghost")


def test_unfitted_model_rejected(tmp_path):
    registry = ModelRegistry(tmp_path)
    with pytest.raises(ValueError):
        registry.save("bad", RidgeSignalModel())


def test_invalid_names_rejected(tmp_path, fitted_model):
    model, _ = fitted_model
    registry = ModelRegistry(tmp_path)
    for bad in ["../escape", "a/b", "", ".hidden", "name with spaces"]:
        with pytest.raises(ValueError):
            registry.save(bad, model)


def test_delete(tmp_path, fitted_model):
    model, _ = fitted_model
    registry = ModelRegistry(tmp_path)
    registry.save("temp", model)
    assert registry.list_models() == ["temp"]
    registry.delete("temp")
    assert registry.list_models() == []
    registry.delete("temp")  # idempotent
