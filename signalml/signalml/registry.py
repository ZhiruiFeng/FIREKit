"""Model registry: persist fitted SignalModels to a local directory.

Uses pickle (stdlib) plus a JSON metadata sidecar per model. Intended for
local, trusted artifacts only -- pickle is not safe for untrusted files.
"""

from __future__ import annotations

import json
import pickle
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from signalml.models import SignalModel

_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._\-]*$")


class ModelRegistry:
    """Save/load fitted models under a root directory."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _validate_name(self, name: str) -> str:
        if not _NAME_RE.match(name):
            raise ValueError(f"invalid model name: {name!r}")
        return name

    def _model_path(self, name: str) -> Path:
        return self.root / f"{self._validate_name(name)}.pkl"

    def _meta_path(self, name: str) -> Path:
        return self.root / f"{self._validate_name(name)}.meta.json"

    def save(
        self, name: str, model: SignalModel, metadata: dict[str, Any] | None = None
    ) -> Path:
        """Persist a fitted model and its metadata; returns the pickle path."""
        if not model.is_fitted:
            raise ValueError("refusing to register an unfitted model")
        path = self._model_path(name)
        with path.open("wb") as fh:
            pickle.dump(model, fh)
        meta = {
            "name": name,
            "model_class": type(model).__name__,
            "model_name": model.name,
            "feature_names": model.feature_names,
            "saved_at": datetime.now(UTC).isoformat(),
            "metadata": metadata or {},
        }
        with self._meta_path(name).open("w") as fh:
            json.dump(meta, fh, indent=2)
        return path

    def load(self, name: str) -> SignalModel:
        """Load a previously saved model."""
        path = self._model_path(name)
        if not path.exists():
            raise FileNotFoundError(f"no registered model named {name!r} in {self.root}")
        with path.open("rb") as fh:
            model = pickle.load(fh)
        if not isinstance(model, SignalModel):
            raise TypeError(f"artifact {name!r} is not a SignalModel")
        return model

    def metadata(self, name: str) -> dict[str, Any]:
        """Load the metadata sidecar for a saved model."""
        path = self._meta_path(name)
        if not path.exists():
            raise FileNotFoundError(f"no metadata for model {name!r} in {self.root}")
        with path.open() as fh:
            return json.load(fh)

    def list_models(self) -> list[str]:
        """Names of all registered models, sorted."""
        return sorted(p.stem for p in self.root.glob("*.pkl"))

    def delete(self, name: str) -> None:
        """Remove a registered model and its metadata."""
        self._model_path(name).unlink(missing_ok=True)
        self._meta_path(name).unlink(missing_ok=True)
