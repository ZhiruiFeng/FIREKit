"""Model zoo: a common SignalModel interface over scikit-learn regressors.

Design-doc substitutions (no LightGBM/XGBoost/torch in this environment):
  - LightGBM -> sklearn HistGradientBoostingRegressor (closest analog:
    histogram-based gradient boosting).
  - XGBoost  -> sklearn RandomForestRegressor (tree-ensemble diversity).
  - LSTM/TFT -> Ridge regression (linear baseline; deep nets out of scope
    for the offline MVP).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge


class NotFittedError(RuntimeError):
    """Raised when predict/importances are requested before fit."""


class SignalModel(ABC):
    """Common interface for signal models: fit / predict / feature_importances."""

    name: str = "signal_model"

    def __init__(self) -> None:
        self._feature_names: list[str] | None = None

    @property
    def is_fitted(self) -> bool:
        return self._feature_names is not None

    @property
    def feature_names(self) -> list[str]:
        self._check_fitted()
        assert self._feature_names is not None
        return list(self._feature_names)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> SignalModel:
        """Fit on a feature matrix X and target Series y (aligned indexes)."""
        if len(X) != len(y):
            raise ValueError(f"X has {len(X)} rows but y has {len(y)}")
        if len(X) == 0:
            raise ValueError("cannot fit on an empty dataset")
        self._feature_names = [str(c) for c in X.columns]
        self._fit(X.to_numpy(dtype=float), y.to_numpy(dtype=float))
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Predict scores; returns a Series aligned to X's index."""
        self._check_fitted()
        if list(map(str, X.columns)) != self._feature_names:
            raise ValueError("feature columns do not match the columns used at fit time")
        preds = self._predict(X.to_numpy(dtype=float))
        return pd.Series(preds, index=X.index, name=self.name)

    def feature_importances(self) -> pd.Series:
        """Non-negative importances over fit-time features, normalized to sum 1."""
        self._check_fitted()
        raw = np.clip(self._importances(), 0.0, None)
        total = raw.sum()
        if total <= 0:
            raw = np.ones_like(raw)
            total = raw.sum()
        return pd.Series(raw / total, index=self._feature_names, name=self.name)

    def _check_fitted(self) -> None:
        if not self.is_fitted:
            raise NotFittedError(f"{type(self).__name__} is not fitted yet")

    @abstractmethod
    def _fit(self, X: np.ndarray, y: np.ndarray) -> None: ...

    @abstractmethod
    def _predict(self, X: np.ndarray) -> np.ndarray: ...

    @abstractmethod
    def _importances(self) -> np.ndarray: ...


class RidgeSignalModel(SignalModel):
    """L2-regularized linear model (baseline; LSTM/TFT substitute)."""

    name = "ridge"

    def __init__(self, alpha: float = 10.0) -> None:
        super().__init__()
        self.alpha = alpha
        self._est = Ridge(alpha=alpha)

    def _fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self._est = Ridge(alpha=self.alpha)
        self._est.fit(X, y)

    def _predict(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(self._est.predict(X), dtype=float)

    def _importances(self) -> np.ndarray:
        return np.abs(np.asarray(self._est.coef_, dtype=float))

    @property
    def coef_(self) -> np.ndarray:
        self._check_fitted()
        return np.asarray(self._est.coef_, dtype=float)


class RandomForestSignalModel(SignalModel):
    """Random forest regressor (XGBoost substitute)."""

    name = "random_forest"

    def __init__(
        self,
        n_estimators: int = 150,
        max_depth: int = 6,
        min_samples_leaf: int = 25,
        random_state: int = 0,
        n_jobs: int = -1,
    ) -> None:
        super().__init__()
        self._params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "min_samples_leaf": min_samples_leaf,
            "random_state": random_state,
            "n_jobs": n_jobs,
        }
        self._est = RandomForestRegressor(**self._params)

    def _fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self._est = RandomForestRegressor(**self._params)
        self._est.fit(X, y)

    def _predict(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(self._est.predict(X), dtype=float)

    def _importances(self) -> np.ndarray:
        return np.asarray(self._est.feature_importances_, dtype=float)


class GradientBoostingSignalModel(SignalModel):
    """Histogram gradient boosting (LightGBM substitute).

    HistGradientBoostingRegressor exposes no native importances, so
    ``feature_importances`` uses permutation importance computed on a
    deterministic subsample of the training data stored at fit time.
    """

    name = "hist_gbdt"

    def __init__(
        self,
        max_iter: int = 120,
        learning_rate: float = 0.08,
        max_depth: int = 4,
        l2_regularization: float = 1.0,
        random_state: int = 0,
        importance_sample: int = 1500,
    ) -> None:
        super().__init__()
        self._params = {
            "max_iter": max_iter,
            "learning_rate": learning_rate,
            "max_depth": max_depth,
            "l2_regularization": l2_regularization,
            "random_state": random_state,
            "early_stopping": False,
        }
        self.importance_sample = importance_sample
        self._est = HistGradientBoostingRegressor(**self._params)
        self._X_sample: np.ndarray | None = None
        self._y_sample: np.ndarray | None = None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self._est = HistGradientBoostingRegressor(**self._params)
        self._est.fit(X, y)
        n = min(self.importance_sample, len(X))
        idx = np.random.default_rng(0).choice(len(X), size=n, replace=False)
        self._X_sample, self._y_sample = X[idx], y[idx]

    def _predict(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(self._est.predict(X), dtype=float)

    def _importances(self) -> np.ndarray:
        assert self._X_sample is not None and self._y_sample is not None
        result = permutation_importance(
            self._est, self._X_sample, self._y_sample, n_repeats=3, random_state=0
        )
        return np.asarray(result.importances_mean, dtype=float)


def default_model_zoo() -> dict[str, type[SignalModel]]:
    """The default model zoo, keyed by model name."""
    return {
        GradientBoostingSignalModel.name: GradientBoostingSignalModel,
        RandomForestSignalModel.name: RandomForestSignalModel,
        RidgeSignalModel.name: RidgeSignalModel,
    }
