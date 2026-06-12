"""Ensembles: combine per-model prediction columns into a single signal."""

from __future__ import annotations

import pandas as pd

from signalml.evaluation import per_date_ic

VALID_METHODS = ("zscore_mean", "rank_mean")


class EnsembleCombiner:
    """Weighted combination of model predictions.

    Methods:
      - ``zscore_mean``: z-score each model's predictions within each date,
        then take the weighted mean (weighted average ensemble).
      - ``rank_mean``: percentile-rank each model's predictions within each
        date (centered at 0), then take the weighted mean (rank average).
    """

    def __init__(self, method: str = "zscore_mean", weights: dict[str, float] | None = None):
        if method not in VALID_METHODS:
            raise ValueError(f"method must be one of {VALID_METHODS}, got {method!r}")
        if weights is not None:
            if not weights:
                raise ValueError("weights dict must not be empty")
            if any(w < 0 for w in weights.values()):
                raise ValueError("weights must be non-negative")
            if sum(weights.values()) <= 0:
                raise ValueError("weights must not all be zero")
        self.method = method
        self.weights = weights

    def combine(self, predictions: pd.DataFrame) -> pd.Series:
        """Combine model prediction columns into one ensemble Series.

        Args:
            predictions: (date, symbol)-indexed frame, one column per model.
        """
        if predictions.empty:
            raise ValueError("predictions frame is empty")
        weights = self._resolve_weights(list(predictions.columns))

        grouped = predictions.groupby(level="date")
        if self.method == "zscore_mean":
            transformed = (predictions - grouped.transform("mean")) / (
                grouped.transform("std") + 1e-9
            )
        else:  # rank_mean
            transformed = grouped.rank(pct=True) - 0.5

        combined = sum(transformed[name] * w for name, w in weights.items())
        combined = pd.Series(combined, index=predictions.index, name=f"ensemble_{self.method}")
        return combined

    def _resolve_weights(self, columns: list[str]) -> dict[str, float]:
        if self.weights is None:
            return {c: 1.0 / len(columns) for c in columns}
        missing = set(self.weights) - set(columns)
        if missing:
            raise KeyError(f"weights refer to unknown model columns: {sorted(missing)}")
        total = sum(self.weights.values())
        return {c: self.weights.get(c, 0.0) / total for c in columns}


def ic_weights(predictions: pd.DataFrame, actuals: pd.Series) -> dict[str, float]:
    """IC-proportional non-negative weights (clipped at zero, sum to 1).

    Falls back to equal weights when every model has non-positive IC.
    """
    raw = {
        str(name): max(float(per_date_ic(predictions[name], actuals).mean()), 0.0)
        for name in predictions.columns
    }
    total = sum(raw.values())
    if total <= 0:
        return {name: 1.0 / len(raw) for name in raw}
    return {name: value / total for name, value in raw.items()}
