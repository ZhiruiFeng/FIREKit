"""Walk-forward training/validation engine with a purged gap.

Splits the unique sorted dates into consecutive (train, gap, test) windows:

    [ train_size dates ][ gap dates (purged) ][ test_size dates ] -> step by test_size

The purged gap prevents label leakage: with a forward-return horizon of h
days, a label at the last training date overlaps the first h test dates, so
``gap`` should be >= the label horizon.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import pandas as pd

from signalml.evaluation import summarize
from signalml.models import SignalModel

ModelFactory = Callable[[], SignalModel]


@dataclass(frozen=True)
class Window:
    """One walk-forward window (dates are inclusive bounds)."""

    index: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    n_train: int
    n_test: int


@dataclass
class WalkForwardResult:
    """Stitched out-of-sample predictions plus per-window diagnostics."""

    predictions: pd.DataFrame  # (date, symbol) index, one column per model
    actuals: pd.Series  # aligned forward returns
    windows: list[Window]
    window_metrics: pd.DataFrame  # columns: window, model, ic, rank_ic, hit_rate, ...
    importances: dict[str, pd.DataFrame] = field(default_factory=dict)
    models: dict[str, SignalModel] = field(default_factory=dict)  # last-window fits


class WalkForwardEngine:
    """Run walk-forward training for a dict of model factories."""

    def __init__(
        self,
        model_factories: dict[str, ModelFactory],
        train_size: int = 252,
        test_size: int = 21,
        gap: int = 5,
        expanding: bool = False,
        quantiles: int = 10,
    ) -> None:
        if not model_factories:
            raise ValueError("model_factories must not be empty")
        if train_size < 2 or test_size < 1 or gap < 0:
            raise ValueError("invalid window sizes")
        self.model_factories = dict(model_factories)
        self.train_size = train_size
        self.test_size = test_size
        self.gap = gap
        self.expanding = expanding
        self.quantiles = quantiles

    def make_windows(self, dates: pd.DatetimeIndex) -> list[Window]:
        """Compute window boundaries over the given unique sorted dates."""
        windows: list[Window] = []
        start = 0
        idx = 0
        while True:
            train_end_i = start + self.train_size  # exclusive
            test_start_i = train_end_i + self.gap
            test_end_i = test_start_i + self.test_size  # exclusive
            if test_end_i > len(dates):
                break
            train_start_i = 0 if self.expanding else start
            windows.append(
                Window(
                    index=idx,
                    train_start=dates[train_start_i],
                    train_end=dates[train_end_i - 1],
                    test_start=dates[test_start_i],
                    test_end=dates[test_end_i - 1],
                    n_train=train_end_i - train_start_i,
                    n_test=self.test_size,
                )
            )
            start += self.test_size
            idx += 1
        return windows

    def run(self, X: pd.DataFrame, y: pd.Series) -> WalkForwardResult:
        """Execute walk-forward training and stitch OOS predictions."""
        if not X.index.equals(y.index):
            raise ValueError("X and y must share an identical (date, symbol) index")
        dates = pd.DatetimeIndex(X.index.get_level_values("date").unique()).sort_values()
        windows = self.make_windows(dates)
        if not windows:
            raise ValueError(
                f"not enough dates ({len(dates)}) for train_size={self.train_size} "
                f"+ gap={self.gap} + test_size={self.test_size}"
            )

        date_level = X.index.get_level_values("date")
        preds: dict[str, list[pd.Series]] = {name: [] for name in self.model_factories}
        importances: dict[str, list[pd.Series]] = {name: [] for name in self.model_factories}
        metric_rows: list[dict[str, object]] = []
        last_models: dict[str, SignalModel] = {}

        for window in windows:
            train_mask = (date_level >= window.train_start) & (date_level <= window.train_end)
            test_mask = (date_level >= window.test_start) & (date_level <= window.test_end)
            X_train, y_train = X[train_mask], y[train_mask]
            X_test, y_test = X[test_mask], y[test_mask]

            for name, factory in self.model_factories.items():
                model = factory()
                model.fit(X_train, y_train)
                pred = model.predict(X_test)
                preds[name].append(pred)
                importances[name].append(model.feature_importances())
                last_models[name] = model

                metrics = summarize(pred, y_test, quantiles=self.quantiles)
                metric_rows.append({"window": window.index, "model": name, **metrics})

        prediction_df = pd.DataFrame(
            {name: pd.concat(series_list) for name, series_list in preds.items()}
        ).sort_index()
        actuals = y.loc[prediction_df.index]
        importance_frames = {
            name: pd.DataFrame(rows).reset_index(drop=True)
            for name, rows in importances.items()
        }
        return WalkForwardResult(
            predictions=prediction_df,
            actuals=actuals,
            windows=windows,
            window_metrics=pd.DataFrame(metric_rows),
            importances=importance_frames,
            models=last_models,
        )
