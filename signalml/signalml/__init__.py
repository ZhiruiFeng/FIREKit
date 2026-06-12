"""SignalML: ML model hub for trading signal generation.

Offline MVP of the FIREKit SignalML design: feature pipeline, model zoo with
a common SignalModel interface, walk-forward validation with a purged gap,
ensembles, evaluation metrics, and a local model registry.
"""

from signalml.data import generate_panel, make_symbols
from signalml.ensemble import EnsembleCombiner, ic_weights
from signalml.evaluation import (
    aggregate_importances,
    cumulative_spread,
    decile_spread,
    hit_rate,
    information_coefficient,
    per_date_ic,
    rank_ic,
    summarize,
)
from signalml.features import (
    build_dataset,
    build_features,
    build_labels,
    compute_rsi,
    cross_sectional_zscore,
)
from signalml.models import (
    GradientBoostingSignalModel,
    NotFittedError,
    RandomForestSignalModel,
    RidgeSignalModel,
    SignalModel,
    default_model_zoo,
)
from signalml.registry import ModelRegistry
from signalml.walkforward import WalkForwardEngine, WalkForwardResult, Window

__version__ = "0.1.0"

__all__ = [
    "EnsembleCombiner",
    "GradientBoostingSignalModel",
    "ModelRegistry",
    "NotFittedError",
    "RandomForestSignalModel",
    "RidgeSignalModel",
    "SignalModel",
    "WalkForwardEngine",
    "WalkForwardResult",
    "Window",
    "__version__",
    "aggregate_importances",
    "build_dataset",
    "build_features",
    "build_labels",
    "compute_rsi",
    "cross_sectional_zscore",
    "cumulative_spread",
    "decile_spread",
    "default_model_zoo",
    "generate_panel",
    "hit_rate",
    "ic_weights",
    "information_coefficient",
    "make_symbols",
    "per_date_ic",
    "rank_ic",
    "summarize",
]
