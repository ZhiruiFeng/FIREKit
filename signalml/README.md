# SignalML

ML model hub for trading signal generation — the offline MVP of the FIREKit
SignalML design (`docs/products/04_signalml.md`).

## What it does

- **Feature pipeline** (`signalml.features`): tabular features from wide
  price/volume panels — multi-horizon returns, realized vol, RSI, moving-average
  gaps, volume features — with cross-sectional z-scoring per date. Labels are
  forward returns (features at `t` predict the return `t -> t+h`); construction
  is lookahead-free and tested for it.
- **Model zoo** (`signalml.models`): a common `SignalModel` interface
  (`fit` / `predict` / `feature_importances`) over scikit-learn regressors.
- **Walk-forward engine** (`signalml.walkforward`): rolling or expanding train
  windows with a **purged gap** between train and test so overlapping
  forward-return labels cannot leak (set `gap >= label horizon`).
- **Ensembles** (`signalml.ensemble`): per-date z-score weighted average or
  rank average, plus IC-proportional weight estimation.
- **Evaluation** (`signalml.evaluation`): OOS IC, rank IC, hit rate, decile
  spread (per date and cumulative), per-window metrics, importance aggregation.
- **Model registry** (`signalml.registry`): pickle + JSON-metadata persistence
  of fitted models in a local directory.
- **Synthetic data** (`signalml.data`): seeded multi-symbol panels with an
  injected momentum/reversal signal so models have something real to find.

## Substitutions vs the design doc

This environment has no LightGBM/XGBoost/torch and no network access, so:

| Design doc | This MVP |
|---|---|
| LightGBM ranker | `GradientBoostingSignalModel` (sklearn `HistGradientBoostingRegressor`) |
| XGBoost ranker | `RandomForestSignalModel` (sklearn `RandomForestRegressor`) |
| LSTM / TFT | `RidgeSignalModel` (linear baseline) |
| lambdarank objective | regression on forward returns + rank-based evaluation |

## Quick start

```python
from signalml import (
    EnsembleCombiner, GradientBoostingSignalModel, RandomForestSignalModel,
    RidgeSignalModel, WalkForwardEngine, build_dataset, generate_panel,
    ic_weights, summarize,
)

close, volume = generate_panel(n_symbols=50, n_days=600, seed=7)
X, y = build_dataset(close, volume, horizon=5)

engine = WalkForwardEngine(
    model_factories={
        "hist_gbdt": GradientBoostingSignalModel,
        "random_forest": RandomForestSignalModel,
        "ridge": RidgeSignalModel,
    },
    train_size=252, test_size=21, gap=5,  # gap >= label horizon
)
result = engine.run(X, y)

weights = ic_weights(result.predictions, result.actuals)
signal = EnsembleCombiner("zscore_mean", weights=weights).combine(result.predictions)
print(summarize(signal, result.actuals))
```

## Demo

```bash
cd signalml && python3 -m signalml.demo
```

Walk-forward trains the zoo + IC-weighted ensemble on a synthetic 50-symbol
panel and writes `hub/data/signalml.json` (per `hub/SCHEMA.md`): per-model OOS
IC/hit-rate table, cumulative decile-spread chart, feature-importance chart.
Deterministic, offline, < 60 s.

## Tests

```bash
cd signalml && python3 -m pytest tests -q
```
