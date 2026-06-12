"""SignalML demo: walk-forward train the model zoo on a synthetic panel.

Writes hub results JSON to ``<repo>/hub/data/signalml.json`` per hub/SCHEMA.md.
Run with: ``cd signalml && python3 -m signalml.demo``
"""

from __future__ import annotations

import json
import math
import time
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from signalml import __version__
from signalml.data import generate_panel
from signalml.ensemble import EnsembleCombiner, ic_weights
from signalml.evaluation import (
    aggregate_importances,
    cumulative_spread,
    decile_spread,
    summarize,
)
from signalml.features import build_dataset
from signalml.models import (
    GradientBoostingSignalModel,
    RandomForestSignalModel,
    RidgeSignalModel,
)
from signalml.walkforward import WalkForwardEngine

HUB_DATA_DIR = Path(__file__).resolve().parents[2] / "hub" / "data"

N_SYMBOLS = 50
N_DAYS = 600
LABEL_HORIZON = 5
SEED = 7


def _clean(value: float) -> float | None:
    """JSON-safe float: NaN/inf become null."""
    f = float(value)
    return None if (math.isnan(f) or math.isinf(f)) else round(f, 6)


def _downsample(x: list, max_points: int = 500) -> list[int]:
    """Indices that keep a series under max_points."""
    if len(x) <= max_points:
        return list(range(len(x)))
    step = math.ceil(len(x) / max_points)
    idx = list(range(0, len(x), step))
    if idx[-1] != len(x) - 1:
        idx.append(len(x) - 1)
    return idx


def run_demo() -> dict:
    """Run the full walk-forward demo and return the hub results payload."""
    t0 = time.time()

    # 1. Synthetic 50-symbol panel with an injected learnable signal.
    close, volume = generate_panel(n_symbols=N_SYMBOLS, n_days=N_DAYS, seed=SEED)
    X, y = build_dataset(close, volume, horizon=LABEL_HORIZON)

    # 2. Walk-forward train the model zoo (purged gap >= label horizon).
    engine = WalkForwardEngine(
        model_factories={
            "hist_gbdt": GradientBoostingSignalModel,
            "random_forest": RandomForestSignalModel,
            "ridge": RidgeSignalModel,
        },
        train_size=252,
        test_size=21,
        gap=LABEL_HORIZON,
    )
    result = engine.run(X, y)

    # 3. IC-weighted ensemble over stitched OOS predictions.
    weights = ic_weights(result.predictions, result.actuals)
    ensemble = EnsembleCombiner(method="zscore_mean", weights=weights)
    ens_pred = ensemble.combine(result.predictions)

    # 4. Evaluate each model + ensemble out of sample.
    all_preds = result.predictions.copy()
    all_preds["ensemble"] = ens_pred
    oos: dict[str, dict[str, float]] = {
        str(name): summarize(all_preds[name], result.actuals) for name in all_preds.columns
    }
    best_model = max(oos, key=lambda m: oos[m]["ic"])

    # 5. Cumulative decile spread for the ensemble.
    spread = decile_spread(ens_pred, result.actuals)
    cum_spread = cumulative_spread(spread)

    # 6. Aggregated feature importances (mean across windows, averaged over models).
    per_model_imp = {
        name: aggregate_importances(frame) for name, frame in result.importances.items()
    }
    avg_imp = pd.DataFrame(per_model_imp).mean(axis=1).sort_values(ascending=False)
    avg_imp = avg_imp / avg_imp.sum()

    # 7. Per-window ensemble-vs-models IC chart.
    win_ic = result.window_metrics.pivot(index="window", columns="model", values="ic")

    elapsed = time.time() - t0

    spread_dates = [d.strftime("%Y-%m-%d") for d in cum_spread.index]
    keep = _downsample(spread_dates)

    payload = {
        "schema_version": 1,
        "product": "signalml",
        "title": "SignalML",
        "tagline": "ML model hub for trading signal generation",
        "version": __version__,
        "generated_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "status": "ok",
        "summary_metrics": [
            {"label": "Symbols", "value": str(N_SYMBOLS)},
            {"label": "Walk-forward windows", "value": str(len(result.windows))},
            {"label": "OOS observations", "value": str(int(oos["ensemble"]["n_obs"]))},
            {"label": "Best model", "value": best_model},
            {"label": "Best OOS IC", "value": f"{oos[best_model]['ic']:.4f}"},
            {"label": "Ensemble rank IC", "value": f"{oos['ensemble']['rank_ic']:.4f}"},
            {"label": "Ensemble hit rate", "value": f"{100 * oos['ensemble']['hit_rate']:.1f}", "unit": "%"},
            {"label": "Demo runtime", "value": f"{elapsed:.1f}", "unit": "s"},
        ],
        "charts": [
            {
                "id": "cumulative_decile_spread",
                "title": "Cumulative decile spread (ensemble, OOS)",
                "type": "line",
                "x_label": "Date",
                "y_label": "Cumulative top-bottom decile return",
                "x": [spread_dates[i] for i in keep],
                "series": [
                    {
                        "name": "Ensemble",
                        "data": [_clean(cum_spread.iloc[i]) for i in keep],
                    }
                ],
            },
            {
                "id": "feature_importance",
                "title": "Aggregated feature importance (mean across windows and models)",
                "type": "bar",
                "x_label": "Feature",
                "y_label": "Normalized importance",
                "x": [str(f) for f in avg_imp.index],
                "series": [
                    {"name": "Importance", "data": [_clean(v) for v in avg_imp.to_numpy()]}
                ],
            },
            {
                "id": "ic_by_window",
                "title": "Per-window OOS IC by model",
                "type": "line",
                "x_label": "Walk-forward window",
                "y_label": "Pearson IC",
                "x": [int(w) for w in win_ic.index],
                "series": [
                    {
                        "name": str(model),
                        "data": [_clean(v) for v in win_ic[model].to_numpy()],
                    }
                    for model in win_ic.columns
                ],
            },
        ],
        "tables": [
            {
                "id": "oos_metrics",
                "title": "Out-of-sample metrics by model",
                "columns": ["Model", "IC", "Rank IC", "Hit rate", "Decile spread (mean)", "Ensemble weight"],
                "rows": [
                    [
                        name,
                        _clean(m["ic"]),
                        _clean(m["rank_ic"]),
                        _clean(m["hit_rate"]),
                        _clean(m["mean_decile_spread"]),
                        _clean(weights.get(name, 0.0)) if name != "ensemble" else None,
                    ]
                    for name, m in oos.items()
                ],
            },
        ],
        "notes": (
            f"Walk-forward (train 252d, purged gap {LABEL_HORIZON}d, test 21d) on a synthetic "
            f"{N_SYMBOLS}-symbol x {N_DAYS}-day panel with an injected momentum/reversal signal; "
            f"labels are {LABEL_HORIZON}-day forward returns. LightGBM/XGBoost/LSTM from the design "
            "doc are substituted with sklearn HistGradientBoosting, RandomForest and Ridge "
            "(offline environment). All metrics are out of sample; seeds fixed."
        ),
    }
    return payload


def main() -> None:
    payload = run_demo()
    HUB_DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = HUB_DATA_DIR / "signalml.json"
    with out_path.open("w") as fh:
        json.dump(payload, fh, indent=2)
    best = max(
        (row for row in payload["tables"][0]["rows"]),
        key=lambda r: (r[1] if r[1] is not None else -1),
    )
    print(f"SignalML demo complete -> {out_path}")
    print(f"  windows={payload['summary_metrics'][1]['value']}  best={best[0]} ic={best[1]}")


if __name__ == "__main__":
    main()
