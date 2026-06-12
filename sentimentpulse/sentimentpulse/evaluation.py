"""Evaluation: scorer accuracy vs ground truth and sentiment-shock event study."""

from __future__ import annotations

import pandas as pd

from sentimentpulse.scorer import classify

_TRUTH_LABELS = {-1: "negative", 0: "neutral", 1: "positive"}
LABELS = ("negative", "neutral", "positive")


def scorer_accuracy(
    scores: list[float],
    truths: list[int],
    pos_threshold: float = 0.15,
    neg_threshold: float = -0.15,
) -> dict:
    """Three-way classification accuracy of polarity scores vs ground truth.

    Args:
        scores: polarity scores in [-1, 1].
        truths: ground-truth polarities in {-1, 0, +1}.

    Returns:
        dict with overall ``accuracy``, ``n``, per-class recall under
        ``per_class``, and a nested ``confusion`` dict
        (confusion[true_label][predicted_label] -> count).
    """
    if len(scores) != len(truths):
        raise ValueError("scores and truths must have equal length")
    if not scores:
        raise ValueError("nothing to evaluate")
    bad = set(truths) - set(_TRUTH_LABELS)
    if bad:
        raise ValueError(f"truths must be in {{-1, 0, 1}}, got {sorted(bad)}")

    confusion: dict[str, dict[str, int]] = {t: dict.fromkeys(LABELS, 0) for t in LABELS}
    correct = 0
    for score, truth in zip(scores, truths, strict=True):
        true_label = _TRUTH_LABELS[truth]
        pred_label = classify(score, pos_threshold, neg_threshold)
        confusion[true_label][pred_label] += 1
        if pred_label == true_label:
            correct += 1

    per_class: dict[str, float | None] = {}
    for label in LABELS:
        total = sum(confusion[label].values())
        per_class[label] = (confusion[label][label] / total) if total else None

    return {
        "accuracy": correct / len(scores),
        "n": len(scores),
        "per_class": per_class,
        "confusion": confusion,
    }


def forward_returns(close: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Forward return from t to t+horizon for a wide close panel."""
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    return close.shift(-horizon) / close - 1.0


def event_study(
    events: pd.DataFrame,
    close: pd.DataFrame,
    horizons: tuple[int, ...] = (1, 3, 5),
) -> pd.DataFrame:
    """Average forward return after positive vs negative sentiment shocks.

    Args:
        events: frame with columns [date, symbol, direction] (see
            ``aggregate.detect_shocks``).
        close: wide close-price panel covering the event dates.
        horizons: forward horizons in days.

    Returns:
        DataFrame indexed by direction ("positive", "negative") with one
        column per horizon (``fwd_{h}d``: mean forward return) plus
        ``n_events``.
    """
    required = {"date", "symbol", "direction"}
    if not required <= set(events.columns):
        raise ValueError(f"events must have columns {sorted(required)}")
    fwd = {h: forward_returns(close, h) for h in horizons}

    rows: dict[str, dict[str, float]] = {}
    for direction in ("positive", "negative"):
        sub = events[events["direction"] == direction]
        row: dict[str, float] = {"n_events": float(len(sub))}
        for h in horizons:
            vals = [
                fwd[h].at[ev.date, ev.symbol]
                for ev in sub.itertuples()
                if ev.date in fwd[h].index and ev.symbol in fwd[h].columns
            ]
            series = pd.Series(vals, dtype=float).dropna()
            row[f"fwd_{h}d"] = float(series.mean()) if len(series) else float("nan")
        rows[direction] = row
    out = pd.DataFrame(rows).T
    out.index.name = "direction"
    return out
