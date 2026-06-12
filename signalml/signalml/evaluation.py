"""Out-of-sample evaluation: IC, rank IC, hit rate, decile spread, importances."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _paired(pred: pd.Series, actual: pd.Series) -> pd.DataFrame:
    df = pd.DataFrame({"pred": pred, "actual": actual}).dropna()
    if df.empty:
        raise ValueError("no overlapping non-NaN observations between pred and actual")
    return df


def per_date_ic(pred: pd.Series, actual: pd.Series, method: str = "pearson") -> pd.Series:
    """Per-date cross-sectional correlation between predictions and outcomes."""
    if method not in ("pearson", "spearman"):
        raise ValueError(f"unknown correlation method: {method!r}")
    df = _paired(pred, actual)
    ic = df.groupby(level="date").apply(lambda g: g["pred"].corr(g["actual"], method=method))
    ic.name = f"{method}_ic"
    return ic


def information_coefficient(pred: pd.Series, actual: pd.Series) -> float:
    """Mean per-date Pearson IC."""
    return float(per_date_ic(pred, actual, "pearson").mean())


def rank_ic(pred: pd.Series, actual: pd.Series) -> float:
    """Mean per-date Spearman (rank) IC."""
    return float(per_date_ic(pred, actual, "spearman").mean())


def hit_rate(pred: pd.Series, actual: pd.Series) -> float:
    """Fraction of observations where sign(pred) == sign(actual)."""
    df = _paired(pred, actual)
    return float((np.sign(df["pred"]) == np.sign(df["actual"])).mean())


def decile_spread(pred: pd.Series, actual: pd.Series, quantiles: int = 10) -> pd.Series:
    """Per-date top-minus-bottom quantile mean return, ranked by prediction."""
    if quantiles < 2:
        raise ValueError("quantiles must be >= 2")
    df = _paired(pred, actual)

    def _one_date(g: pd.DataFrame) -> float:
        ranks = g["pred"].rank(pct=True, method="first")
        bucket = np.ceil(ranks * quantiles).clip(upper=quantiles)
        top = g.loc[bucket == quantiles, "actual"].mean()
        bottom = g.loc[bucket == 1, "actual"].mean()
        return float(top - bottom)

    spread = df.groupby(level="date").apply(_one_date)
    spread.name = f"q{quantiles}_spread"
    return spread


def cumulative_spread(spread: pd.Series) -> pd.Series:
    """Cumulative sum of the per-date spread (simple, non-compounded)."""
    out = spread.cumsum()
    out.name = "cumulative_spread"
    return out


def summarize(pred: pd.Series, actual: pd.Series, quantiles: int = 10) -> dict[str, float]:
    """All headline OOS metrics in one dict."""
    spread = decile_spread(pred, actual, quantiles=quantiles)
    return {
        "ic": information_coefficient(pred, actual),
        "rank_ic": rank_ic(pred, actual),
        "hit_rate": hit_rate(pred, actual),
        "mean_decile_spread": float(spread.mean()),
        "n_obs": int(len(_paired(pred, actual))),
    }


def aggregate_importances(importances: pd.DataFrame) -> pd.Series:
    """Average per-window importances (rows = windows, columns = features).

    Returns a Series normalized to sum 1, sorted descending.
    """
    if importances.empty:
        raise ValueError("importances frame is empty")
    mean = importances.mean(axis=0).clip(lower=0.0)
    total = mean.sum()
    if total <= 0:
        mean = pd.Series(1.0, index=mean.index)
        total = mean.sum()
    return (mean / total).sort_values(ascending=False)
