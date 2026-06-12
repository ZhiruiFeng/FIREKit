"""Aggregation: per-symbol daily sentiment indexes, shocks, breadth."""

from __future__ import annotations

import numpy as np
import pandas as pd

from sentimentpulse.news import NewsItem
from sentimentpulse.scorer import SentimentResult


def scores_frame(items: list[NewsItem], results: list[SentimentResult]) -> pd.DataFrame:
    """Combine items and their scores into a tidy frame.

    Columns: timestamp, date, symbol, score, uncertainty.
    """
    if len(items) != len(results):
        raise ValueError("items and results must have equal length")
    rows = [
        {
            "timestamp": item.timestamp,
            "date": pd.Timestamp(item.timestamp).normalize(),
            "symbol": item.symbol,
            "score": res.score,
            "uncertainty": res.uncertainty,
        }
        for item, res in zip(items, results, strict=True)
    ]
    return pd.DataFrame(rows)


def daily_scores(
    frame: pd.DataFrame, dates: pd.DatetimeIndex | None = None
) -> pd.DataFrame:
    """Per-symbol daily mean score (wide: date x symbol).

    Days without news are NaN unless ``dates`` is given, in which case the
    output is reindexed to that calendar (still NaN on no-news days).
    """
    if frame.empty:
        raise ValueError("scores frame is empty")
    wide = frame.pivot_table(index="date", columns="symbol", values="score", aggfunc="mean")
    wide.index.name = "date"
    if dates is not None:
        wide = wide.reindex(dates)
        wide.index.name = "date"
    return wide.sort_index()


def sentiment_index(daily: pd.DataFrame, halflife: float = 5.0) -> pd.DataFrame:
    """Decay-weighted daily sentiment index per symbol.

    Exponentially weighted mean with the given halflife (in days); NaN
    (no-news) days carry the decayed index forward without contributing.
    """
    if halflife <= 0:
        raise ValueError("halflife must be positive")
    return daily.ewm(halflife=halflife, ignore_na=False, min_periods=1).mean()


def sentiment_momentum(index_df: pd.DataFrame, lookback: int = 5) -> pd.DataFrame:
    """Change in the sentiment index over ``lookback`` days."""
    return index_df.diff(lookback)


def detect_shocks(
    daily: pd.DataFrame,
    window: int = 20,
    z_threshold: float = 2.0,
    min_obs: int = 5,
) -> pd.DataFrame:
    """Detect sentiment shocks: daily-score z-spikes vs trailing history.

    For each symbol, missing (no-news) days are treated as neutral (0.0).
    z_t = (x_t - mean(x_{t-window..t-1})) / std(x_{t-window..t-1}); a shock
    is a day with news where |z| >= z_threshold.

    Returns:
        DataFrame with columns [date, symbol, score, zscore, direction]
        sorted by date; direction is "positive" or "negative".
    """
    observed = daily.notna()
    filled = daily.fillna(0.0)
    mean = filled.rolling(window, min_periods=min_obs).mean().shift(1)
    std = filled.rolling(window, min_periods=min_obs).std().shift(1)
    z = (filled - mean) / std.replace(0.0, np.nan)

    shock_mask = observed & (z.abs() >= z_threshold)
    records: list[dict[str, object]] = []
    for symbol in daily.columns:
        hits = shock_mask[symbol]
        for date in daily.index[hits.fillna(False)]:
            zval = float(z.at[date, symbol])
            records.append(
                {
                    "date": date,
                    "symbol": symbol,
                    "score": float(daily.at[date, symbol]),
                    "zscore": zval,
                    "direction": "positive" if zval > 0 else "negative",
                }
            )
    out = pd.DataFrame(records, columns=["date", "symbol", "score", "zscore", "direction"])
    return out.sort_values(["date", "symbol"]).reset_index(drop=True)


def sentiment_breadth(index_df: pd.DataFrame) -> pd.Series:
    """Fraction of symbols with a strictly positive sentiment index per day."""
    valid = index_df.notna().sum(axis=1)
    positive = (index_df > 0).sum(axis=1)
    breadth = positive / valid.replace(0, np.nan)
    breadth.name = "breadth"
    return breadth
