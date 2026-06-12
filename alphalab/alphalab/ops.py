"""Cross-sectional and time-series operators on wide (date x symbol) frames.

These mirror the operator vocabulary of WorldQuant's Alpha101 paper:
``rank`` is cross-sectional (within a date), ``ts_*`` operate down the
time axis per symbol.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Replace +/-inf with NaN (division-heavy formulas produce them)."""
    return df.replace([np.inf, -np.inf], np.nan)


# -- cross-sectional ----------------------------------------------------------


def cs_rank(df: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional percentile rank in (0, 1] within each date."""
    return df.rank(axis=1, pct=True)


def cs_zscore(df: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional z-score within each date."""
    demeaned = df.sub(df.mean(axis=1), axis=0)
    return demeaned.div(df.std(axis=1), axis=0)


def cs_demean(df: pd.DataFrame) -> pd.DataFrame:
    """Subtract the cross-sectional mean (market-neutralize)."""
    return df.sub(df.mean(axis=1), axis=0)


# -- time-series --------------------------------------------------------------


def delay(df: pd.DataFrame, d: int) -> pd.DataFrame:
    """Value d days ago."""
    return df.shift(d)


def delta(df: pd.DataFrame, d: int) -> pd.DataFrame:
    """Difference vs d days ago."""
    return df.diff(d)


def ts_sum(df: pd.DataFrame, d: int) -> pd.DataFrame:
    return df.rolling(d).sum()


def ts_mean(df: pd.DataFrame, d: int) -> pd.DataFrame:
    return df.rolling(d).mean()


def ts_std(df: pd.DataFrame, d: int) -> pd.DataFrame:
    return df.rolling(d).std()


def ts_min(df: pd.DataFrame, d: int) -> pd.DataFrame:
    return df.rolling(d).min()


def ts_max(df: pd.DataFrame, d: int) -> pd.DataFrame:
    return df.rolling(d).max()


def ts_rank(df: pd.DataFrame, d: int) -> pd.DataFrame:
    """Percentile rank of today's value within the trailing d-day window."""
    return df.rolling(d).rank(pct=True)


def ts_corr(x: pd.DataFrame, y: pd.DataFrame, d: int) -> pd.DataFrame:
    """Rolling d-day correlation, column by column."""
    return clean(x.rolling(d).corr(y))


def ts_cov(x: pd.DataFrame, y: pd.DataFrame, d: int) -> pd.DataFrame:
    """Rolling d-day covariance, column by column."""
    return x.rolling(d).cov(y)


def returns(close: pd.DataFrame, d: int = 1) -> pd.DataFrame:
    """Simple d-day returns."""
    return close.pct_change(d)
