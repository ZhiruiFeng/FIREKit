"""Feature pipeline: tabular features and forward-return labels from price panels.

All features at date ``t`` are computed strictly from data up to and including
``t``; labels at date ``t`` are the forward return from ``t`` to ``t + horizon``
(features at t predict the return t -> t+h, no lookahead).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_RETURN_HORIZONS = (1, 5, 10, 21)


def compute_rsi(close: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Relative Strength Index (0-100) on a wide close-price panel."""
    delta = close.diff()
    gain = delta.clip(lower=0.0).rolling(period).mean()
    loss = (-delta).clip(lower=0.0).rolling(period).mean()
    rs = gain / (loss + 1e-12)
    return 100.0 - 100.0 / (1.0 + rs)


def _stack(wide: pd.DataFrame) -> pd.Series:
    """Wide (date x symbol) -> long Series with MultiIndex (date, symbol)."""
    s = wide.stack()
    s.index.names = ["date", "symbol"]
    return s


def cross_sectional_zscore(features: pd.DataFrame) -> pd.DataFrame:
    """Z-score each column within each date (cross-sectional normalization)."""
    grouped = features.groupby(level="date")
    mean = grouped.transform("mean")
    std = grouped.transform("std")
    return (features - mean) / (std + 1e-9)


def build_features(
    close: pd.DataFrame,
    volume: pd.DataFrame | None = None,
    return_horizons: tuple[int, ...] = DEFAULT_RETURN_HORIZONS,
    normalize: bool = True,
) -> pd.DataFrame:
    """Build a long-format feature matrix from wide price/volume panels.

    Features per (date, symbol):
      - ret_{h}: trailing h-day return for each horizon
      - vol_21: trailing 21-day stdev of daily returns
      - rsi_14: relative strength index
      - ma_gap_10 / ma_gap_21: close vs trailing moving average
      - volume_z_21, volume_ratio_5_21 (when volume is provided)

    Returns:
        DataFrame indexed by MultiIndex (date, symbol). Rows where any
        feature is NaN (warm-up period) are dropped.
    """
    daily_ret = close.pct_change()
    feats: dict[str, pd.DataFrame] = {}
    for h in return_horizons:
        feats[f"ret_{h}"] = close.pct_change(h)
    feats["vol_21"] = daily_ret.rolling(21).std()
    feats["rsi_14"] = compute_rsi(close, 14)
    feats["ma_gap_10"] = close / close.rolling(10).mean() - 1.0
    feats["ma_gap_21"] = close / close.rolling(21).mean() - 1.0
    if volume is not None:
        log_vol = np.log(volume.clip(lower=1e-9))
        feats["volume_z_21"] = (log_vol - log_vol.rolling(21).mean()) / (
            log_vol.rolling(21).std() + 1e-9
        )
        feats["volume_ratio_5_21"] = volume.rolling(5).mean() / volume.rolling(21).mean() - 1.0

    matrix = pd.DataFrame({name: _stack(wide) for name, wide in feats.items()})
    matrix = matrix.dropna()
    if normalize:
        matrix = cross_sectional_zscore(matrix)
    return matrix.sort_index()


def build_labels(close: pd.DataFrame, horizon: int = 5) -> pd.Series:
    """Forward-return labels: label at date t is close[t+h] / close[t] - 1.

    The last ``horizon`` dates have no label and are dropped.
    """
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    fwd = close.shift(-horizon) / close - 1.0
    label = _stack(fwd).dropna()
    label.name = f"fwd_ret_{horizon}"
    return label.sort_index()


def build_dataset(
    close: pd.DataFrame,
    volume: pd.DataFrame | None = None,
    horizon: int = 5,
    return_horizons: tuple[int, ...] = DEFAULT_RETURN_HORIZONS,
    normalize: bool = True,
) -> tuple[pd.DataFrame, pd.Series]:
    """Build an aligned (X, y) supervised dataset.

    Returns:
        X: feature DataFrame indexed by (date, symbol).
        y: forward-return Series with the identical index.
    """
    features = build_features(
        close, volume, return_horizons=return_horizons, normalize=normalize
    )
    labels = build_labels(close, horizon=horizon)
    common = features.index.intersection(labels.index)
    return features.loc[common], labels.loc[common]
