"""
Validation Utilities

Data and configuration validation helpers.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


class ValidationError(Exception):
    """Raised when validation fails."""
    pass


def validate_data(
    data: pd.DataFrame,
    required_columns: set[str] | None = None,
    check_datetime_index: bool = True,
    check_duplicates: bool = True,
    check_sorted: bool = True,
    check_nans: bool = False,
    check_positive: list[str] | None = None,
) -> list[str]:
    """
    Validate OHLCV data for backtesting.

    Args:
        data: DataFrame to validate
        required_columns: Required column names
        check_datetime_index: Require DatetimeIndex
        check_duplicates: Check for duplicate timestamps
        check_sorted: Check chronological order
        check_nans: Check for NaN values
        check_positive: Columns that must be positive

    Returns:
        List of warning messages (empty if valid)

    Raises:
        ValidationError: If critical validation fails
    """
    warnings = []

    # Check empty
    if len(data) == 0:
        raise ValidationError("Data is empty")

    # Check required columns
    if required_columns is None:
        required_columns = {"open", "high", "low", "close", "volume"}

    data_columns = {col.lower() for col in data.columns}
    missing = required_columns - data_columns
    if missing:
        raise ValidationError(f"Missing required columns: {missing}")

    # Check datetime index
    if check_datetime_index and not isinstance(data.index, pd.DatetimeIndex):
        raise ValidationError("Index must be DatetimeIndex")

    # Check duplicates
    if check_duplicates and data.index.duplicated().any():
        n_dups = data.index.duplicated().sum()
        warnings.append(f"Found {n_dups} duplicate timestamps")

    # Check sorted
    if check_sorted and not data.index.is_monotonic_increasing:
        raise ValidationError("Data must be sorted in ascending order")

    # Check NaNs
    if check_nans:
        nan_counts = data.isna().sum()
        if nan_counts.any():
            for col, count in nan_counts.items():
                if count > 0:
                    warnings.append(f"Column '{col}' has {count} NaN values")

    # Check positive values
    if check_positive:
        for col in check_positive:
            if col in data.columns:
                if (data[col] < 0).any():
                    warnings.append(f"Column '{col}' has negative values")

    # OHLC consistency checks
    close_col = _find_column(data, "close")
    open_col = _find_column(data, "open")
    high_col = _find_column(data, "high")
    low_col = _find_column(data, "low")

    if all([close_col, open_col, high_col, low_col]):
        # High should be >= Open, Close, Low
        if (data[high_col] < data[low_col]).any():
            warnings.append("Found bars where high < low")

        if (data[high_col] < data[open_col]).any():
            warnings.append("Found bars where high < open")

        if (data[high_col] < data[close_col]).any():
            warnings.append("Found bars where high < close")

    return warnings


def validate_config(config: Any) -> list[str]:
    """
    Validate VectorForge configuration.

    Args:
        config: Configuration object to validate

    Returns:
        List of warning messages
    """
    warnings = []

    # Check capital
    if hasattr(config, "default_capital"):
        if config.default_capital <= 0:
            warnings.append("default_capital should be positive")

    # Check execution config
    if hasattr(config, "execution"):
        exec_config = config.execution

        if hasattr(exec_config, "slippage"):
            if exec_config.slippage.base_bps < 0:
                warnings.append("slippage.base_bps should be non-negative")

        if hasattr(exec_config, "commission"):
            if exec_config.commission.min_commission < 0:
                warnings.append("commission.min_commission should be non-negative")

    # Check validation config
    if hasattr(config, "validation"):
        val_config = config.validation

        if hasattr(val_config, "walk_forward"):
            wf = val_config.walk_forward
            if wf.train_period <= wf.test_period:
                warnings.append("train_period should be > test_period")

    return warnings


def _find_column(data: pd.DataFrame, name: str) -> str | None:
    """Find column case-insensitively."""
    for col in data.columns:
        if col.lower() == name.lower():
            return col
    return None
