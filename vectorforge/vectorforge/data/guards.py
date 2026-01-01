"""
Data Guards

Utilities to prevent lookahead bias and other data issues.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


class LookaheadError(Exception):
    """Raised when future data access is attempted."""
    pass


class DataGuard:
    """
    Prevents accidental lookahead bias.

    Wraps data and restricts access to only historical data
    up to the current simulation index.

    Example:
        >>> guard = DataGuard(price_data, current_idx=100)
        >>> guard.close  # Only returns data up to index 100
        >>> guard[105:]  # Raises LookaheadError
    """

    def __init__(self, data: pd.DataFrame, current_idx: int):
        """
        Initialize data guard.

        Args:
            data: Full dataset
            current_idx: Current simulation index (0-based)
        """
        self._data = data
        self._current_idx = current_idx
        self._max_idx = len(data) - 1

    def update_index(self, new_idx: int) -> None:
        """Update current index as simulation progresses."""
        if new_idx < 0 or new_idx > self._max_idx:
            raise ValueError(f"Index {new_idx} out of bounds [0, {self._max_idx}]")
        self._current_idx = new_idx

    @property
    def current_idx(self) -> int:
        """Current simulation index."""
        return self._current_idx

    def __getitem__(self, key: Any) -> pd.DataFrame | pd.Series:
        """Access data with lookahead protection."""
        if isinstance(key, slice):
            # Check slice bounds
            start = key.start or 0
            stop = key.stop

            if stop is not None and stop > self._current_idx + 1:
                raise LookaheadError(
                    f"Attempted to access future data at index {stop}, "
                    f"current index is {self._current_idx}"
                )

            return self._data.iloc[:self._current_idx + 1][key]

        elif isinstance(key, int):
            if key > self._current_idx:
                raise LookaheadError(
                    f"Attempted to access future data at index {key}, "
                    f"current index is {self._current_idx}"
                )
            return self._data.iloc[key]

        elif isinstance(key, str):
            # Column access - return guarded series
            return GuardedSeries(self._data[key], self._current_idx)

        else:
            raise TypeError(f"Invalid key type: {type(key)}")

    def __getattr__(self, name: str) -> "GuardedSeries":
        """Attribute access for columns."""
        if name.startswith("_"):
            raise AttributeError(name)

        if name in self._data.columns:
            return GuardedSeries(self._data[name], self._current_idx)

        raise AttributeError(f"Column '{name}' not found")

    @property
    def close(self) -> "GuardedSeries":
        """Get guarded close prices."""
        return GuardedSeries(self._data["close"], self._current_idx)

    @property
    def open(self) -> "GuardedSeries":
        """Get guarded open prices."""
        return GuardedSeries(self._data["open"], self._current_idx)

    @property
    def high(self) -> "GuardedSeries":
        """Get guarded high prices."""
        return GuardedSeries(self._data["high"], self._current_idx)

    @property
    def low(self) -> "GuardedSeries":
        """Get guarded low prices."""
        return GuardedSeries(self._data["low"], self._current_idx)

    @property
    def volume(self) -> "GuardedSeries":
        """Get guarded volume."""
        return GuardedSeries(self._data["volume"], self._current_idx)


class GuardedSeries:
    """
    Guarded series that prevents lookahead access.

    Wraps a pandas Series and restricts operations to historical data.
    """

    def __init__(self, series: pd.Series, current_idx: int):
        self._series = series
        self._current_idx = current_idx

    def __getitem__(self, key: Any) -> Any:
        """Access with lookahead protection."""
        if isinstance(key, slice):
            stop = key.stop
            if stop is not None and stop > 0:
                raise LookaheadError(
                    f"Slice stop must be <= 0 for safe access, got {stop}"
                )
            # Negative indexing is safe
            return self._series.iloc[:self._current_idx + 1][key]

        elif isinstance(key, int):
            if key >= 0 and key > self._current_idx:
                raise LookaheadError(
                    f"Attempted to access future data at index {key}"
                )
            return self._series.iloc[:self._current_idx + 1].iloc[key]

        raise TypeError(f"Invalid key type: {type(key)}")

    def to_array(self) -> np.ndarray:
        """Get available data as array."""
        return self._series.iloc[:self._current_idx + 1].values

    def mean(self) -> float:
        """Calculate mean of available data."""
        return float(self._series.iloc[:self._current_idx + 1].mean())

    def std(self) -> float:
        """Calculate std of available data."""
        return float(self._series.iloc[:self._current_idx + 1].std())

    def min(self) -> float:
        """Get minimum of available data."""
        return float(self._series.iloc[:self._current_idx + 1].min())

    def max(self) -> float:
        """Get maximum of available data."""
        return float(self._series.iloc[:self._current_idx + 1].max())

    def __len__(self) -> int:
        """Return length of available data."""
        return self._current_idx + 1

    @property
    def values(self) -> np.ndarray:
        """Get values as array."""
        return self.to_array()
