"""
Cross-Validation

Purged K-Fold and combinatorial cross-validation for time series.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Iterator

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from vectorforge.engine.base import BacktestEngine
    from vectorforge.strategy.base import BaseStrategy


@dataclass
class CVFold:
    """A single cross-validation fold."""
    fold_idx: int
    train_indices: np.ndarray
    test_indices: np.ndarray
    train_start: int
    train_end: int
    test_start: int
    test_end: int


class PurgedKFold:
    """
    Purged K-Fold cross-validation for time series.

    Prevents data leakage by:
    1. Ensuring train data comes before test data
    2. Adding an embargo period between train and test
    3. Purging overlapping samples

    Example:
        >>> cv = PurgedKFold(n_splits=5, embargo_period=5)
        >>> for train_idx, test_idx in cv.split(data):
        ...     train_data = data.iloc[train_idx]
        ...     test_data = data.iloc[test_idx]
        ...     # Train and validate
    """

    def __init__(
        self,
        n_splits: int = 5,
        embargo_period: int = 5,
        purge_period: int = 0,
    ):
        """
        Initialize purged K-Fold.

        Args:
            n_splits: Number of folds
            embargo_period: Number of samples to skip between train/test
            purge_period: Number of samples to purge from training
        """
        if n_splits < 2:
            raise ValueError("n_splits must be at least 2")

        self.n_splits = n_splits
        self.embargo_period = embargo_period
        self.purge_period = purge_period

    def split(
        self, data: pd.DataFrame | np.ndarray
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """
        Generate train/test indices for each fold.

        Args:
            data: Data to split

        Yields:
            Tuple of (train_indices, test_indices) for each fold
        """
        n_samples = len(data)
        fold_size = n_samples // self.n_splits

        for fold in range(self.n_splits):
            # Test set for this fold
            test_start = fold * fold_size
            test_end = (fold + 1) * fold_size if fold < self.n_splits - 1 else n_samples

            # Training set: everything before test, with embargo
            train_end = max(0, test_start - self.embargo_period - self.purge_period)

            # Create indices
            train_indices = np.arange(0, train_end)
            test_indices = np.arange(test_start, test_end)

            if len(train_indices) > 0 and len(test_indices) > 0:
                yield train_indices, test_indices

    def get_n_splits(self) -> int:
        """Return number of splits."""
        return self.n_splits


class CombinatorialPurgedKFold:
    """
    Combinatorial Purged K-Fold (CPCV).

    Tests multiple train/test combinations for more robust
    validation. Particularly useful for small datasets.

    Reference: "Advances in Financial Machine Learning" by M. López de Prado
    """

    def __init__(
        self,
        n_splits: int = 5,
        n_test_splits: int = 2,
        embargo_period: int = 5,
    ):
        """
        Initialize CPCV.

        Args:
            n_splits: Total number of groups
            n_test_splits: Number of groups to use as test
            embargo_period: Samples to skip between train/test
        """
        self.n_splits = n_splits
        self.n_test_splits = n_test_splits
        self.embargo_period = embargo_period

    def split(
        self, data: pd.DataFrame | np.ndarray
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """
        Generate train/test indices for each combination.

        Args:
            data: Data to split

        Yields:
            Tuple of (train_indices, test_indices)
        """
        from itertools import combinations

        n_samples = len(data)
        group_size = n_samples // self.n_splits

        # Create groups
        groups = []
        for i in range(self.n_splits):
            start = i * group_size
            end = (i + 1) * group_size if i < self.n_splits - 1 else n_samples
            groups.append(np.arange(start, end))

        # Generate all combinations of test groups
        for test_group_indices in combinations(range(self.n_splits), self.n_test_splits):
            test_indices = np.concatenate([groups[i] for i in test_group_indices])
            test_indices = np.sort(test_indices)

            # Training: all groups not in test
            train_group_indices = [i for i in range(self.n_splits) if i not in test_group_indices]

            # Apply embargo
            test_min = test_indices.min()
            test_max = test_indices.max()

            train_indices = []
            for i in train_group_indices:
                group = groups[i]
                # Remove samples too close to test
                if group.max() < test_min - self.embargo_period:
                    train_indices.append(group)
                elif group.min() > test_max + self.embargo_period:
                    train_indices.append(group)

            if train_indices:
                train_indices = np.concatenate(train_indices)
                train_indices = np.sort(train_indices)
                yield train_indices, test_indices

    def get_n_splits(self) -> int:
        """Return number of combinations."""
        from math import comb
        return comb(self.n_splits, self.n_test_splits)
