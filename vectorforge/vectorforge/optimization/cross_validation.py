"""
Cross-Validation

Purged K-Fold and combinatorial cross-validation for time series.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    pass


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

    def split(self, data: pd.DataFrame | np.ndarray) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """
        Generate train/test indices for each fold.

        Uses walk-forward approach: for each fold, training data comes before
        test data with an embargo gap. Each successive fold has more training data.

        Args:
            data: Data to split

        Yields:
            Tuple of (train_indices, test_indices) for each fold
        """
        n_samples = len(data)
        # Calculate test size - divide remaining samples (after minimum train) into n_splits
        min_train_size = max(1, self.embargo_period + self.purge_period + 1)
        test_size = (n_samples - min_train_size) // self.n_splits

        for fold in range(self.n_splits):
            # Test set starts after enough training data for this fold
            # Each fold adds one test_size worth of training data
            test_start = min_train_size + fold * test_size + self.embargo_period + self.purge_period
            test_end = test_start + test_size
            if fold == self.n_splits - 1:
                test_end = n_samples  # Last fold gets remaining samples

            # Training set: everything before test, minus embargo/purge
            train_end = test_start - self.embargo_period - self.purge_period

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

    def split(self, data: pd.DataFrame | np.ndarray) -> Iterator[tuple[np.ndarray, np.ndarray]]:
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
            test_set = set(test_indices)

            # Training: all groups not in test, with embargo applied at sample level
            train_group_indices = [i for i in range(self.n_splits) if i not in test_group_indices]

            # Collect all training indices and apply embargo
            all_train = np.concatenate([groups[i] for i in train_group_indices])

            # Apply embargo: remove samples within embargo_period of any test sample
            if self.embargo_period > 0:
                # Create embargo zones around test indices
                embargo_mask = np.ones(len(all_train), dtype=bool)
                for test_idx in test_indices:
                    # Mark samples within embargo distance as excluded
                    distances = np.abs(all_train - test_idx)
                    embargo_mask &= distances > self.embargo_period
                train_indices = all_train[embargo_mask]
            else:
                train_indices = all_train

            train_indices = np.sort(train_indices)

            if len(train_indices) > 0:
                yield train_indices, test_indices

    def get_n_splits(self) -> int:
        """Return number of combinations."""
        from math import comb

        return comb(self.n_splits, self.n_test_splits)
