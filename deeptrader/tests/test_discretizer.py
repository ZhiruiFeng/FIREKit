"""Quantile discretizer: bins, state ids, serialization."""

import numpy as np
import pytest

from deeptrader.discretize import Discretizer


class TestDiscretizer:
    def test_two_bins_split_at_median(self):
        data = np.array([[1.0], [2.0], [3.0], [4.0]])
        disc = Discretizer(n_bins=2).fit(data)
        assert disc.n_states == 2
        assert disc.transform([0.0]) == 0
        assert disc.transform([2.0]) == 0  # below/at median edge (2.5)
        assert disc.transform([3.0]) == 1
        assert disc.transform([99.0]) == 1

    def test_mixed_radix_state_ids(self):
        # dim0: 2 bins split at 0; dim1: 3 bins at terciles of [0..8]
        data = np.column_stack(
            [np.array([-1.0, 1.0] * 5)[:10], np.linspace(0, 8, 10)]
        )
        disc = Discretizer(n_bins=[2, 3]).fit(data)
        assert disc.n_states == 6
        states = {disc.transform([x, y]) for x in (-2.0, 2.0) for y in (0.0, 4.0, 8.0)}
        assert states == set(range(6))

    def test_state_ids_always_in_range(self):
        rng = np.random.default_rng(0)
        data = rng.normal(size=(500, 3))
        disc = Discretizer(n_bins=4).fit(data)
        states = disc.transform_batch(rng.normal(size=(200, 3)) * 10)
        assert states.min() >= 0
        assert states.max() < disc.n_states == 64

    def test_transform_before_fit_raises(self):
        with pytest.raises(RuntimeError):
            Discretizer(3).transform([1.0])

    def test_dimension_mismatch_raises(self):
        with pytest.raises(ValueError):
            Discretizer(n_bins=[2, 2]).fit(np.zeros((10, 3)))

    def test_fit_transform_matches_individual(self):
        rng = np.random.default_rng(1)
        data = rng.normal(size=(100, 2))
        disc = Discretizer(3)
        batch = disc.fit_transform(data)
        individual = [disc.transform(row) for row in data]
        assert batch.tolist() == individual

    def test_serialization_roundtrip(self):
        rng = np.random.default_rng(2)
        data = rng.normal(size=(100, 3))
        disc = Discretizer(n_bins=[3, 4, 2]).fit(data)
        clone = Discretizer.from_dict(disc.to_dict())
        probe = rng.normal(size=(50, 3))
        assert clone.n_states == disc.n_states
        assert clone.transform_batch(probe).tolist() == disc.transform_batch(probe).tolist()
