"""Quantile-bin discretizer: continuous observations -> tabular state ids."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


class Discretizer:
    """Maps observation vectors to integer state ids via per-dimension quantile bins.

    Bin edges are fit on (training) observations; the state id is the mixed-radix
    encoding of the per-dimension bin digits.
    """

    def __init__(self, n_bins: int | Sequence[int] = 3):
        self._n_bins_spec = n_bins
        self.n_bins_: np.ndarray | None = None
        self.edges_: list[np.ndarray] | None = None

    @property
    def is_fit(self) -> bool:
        return self.edges_ is not None

    @property
    def n_states(self) -> int:
        if self.n_bins_ is None:
            raise RuntimeError("discretizer is not fit")
        return int(np.prod(self.n_bins_))

    def fit(self, observations: np.ndarray) -> Discretizer:
        observations = np.asarray(observations, dtype=float)
        if observations.ndim != 2 or len(observations) == 0:
            raise ValueError("observations must be a non-empty 2D array")
        n_dims = observations.shape[1]
        if isinstance(self._n_bins_spec, int):
            bins = np.full(n_dims, self._n_bins_spec, dtype=np.int64)
        else:
            bins = np.asarray(self._n_bins_spec, dtype=np.int64)
            if len(bins) != n_dims:
                raise ValueError(f"n_bins has {len(bins)} entries for {n_dims} dims")
        if (bins < 1).any():
            raise ValueError("every dimension needs at least 1 bin")
        self.n_bins_ = bins
        self.edges_ = []
        for d in range(n_dims):
            quantiles = np.linspace(0, 1, bins[d] + 1)[1:-1]
            self.edges_.append(np.quantile(observations[:, d], quantiles))
        return self

    def transform(self, obs: np.ndarray) -> int:
        if not self.is_fit:
            raise RuntimeError("discretizer is not fit")
        obs = np.asarray(obs, dtype=float)
        state = 0
        for d, edges in enumerate(self.edges_):
            digit = int(np.searchsorted(edges, obs[d], side="right"))
            state = state * int(self.n_bins_[d]) + digit
        return state

    def transform_batch(self, observations: np.ndarray) -> np.ndarray:
        return np.array([self.transform(o) for o in np.asarray(observations, dtype=float)])

    def fit_transform(self, observations: np.ndarray) -> np.ndarray:
        return self.fit(observations).transform_batch(observations)

    # ------------------------------------------------------- (de)serialisation
    def to_dict(self) -> dict:
        if not self.is_fit:
            raise RuntimeError("discretizer is not fit")
        return {
            "n_bins": self.n_bins_.tolist(),
            "edges": [e.tolist() for e in self.edges_],
        }

    @classmethod
    def from_dict(cls, payload: dict) -> Discretizer:
        disc = cls(n_bins=list(payload["n_bins"]))
        disc.n_bins_ = np.asarray(payload["n_bins"], dtype=np.int64)
        disc.edges_ = [np.asarray(e, dtype=float) for e in payload["edges"]]
        return disc
