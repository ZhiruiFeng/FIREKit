"""Portfolio constraints shared by all optimizers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Constraints:
    """Constraint set for portfolio construction.

    - ``long_only``: weights must be >= 0 (always combined with fully
      invested: weights sum to 1).
    - ``max_weight``: optional per-asset cap.
    - ``sector_caps``: optional ``{sector: cap}`` on the summed weight of
      each sector; requires ``sectors`` (one label per asset, in order).

    SLSQP-based optimizers enforce everything exactly. Analytic/heuristic
    optimizers (equal weight, inverse vol, risk parity, HRP) enforce
    ``max_weight`` via iterative clip-and-redistribute; sector caps are only
    supported by SLSQP-based optimizers.
    """

    long_only: bool = True
    max_weight: float | None = None
    sector_caps: dict[str, float] | None = None
    sectors: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if self.max_weight is not None and not 0.0 < self.max_weight <= 1.0:
            raise ValueError(f"max_weight must be in (0, 1], got {self.max_weight}")
        if self.sector_caps is not None and self.sectors is None:
            raise ValueError("sector_caps requires sectors (one label per asset)")

    def validate_dimension(self, n_assets: int) -> None:
        if self.sectors is not None and len(self.sectors) != n_assets:
            raise ValueError(
                f"sectors has {len(self.sectors)} labels for {n_assets} assets"
            )
        if self.max_weight is not None and self.max_weight * n_assets < 1.0 - 1e-12:
            raise ValueError(
                f"infeasible: max_weight={self.max_weight} cannot sum to 1 over {n_assets} assets"
            )

    def bounds(self, n_assets: int) -> list[tuple[float, float]]:
        """Per-asset bounds for scipy.optimize."""
        lo = 0.0 if self.long_only else -(self.max_weight or 1.0)
        hi = self.max_weight if self.max_weight is not None else 1.0
        return [(lo, hi)] * n_assets

    def scipy_constraints(self, n_assets: int) -> list[dict[str, Any]]:
        """Fully-invested equality plus sector-cap inequalities for SLSQP."""
        cons: list[dict[str, Any]] = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
        ]
        if self.sector_caps and self.sectors is not None:
            for sector, cap in self.sector_caps.items():
                mask = np.array([s == sector for s in self.sectors], dtype=float)
                cons.append(
                    {"type": "ineq", "fun": lambda w, m=mask, c=cap: c - float(m @ w)}
                )
        return cons


def apply_weight_cap(weights: np.ndarray, max_weight: float | None) -> np.ndarray:
    """Cap long-only weights and redistribute the excess proportionally.

    Iteratively clips weights above ``max_weight`` and renormalizes the
    uncapped remainder so the result still sums to 1.
    """
    w = np.asarray(weights, dtype=float).copy()
    if max_weight is None:
        return w
    n = w.size
    if max_weight * n < 1.0 - 1e-12:
        raise ValueError(f"infeasible cap: {max_weight} * {n} < 1")
    capped = np.zeros(n, dtype=bool)
    for _ in range(n):
        over = w > max_weight + 1e-15
        if not over.any():
            break
        capped |= over
        excess = float(np.sum(w[over] - max_weight))
        w[over] = max_weight
        free = ~capped
        if not free.any():
            break
        free_total = float(w[free].sum())
        if free_total <= 0.0:
            # Spread excess by remaining headroom when free weights are zero.
            headroom = max_weight - w[free]
            w[free] += excess * headroom / headroom.sum()
        else:
            w[free] += excess * w[free] / free_total
    return w
