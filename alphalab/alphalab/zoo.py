"""FactorZoo: registry plus batch evaluation and ranking."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

import pandas as pd

from alphalab.evaluation import (
    FactorEvaluation,
    evaluate_factor,
    factor_correlation,
    forward_returns,
)
from alphalab.factors import Factor, default_factors
from alphalab.panel import Panel


@dataclass
class ZooReport:
    """Batch evaluation output, ranked by |rank-IC information ratio|."""

    horizon: int
    evaluations: list[FactorEvaluation]
    correlation: pd.DataFrame
    factor_values: dict[str, pd.DataFrame] = field(repr=False, default_factory=dict)

    @property
    def best(self) -> FactorEvaluation:
        return self.evaluations[0]

    def to_frame(self) -> pd.DataFrame:
        """Summary table, one row per factor, in ranked order."""
        rows = [ev.summary() for ev in self.evaluations]
        return pd.DataFrame(rows).set_index("factor")

    def top_correlated_pairs(self, n: int = 5) -> list[tuple[str, str, float]]:
        """The n most correlated factor pairs by |rho| (redundancy candidates)."""
        corr = self.correlation
        pairs = []
        names = list(corr.columns)
        for i, a in enumerate(names):
            for b in names[i + 1 :]:
                rho = float(corr.loc[a, b])
                if pd.notna(rho):
                    pairs.append((a, b, rho))
        pairs.sort(key=lambda p: -abs(p[2]))
        return pairs[:n]


class FactorZoo:
    """A named collection of factors that can be evaluated as a batch."""

    def __init__(self, factors: Iterable[Factor] | None = None) -> None:
        self._factors: dict[str, Factor] = {}
        for factor in factors or []:
            self.register(factor)

    @classmethod
    def default(cls) -> FactorZoo:
        """Zoo preloaded with the built-in factor library."""
        return cls(default_factors())

    def register(self, factor: Factor) -> Factor:
        if not isinstance(factor, Factor):
            raise TypeError(f"expected a Factor, got {type(factor).__name__}")
        if factor.name in self._factors:
            raise ValueError(f"factor already registered: {factor.name}")
        self._factors[factor.name] = factor
        return factor

    def __len__(self) -> int:
        return len(self._factors)

    def __contains__(self, name: str) -> bool:
        return name in self._factors

    def __getitem__(self, name: str) -> Factor:
        return self._factors[name]

    @property
    def names(self) -> list[str]:
        return list(self._factors)

    def compute(self, panel: Panel) -> dict[str, pd.DataFrame]:
        """Compute every registered factor on the panel."""
        return {name: factor.compute(panel) for name, factor in self._factors.items()}

    def evaluate(
        self,
        panel: Panel,
        horizon: int = 5,
        n_quantiles: int = 5,
    ) -> ZooReport:
        """Compute and evaluate all factors vs ``horizon``-day forward returns,
        ranked by |rank-IC IR| descending."""
        if not self._factors:
            raise ValueError("zoo is empty; register factors first")
        fwd = forward_returns(panel.close, horizon)
        values = self.compute(panel)
        evaluations = [
            evaluate_factor(
                name,
                vals,
                fwd,
                category=self._factors[name].category,
                horizon=horizon,
                n_quantiles=n_quantiles,
            )
            for name, vals in values.items()
        ]
        evaluations.sort(
            key=lambda ev: abs(ev.rank_ic.ir) if pd.notna(ev.rank_ic.ir) else -1.0,
            reverse=True,
        )
        return ZooReport(
            horizon=horizon,
            evaluations=evaluations,
            correlation=factor_correlation(values),
            factor_values=values,
        )
