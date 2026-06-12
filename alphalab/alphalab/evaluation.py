"""Factor evaluation: IC/rank-IC time series, quantile portfolios, turnover,
and factor correlation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from alphalab.ops import cs_rank

MIN_CROSS_SECTION = 5  # minimum names per date for an IC observation


def forward_returns(close: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
    """Forward ``horizon``-day simple returns, aligned to signal date."""
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    return close.shift(-horizon) / close - 1.0


def _rowwise_corr(x: pd.DataFrame, y: pd.DataFrame) -> pd.Series:
    """Pearson correlation per row with pairwise NaN masking."""
    valid = x.notna() & y.notna()
    xv = x.where(valid)
    yv = y.where(valid)
    n = valid.sum(axis=1)
    xm = xv.sub(xv.mean(axis=1), axis=0)
    ym = yv.sub(yv.mean(axis=1), axis=0)
    cov = (xm * ym).sum(axis=1)
    denom = np.sqrt((xm**2).sum(axis=1) * (ym**2).sum(axis=1))
    corr = cov / denom.replace(0.0, np.nan)
    return corr.where(n >= MIN_CROSS_SECTION)


def ic_series(
    factor: pd.DataFrame, fwd: pd.DataFrame, method: str = "pearson"
) -> pd.Series:
    """Per-date information coefficient between factor values and forward returns.

    ``method='spearman'`` gives the rank IC.
    """
    if method not in {"pearson", "spearman"}:
        raise ValueError("method must be 'pearson' or 'spearman'")
    factor, fwd = factor.align(fwd, join="inner")
    if method == "spearman":
        valid = factor.notna() & fwd.notna()
        factor = factor.where(valid).rank(axis=1)
        fwd = fwd.where(valid).rank(axis=1)
    return _rowwise_corr(factor, fwd).dropna()


@dataclass(frozen=True)
class ICStats:
    mean: float
    std: float
    ir: float  #: mean / std (per-period information ratio)
    t_stat: float
    pct_positive: float
    n_obs: int

    @classmethod
    def from_series(cls, ic: pd.Series) -> ICStats:
        ic = ic.dropna()
        n = len(ic)
        if n == 0:
            return cls(np.nan, np.nan, np.nan, np.nan, np.nan, 0)
        mean = float(ic.mean())
        std = float(ic.std()) if n > 1 else np.nan
        ir = mean / std if std and np.isfinite(std) and std > 0 else np.nan
        t_stat = ir * np.sqrt(n) if np.isfinite(ir) else np.nan
        return cls(mean, std, ir, t_stat, float((ic > 0).mean()), n)


def quantile_returns(
    factor: pd.DataFrame, fwd: pd.DataFrame, n_quantiles: int = 5
) -> pd.DataFrame:
    """Per-date mean forward return of each factor quantile.

    Returns a frame indexed by date with columns ``Q1`` (lowest factor values)
    through ``Q<n>`` (highest).
    """
    if n_quantiles < 2:
        raise ValueError("n_quantiles must be >= 2")
    factor, fwd = factor.align(fwd, join="inner")
    valid = factor.notna() & fwd.notna()
    ranks = factor.where(valid).rank(axis=1, pct=True, method="first")
    buckets = np.ceil(ranks * n_quantiles).clip(1, n_quantiles)
    out = {}
    for q in range(1, n_quantiles + 1):
        out[f"Q{q}"] = fwd.where(buckets == q).mean(axis=1)
    frame = pd.DataFrame(out)
    return frame.dropna(how="all")


def quantile_spread(qret: pd.DataFrame) -> pd.Series:
    """Top-minus-bottom quantile return series (long high factor, short low)."""
    top = qret.columns[-1]
    return (qret[top] - qret["Q1"]).dropna()


def factor_turnover(factor: pd.DataFrame, top_pct: float = 0.2) -> float:
    """Average one-period churn of the top-``top_pct`` bucket.

    0 = the long portfolio never changes; 1 = it is fully replaced every period.
    """
    if not 0 < top_pct < 1:
        raise ValueError("top_pct must be in (0, 1)")
    ranks = factor.rank(axis=1, pct=True, method="first")
    in_top = ranks > (1.0 - top_pct)
    counts = in_top.sum(axis=1)
    usable = counts > 0
    if usable.sum() < 2:
        return float("nan")
    stayed = (in_top & in_top.shift(1)).sum(axis=1)
    prev_counts = counts.shift(1)
    turnover = 1.0 - stayed / prev_counts.where(prev_counts > 0)
    return float(turnover.dropna().mean())


def factor_correlation(factors: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Correlation matrix of factors, computed on stacked cross-sectional ranks
    (i.e. Spearman-style, scale-free)."""
    stacked = {}
    for name, values in factors.items():
        stacked[name] = cs_rank(values).stack()
    combined = pd.DataFrame(stacked)
    return combined.corr(method="pearson")


@dataclass
class FactorEvaluation:
    """Full evaluation result for one factor vs one forward-return horizon."""

    name: str
    category: str
    horizon: int
    ic: ICStats
    rank_ic: ICStats
    ic_timeseries: pd.Series
    rank_ic_timeseries: pd.Series
    quantile_mean_returns: pd.Series  #: index Q1..Qn, mean per-period return
    spread_mean: float  #: mean top-minus-bottom per-period return
    spread_timeseries: pd.Series
    turnover: float

    def summary(self) -> dict[str, float | str | int]:
        return {
            "factor": self.name,
            "category": self.category,
            "horizon": self.horizon,
            "ic_mean": self.ic.mean,
            "ic_std": self.ic.std,
            "ic_ir": self.ic.ir,
            "rank_ic_mean": self.rank_ic.mean,
            "rank_ic_ir": self.rank_ic.ir,
            "t_stat": self.rank_ic.t_stat,
            "pct_positive": self.rank_ic.pct_positive,
            "spread_mean": self.spread_mean,
            "turnover": self.turnover,
            "n_obs": self.rank_ic.n_obs,
        }


def evaluate_factor(
    name: str,
    values: pd.DataFrame,
    fwd: pd.DataFrame,
    *,
    category: str = "custom",
    horizon: int = 1,
    n_quantiles: int = 5,
) -> FactorEvaluation:
    """Evaluate one factor against pre-computed forward returns."""
    ic_ts = ic_series(values, fwd, method="pearson")
    rank_ic_ts = ic_series(values, fwd, method="spearman")
    qret = quantile_returns(values, fwd, n_quantiles)
    spread = quantile_spread(qret)
    return FactorEvaluation(
        name=name,
        category=category,
        horizon=horizon,
        ic=ICStats.from_series(ic_ts),
        rank_ic=ICStats.from_series(rank_ic_ts),
        ic_timeseries=ic_ts,
        rank_ic_timeseries=rank_ic_ts,
        quantile_mean_returns=qret.mean(),
        spread_mean=float(spread.mean()) if len(spread) else float("nan"),
        spread_timeseries=spread,
        turnover=factor_turnover(values),
    )
