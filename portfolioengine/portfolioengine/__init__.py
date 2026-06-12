"""PortfolioEngine: asset allocation and portfolio optimization.

Public API:

- Optimizers (common ``allocate(mu, cov)`` interface): :class:`EqualWeight`,
  :class:`InverseVolatility`, :class:`MinimumVariance`, :class:`MaxSharpe`,
  :class:`RiskParity`, :class:`HierarchicalRiskParity`
- Covariance estimation: :func:`sample_cov`, :func:`ledoit_wolf_cov`,
  :func:`ewma_cov`
- :func:`efficient_frontier` target-return sweep
- :func:`run_backtest` rolling re-estimation comparison harness
- :class:`Constraints` (long-only, max weight, sector caps)
- :func:`make_universe` synthetic block-correlated universes
"""

from portfolioengine.backtest import BacktestResult, run_backtest
from portfolioengine.constraints import Constraints, apply_weight_cap
from portfolioengine.covariance import (
    annualize_cov,
    ewma_cov,
    ledoit_wolf_cov,
    sample_cov,
    to_correlation,
)
from portfolioengine.frontier import FrontierPoint, efficient_frontier
from portfolioengine.optimizers import (
    DEFAULT_OPTIMIZERS,
    EqualWeight,
    HierarchicalRiskParity,
    InverseVolatility,
    MaxSharpe,
    MinimumVariance,
    Optimizer,
    RiskParity,
    portfolio_volatility,
    risk_contributions,
)
from portfolioengine.synthetic import Universe, make_universe

__version__ = "0.1.0"

__all__ = [
    "DEFAULT_OPTIMIZERS",
    "BacktestResult",
    "Constraints",
    "EqualWeight",
    "FrontierPoint",
    "HierarchicalRiskParity",
    "InverseVolatility",
    "MaxSharpe",
    "MinimumVariance",
    "Optimizer",
    "RiskParity",
    "Universe",
    "__version__",
    "annualize_cov",
    "apply_weight_cap",
    "efficient_frontier",
    "ewma_cov",
    "ledoit_wolf_cov",
    "make_universe",
    "portfolio_volatility",
    "risk_contributions",
    "run_backtest",
    "sample_cov",
    "to_correlation",
]
