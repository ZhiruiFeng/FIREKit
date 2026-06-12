"""AlphaLab — factor mining workbench (offline MVP).

Public API:

- Panels: :class:`Panel`, :func:`make_synthetic_panel`
- Operators: :mod:`alphalab.ops`
- Factors: :class:`Factor`, the built-in library, :func:`default_factors`
- Evaluation: :func:`ic_series`, :func:`quantile_returns`, :func:`factor_turnover`,
  :func:`factor_correlation`, :func:`evaluate_factor`
- Batch: :class:`FactorZoo`, :class:`ZooReport`
"""

from alphalab import ops
from alphalab.evaluation import (
    FactorEvaluation,
    ICStats,
    evaluate_factor,
    factor_correlation,
    factor_turnover,
    forward_returns,
    ic_series,
    quantile_returns,
    quantile_spread,
)
from alphalab.factors import (
    AbnormalVolume,
    Alpha003,
    Alpha006,
    Alpha012,
    Alpha053,
    Alpha101,
    AmihudIlliquidity,
    Factor,
    High52WeekDistance,
    LowBeta,
    LowDownsideVolatility,
    LowVolatility,
    MarketCorrelation,
    Momentum,
    OvernightGap,
    PriceRange,
    ShortTermReversal,
    VolumeMomentum,
    default_factors,
)
from alphalab.panel import Panel, make_synthetic_panel
from alphalab.zoo import FactorZoo, ZooReport

__version__ = "0.1.0"

__all__ = [
    "AbnormalVolume",
    "Alpha003",
    "Alpha006",
    "Alpha012",
    "Alpha053",
    "Alpha101",
    "AmihudIlliquidity",
    "Factor",
    "FactorEvaluation",
    "FactorZoo",
    "High52WeekDistance",
    "ICStats",
    "LowBeta",
    "LowDownsideVolatility",
    "LowVolatility",
    "MarketCorrelation",
    "Momentum",
    "OvernightGap",
    "Panel",
    "PriceRange",
    "ShortTermReversal",
    "VolumeMomentum",
    "ZooReport",
    "default_factors",
    "evaluate_factor",
    "factor_correlation",
    "factor_turnover",
    "forward_returns",
    "ic_series",
    "make_synthetic_panel",
    "ops",
    "quantile_returns",
    "quantile_spread",
    "__version__",
]
