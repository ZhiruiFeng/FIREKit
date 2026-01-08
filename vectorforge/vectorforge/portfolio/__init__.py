"""
VectorForge Portfolio Module

Multi-asset portfolio support for VectorForge backtesting engine.

Key Components:
- PortfolioData: Container for multi-symbol OHLCV data with alignment
- CrossSectionalSignal: Cross-sectional signal generation
- Rebalancer: Portfolio rebalancing logic with turnover constraints
- PortfolioMetrics: Portfolio-level analytics and attribution

Example:
    >>> from vectorforge.portfolio import (
    ...     PortfolioData,
    ...     CrossSectionalSignal,
    ...     Rebalancer,
    ...     RebalanceFrequency,
    ... )
    >>> data = PortfolioData.from_dict({"AAPL": aapl_df, "GOOG": goog_df})
    >>> aligned = data.align().validate()
    >>> momentum = CrossSectionalSignal.momentum(lookback=252)
    >>> weights = momentum.generate(aligned).top_percentile(10).to_weights()
"""

from vectorforge.portfolio.corporate_actions import (
    CorporateAction,
    CorporateActionsRegistry,
    CorporateActionType,
    Dividend,
    Merger,
    Spinoff,
    Split,
)
from vectorforge.portfolio.data import (
    MissingDataPolicy,
    PortfolioData,
    SymbolMetadata,
)
from vectorforge.portfolio.metrics import (
    ConcentrationMetrics,
    DiversificationMetrics,
    PortfolioMetrics,
    ReturnAttribution,
    SectorExposure,
)
from vectorforge.portfolio.rebalancer import (
    CalendarTrigger,
    DriftTrigger,
    HybridTrigger,
    RebalanceFrequency,
    RebalanceOrders,
    Rebalancer,
    RebalanceResult,
    RebalanceTrigger,
)
from vectorforge.portfolio.signals import (
    CrossSectionalSignal,
    RankMethod,
    SignalResult,
    TargetWeights,
)

__all__ = [
    # Data container
    "PortfolioData",
    "MissingDataPolicy",
    "SymbolMetadata",
    # Signals
    "CrossSectionalSignal",
    "RankMethod",
    "SignalResult",
    "TargetWeights",
    # Rebalancing
    "CalendarTrigger",
    "DriftTrigger",
    "HybridTrigger",
    "RebalanceFrequency",
    "RebalanceOrders",
    "RebalanceResult",
    "Rebalancer",
    "RebalanceTrigger",
    # Metrics
    "ConcentrationMetrics",
    "DiversificationMetrics",
    "PortfolioMetrics",
    "ReturnAttribution",
    "SectorExposure",
    # Corporate Actions
    "CorporateAction",
    "CorporateActionType",
    "CorporateActionsRegistry",
    "Dividend",
    "Merger",
    "Spinoff",
    "Split",
]
