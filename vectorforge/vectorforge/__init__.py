"""
VectorForge: High-Performance Backtesting Engine

VectorForge is the foundational backtesting engine of the FIREKit ecosystem.
It implements a hybrid architecture combining vectorized operations for rapid
prototyping with event-driven simulation for production validation.

Key Features:
- 1000x faster vectorized mode for research and optimization
- Production-ready event-driven mode mirroring live trading
- JAX/GPU acceleration for parameter sweeps
- Built-in bias prevention (lookahead, survivorship)
- Comprehensive performance analytics

Example:
    >>> from vectorforge import VectorizedBacktester, MomentumStrategy
    >>> backtester = VectorizedBacktester()
    >>> strategy = MomentumStrategy(lookback=20)
    >>> results = backtester.run(strategy, price_data)
    >>> print(results.sharpe_ratio)
"""

from vectorforge.analysis.metrics import PerformanceMetrics
from vectorforge.config import VectorForgeConfig
from vectorforge.engine.base import BacktestEngine, PortfolioBacktestResult
from vectorforge.engine.event_driven import EventDrivenBacktester
from vectorforge.engine.hybrid import HybridRunner
from vectorforge.engine.vectorized import VectorizedBacktester
from vectorforge.portfolio.corporate_actions import Dividend, Split
from vectorforge.portfolio.data import MissingDataPolicy, PortfolioData, SymbolMetadata
from vectorforge.portfolio.metrics import PortfolioMetrics
from vectorforge.portfolio.rebalancer import RebalanceFrequency, Rebalancer
from vectorforge.portfolio.signals import CrossSectionalSignal, SignalResult, TargetWeights
from vectorforge.strategy.base import BaseStrategy, PortfolioStrategy

__version__ = "0.3.0"
__author__ = "FIREKit Team"

__all__ = [
    # Engine
    "BacktestEngine",
    "VectorizedBacktester",
    "EventDrivenBacktester",
    "HybridRunner",
    "PortfolioBacktestResult",
    # Strategy
    "BaseStrategy",
    "PortfolioStrategy",
    # Analysis
    "PerformanceMetrics",
    # Config
    "VectorForgeConfig",
    # Portfolio (v0.3.0)
    "PortfolioData",
    "MissingDataPolicy",
    "SymbolMetadata",
    # Signals (v0.3.0)
    "CrossSectionalSignal",
    "SignalResult",
    "TargetWeights",
    # Rebalancing (v0.3.0)
    "Rebalancer",
    "RebalanceFrequency",
    # Metrics (v0.3.0)
    "PortfolioMetrics",
    # Corporate Actions (v0.3.0)
    "Split",
    "Dividend",
]
