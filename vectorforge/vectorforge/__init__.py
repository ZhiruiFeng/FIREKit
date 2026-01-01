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

from vectorforge.engine.base import BacktestEngine
from vectorforge.engine.vectorized import VectorizedBacktester
from vectorforge.engine.event_driven import EventDrivenBacktester
from vectorforge.engine.hybrid import HybridRunner
from vectorforge.strategy.base import BaseStrategy
from vectorforge.analysis.metrics import PerformanceMetrics
from vectorforge.config import VectorForgeConfig

__version__ = "0.1.0"
__author__ = "FIREKit Team"

__all__ = [
    # Engine
    "BacktestEngine",
    "VectorizedBacktester",
    "EventDrivenBacktester",
    "HybridRunner",
    # Strategy
    "BaseStrategy",
    # Analysis
    "PerformanceMetrics",
    # Config
    "VectorForgeConfig",
]
