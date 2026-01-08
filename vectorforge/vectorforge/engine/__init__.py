"""
VectorForge Engine Module

Contains the core backtesting engine implementations:
- VectorizedBacktester: Fast array-based simulation for research
- EventDrivenBacktester: Production-grade sequential simulation
- HybridRunner: Automatic mode selection based on requirements
"""

from vectorforge.engine.base import BacktestEngine
from vectorforge.engine.event_driven import EventDrivenBacktester
from vectorforge.engine.hybrid import HybridRunner
from vectorforge.engine.vectorized import VectorizedBacktester

__all__ = [
    "BacktestEngine",
    "VectorizedBacktester",
    "EventDrivenBacktester",
    "HybridRunner",
]
