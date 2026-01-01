"""
VectorForge Strategy Module

Provides base classes and utilities for implementing trading strategies.
"""

from vectorforge.strategy.base import BaseStrategy, SignalType
from vectorforge.strategy.position import PositionManager

__all__ = [
    "BaseStrategy",
    "SignalType",
    "PositionManager",
]
