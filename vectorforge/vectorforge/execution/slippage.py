"""
Slippage Models

Models for simulating realistic execution costs.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass
class OrderInfo:
    """Information needed for slippage calculation."""
    price: float
    quantity: float
    side: str  # "buy" or "sell"
    volume: float  # Average daily volume
    volatility: float = 0.0  # Daily volatility


class SlippageModel(ABC):
    """Abstract base for slippage models."""

    @abstractmethod
    def calculate(self, order: OrderInfo) -> float:
        """
        Calculate slippage for an order.

        Args:
            order: Order information

        Returns:
            Slippage cost per share (positive value)
        """
        pass

    def apply(self, order: OrderInfo) -> float:
        """
        Apply slippage to get execution price.

        Args:
            order: Order information

        Returns:
            Adjusted execution price
        """
        slippage = self.calculate(order)
        if order.side == "buy":
            return order.price + slippage
        return order.price - slippage


class FixedSlippage(SlippageModel):
    """Fixed slippage in basis points."""

    def __init__(self, bps: float = 5.0):
        """
        Initialize fixed slippage.

        Args:
            bps: Slippage in basis points (1 bp = 0.01%)
        """
        self.bps = bps

    def calculate(self, order: OrderInfo) -> float:
        return order.price * self.bps / 10000


class PercentageSlippage(SlippageModel):
    """Percentage-based slippage."""

    def __init__(self, pct: float = 0.001):
        """
        Initialize percentage slippage.

        Args:
            pct: Slippage as decimal (0.001 = 0.1%)
        """
        self.pct = pct

    def calculate(self, order: OrderInfo) -> float:
        return order.price * self.pct


class VolumeSlippage(SlippageModel):
    """Volume-dependent slippage (square root impact)."""

    def __init__(self, base_bps: float = 5.0, impact_factor: float = 0.1):
        """
        Initialize volume-dependent slippage.

        Args:
            base_bps: Base slippage in bps
            impact_factor: Market impact coefficient
        """
        self.base_bps = base_bps
        self.impact_factor = impact_factor

    def calculate(self, order: OrderInfo) -> float:
        # Base slippage
        base = order.price * self.base_bps / 10000

        # Market impact based on participation rate
        if order.volume > 0:
            participation = order.quantity / order.volume
            impact = order.price * self.impact_factor * np.sqrt(participation)
        else:
            impact = 0

        return base + impact


class AlmgrenChrissSlippage(SlippageModel):
    """
    Almgren-Chriss market impact model.

    Academic model for optimal execution that considers
    both temporary and permanent price impact.
    """

    def __init__(
        self,
        eta: float = 0.01,  # Temporary impact coefficient
        gamma: float = 0.1,  # Permanent impact coefficient
    ):
        """
        Initialize Almgren-Chriss model.

        Args:
            eta: Temporary impact coefficient
            gamma: Permanent impact coefficient
        """
        self.eta = eta
        self.gamma = gamma

    def calculate(self, order: OrderInfo) -> float:
        if order.volume <= 0 or order.volatility <= 0:
            return order.price * 0.0005  # Default fallback

        participation = order.quantity / order.volume

        # Temporary impact (diminishes)
        temporary = self.eta * order.volatility * participation

        # Permanent impact (persists)
        permanent = self.gamma * order.volatility * np.sqrt(participation)

        return order.price * (temporary + permanent)


class SpreadSlippage(SlippageModel):
    """
    Bid-ask spread based slippage.

    Assumes crossing half the spread on each trade.
    """

    def __init__(self, half_spread_bps: float = 2.5):
        """
        Initialize spread slippage.

        Args:
            half_spread_bps: Half the bid-ask spread in bps
        """
        self.half_spread_bps = half_spread_bps

    def calculate(self, order: OrderInfo) -> float:
        return order.price * self.half_spread_bps / 10000


def create_slippage_model(
    model_type: str, **kwargs
) -> SlippageModel:
    """
    Factory function to create slippage models.

    Args:
        model_type: Type of model (fixed, percentage, volume, almgren_chriss, spread)
        **kwargs: Model-specific parameters

    Returns:
        SlippageModel instance
    """
    models = {
        "fixed": FixedSlippage,
        "percentage": PercentageSlippage,
        "volume": VolumeSlippage,
        "volume_dependent": VolumeSlippage,
        "almgren_chriss": AlmgrenChrissSlippage,
        "spread": SpreadSlippage,
    }

    if model_type.lower() not in models:
        raise ValueError(f"Unknown slippage model: {model_type}")

    return models[model_type.lower()](**kwargs)
