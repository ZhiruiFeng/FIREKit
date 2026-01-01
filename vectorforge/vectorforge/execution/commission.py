"""
Commission Models

Models for simulating broker commission structures.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class TradeInfo:
    """Information needed for commission calculation."""
    shares: int
    price: float
    value: float = 0.0  # Trade value (shares * price)

    def __post_init__(self):
        if self.value == 0.0:
            self.value = self.shares * self.price


class CommissionModel(ABC):
    """Abstract base for commission models."""

    @abstractmethod
    def calculate(self, trade: TradeInfo) -> float:
        """
        Calculate commission for a trade.

        Args:
            trade: Trade information

        Returns:
            Commission amount
        """
        pass


class ZeroCommission(CommissionModel):
    """Zero commission model (Robinhood, Alpaca)."""

    def calculate(self, trade: TradeInfo) -> float:
        return 0.0


class FixedCommission(CommissionModel):
    """Fixed commission per trade."""

    def __init__(self, amount: float = 5.0):
        """
        Initialize fixed commission.

        Args:
            amount: Fixed commission per trade
        """
        self.amount = amount

    def calculate(self, trade: TradeInfo) -> float:
        return self.amount


class PerShareCommission(CommissionModel):
    """Per-share commission with minimums and caps."""

    def __init__(
        self,
        per_share: float = 0.005,
        min_commission: float = 1.0,
        max_pct: float = 0.01,
    ):
        """
        Initialize per-share commission.

        Args:
            per_share: Commission per share
            min_commission: Minimum commission per order
            max_pct: Maximum as percentage of trade value
        """
        self.per_share = per_share
        self.min_commission = min_commission
        self.max_pct = max_pct

    def calculate(self, trade: TradeInfo) -> float:
        commission = trade.shares * self.per_share
        commission = max(commission, self.min_commission)
        commission = min(commission, trade.value * self.max_pct)
        return commission


class PercentageCommission(CommissionModel):
    """Percentage of trade value commission."""

    def __init__(
        self,
        pct: float = 0.001,
        min_commission: float = 0.0,
    ):
        """
        Initialize percentage commission.

        Args:
            pct: Commission as percentage (0.001 = 0.1%)
            min_commission: Minimum commission
        """
        self.pct = pct
        self.min_commission = min_commission

    def calculate(self, trade: TradeInfo) -> float:
        return max(trade.value * self.pct, self.min_commission)


class TieredCommission(CommissionModel):
    """
    Interactive Brokers tiered pricing.

    Per-share rate that decreases with volume.
    """

    def __init__(self):
        self.min_commission = 0.35
        self.max_pct = 0.01

    def calculate(self, trade: TradeInfo) -> float:
        # Tiered rate based on share count
        if trade.shares <= 0:
            return 0.0

        per_share = max(0.0035, min(0.005, 1.0 / trade.shares))
        commission = per_share * trade.shares

        # Apply min and max
        commission = max(commission, self.min_commission)
        commission = min(commission, trade.value * self.max_pct)

        return commission


class CryptoCommission(CommissionModel):
    """Cryptocurrency exchange commission."""

    def __init__(
        self,
        maker_fee: float = 0.001,
        taker_fee: float = 0.002,
        is_taker: bool = True,
    ):
        """
        Initialize crypto commission.

        Args:
            maker_fee: Fee for limit orders (add liquidity)
            taker_fee: Fee for market orders (take liquidity)
            is_taker: Whether order is taker (market order)
        """
        self.maker_fee = maker_fee
        self.taker_fee = taker_fee
        self.is_taker = is_taker

    def calculate(self, trade: TradeInfo) -> float:
        fee_rate = self.taker_fee if self.is_taker else self.maker_fee
        return trade.value * fee_rate


def create_commission_model(
    model_type: str, **kwargs
) -> CommissionModel:
    """
    Factory function to create commission models.

    Args:
        model_type: Type of model
        **kwargs: Model-specific parameters

    Returns:
        CommissionModel instance
    """
    models = {
        "zero": ZeroCommission,
        "alpaca": ZeroCommission,
        "fixed": FixedCommission,
        "per_share": PerShareCommission,
        "percentage": PercentageCommission,
        "tiered": TieredCommission,
        "ibkr": TieredCommission,
        "crypto": CryptoCommission,
    }

    if model_type.lower() not in models:
        raise ValueError(f"Unknown commission model: {model_type}")

    return models[model_type.lower()](**kwargs)
