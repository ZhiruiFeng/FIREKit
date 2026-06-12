"""Slippage and commission models for the paper broker."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod

from executioncore.market import Bar
from executioncore.orders import OrderSide


class SlippageModel(ABC):
    """Returns an adverse price fraction; fill price moves against the order."""

    @abstractmethod
    def slippage_fraction(self, side: OrderSide, quantity: float, bar: Bar) -> float:
        """Non-negative fraction of price (e.g. 0.0005 = 5 bps adverse)."""

    def fill_price(self, ref_price: float, side: OrderSide, quantity: float, bar: Bar) -> float:
        frac = self.slippage_fraction(side, quantity, bar)
        return ref_price * (1.0 + side.sign * frac)


class NoSlippage(SlippageModel):
    def slippage_fraction(self, side: OrderSide, quantity: float, bar: Bar) -> float:
        return 0.0


class FixedBpsSlippage(SlippageModel):
    """Constant adverse slippage in basis points."""

    def __init__(self, bps: float = 2.0):
        if bps < 0:
            raise ValueError("bps must be non-negative")
        self.bps = bps

    def slippage_fraction(self, side: OrderSide, quantity: float, bar: Bar) -> float:
        return self.bps / 1e4


class VolumeImpactSlippage(SlippageModel):
    """Square-root market impact: ``impact_bps * sqrt(qty / bar_volume)`` bps.

    ``impact_bps`` is the adverse cost in bps when the fill equals one full
    bar of volume.
    """

    def __init__(self, impact_bps: float = 25.0):
        if impact_bps < 0:
            raise ValueError("impact_bps must be non-negative")
        self.impact_bps = impact_bps

    def slippage_fraction(self, side: OrderSide, quantity: float, bar: Bar) -> float:
        if bar.volume <= 0:
            return self.impact_bps / 1e4
        return (self.impact_bps / 1e4) * math.sqrt(quantity / bar.volume)


class CommissionModel(ABC):
    @abstractmethod
    def commission(self, quantity: float, price: float) -> float:
        """Commission in dollars for a fill of ``quantity`` at ``price``."""


class NoCommission(CommissionModel):
    def commission(self, quantity: float, price: float) -> float:
        return 0.0


class PerShareCommission(CommissionModel):
    """Fixed rate per share with an optional per-fill minimum."""

    def __init__(self, rate: float = 0.005, minimum: float = 0.0):
        if rate < 0 or minimum < 0:
            raise ValueError("rate and minimum must be non-negative")
        self.rate = rate
        self.minimum = minimum

    def commission(self, quantity: float, price: float) -> float:
        return max(quantity * self.rate, self.minimum)


class PercentageCommission(CommissionModel):
    """Commission as a fraction of fill notional (e.g. 0.001 = 10 bps)."""

    def __init__(self, rate: float = 0.001):
        if rate < 0:
            raise ValueError("rate must be non-negative")
        self.rate = rate

    def commission(self, quantity: float, price: float) -> float:
        return quantity * price * self.rate


class TieredCommission(CommissionModel):
    """Per-share rate selected by fill size.

    ``tiers`` is a list of ``(max_quantity, per_share_rate)`` sorted by
    ascending ``max_quantity``; the last tier's threshold may be ``inf``.
    The rate of the first tier whose threshold covers the fill quantity is
    applied to the whole fill.
    """

    def __init__(self, tiers: list[tuple[float, float]] | None = None, minimum: float = 0.0):
        self.tiers = tiers or [(300.0, 0.0065), (3_000.0, 0.005), (float("inf"), 0.0035)]
        if any(r < 0 for _, r in self.tiers):
            raise ValueError("rates must be non-negative")
        if sorted(t for t, _ in self.tiers) != [t for t, _ in self.tiers]:
            raise ValueError("tiers must be sorted by ascending threshold")
        self.minimum = minimum

    def commission(self, quantity: float, price: float) -> float:
        rate = self.tiers[-1][1]
        for threshold, tier_rate in self.tiers:
            if quantity <= threshold:
                rate = tier_rate
                break
        return max(quantity * rate, self.minimum)
