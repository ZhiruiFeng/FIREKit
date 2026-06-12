"""Pluggable pre-trade risk validators."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from executioncore.orders import Order


@dataclass
class RiskContext:
    """Snapshot of state available to pre-trade checks."""

    reference_price: float
    position_quantity: float = 0.0
    cash: float = 0.0
    equity: float = 0.0


@dataclass
class RiskCheckResult:
    ok: bool
    reason: str | None = None


class RiskValidator(ABC):
    """A pre-trade check that may reject an order before it reaches the broker."""

    name: str = "risk"

    @abstractmethod
    def validate(self, order: Order, context: RiskContext) -> RiskCheckResult:
        """Return ok=False with a reason to reject the order."""


class MaxOrderSize(RiskValidator):
    """Reject orders larger than ``max_quantity`` shares."""

    name = "max_order_size"

    def __init__(self, max_quantity: float):
        if max_quantity <= 0:
            raise ValueError("max_quantity must be positive")
        self.max_quantity = max_quantity

    def validate(self, order: Order, context: RiskContext) -> RiskCheckResult:
        if order.quantity > self.max_quantity:
            return RiskCheckResult(
                False,
                f"order size {order.quantity:g} exceeds max {self.max_quantity:g}",
            )
        return RiskCheckResult(True)


class MaxNotional(RiskValidator):
    """Reject orders whose estimated notional exceeds ``max_notional``."""

    name = "max_notional"

    def __init__(self, max_notional: float):
        if max_notional <= 0:
            raise ValueError("max_notional must be positive")
        self.max_notional = max_notional

    def validate(self, order: Order, context: RiskContext) -> RiskCheckResult:
        price = order.limit_price if order.limit_price is not None else context.reference_price
        notional = order.quantity * price
        if notional > self.max_notional:
            return RiskCheckResult(
                False,
                f"notional ${notional:,.0f} exceeds max ${self.max_notional:,.0f}",
            )
        return RiskCheckResult(True)


class PriceBand(RiskValidator):
    """Fat-finger guard: limit/stop prices must sit within a band around the mark.

    ``max_deviation_pct`` of 0.05 means prices more than 5% away from the
    reference price are rejected. Market orders always pass.
    """

    name = "price_band"

    def __init__(self, max_deviation_pct: float = 0.05):
        if max_deviation_pct <= 0:
            raise ValueError("max_deviation_pct must be positive")
        self.max_deviation_pct = max_deviation_pct

    def validate(self, order: Order, context: RiskContext) -> RiskCheckResult:
        ref = context.reference_price
        if ref <= 0:
            return RiskCheckResult(True)
        for label, price in (("limit", order.limit_price), ("stop", order.stop_price)):
            if price is None:
                continue
            deviation = abs(price - ref) / ref
            if deviation > self.max_deviation_pct:
                return RiskCheckResult(
                    False,
                    f"{label} price {price:g} deviates {deviation:.1%} from mark "
                    f"{ref:g} (band {self.max_deviation_pct:.1%})",
                )
        return RiskCheckResult(True)
