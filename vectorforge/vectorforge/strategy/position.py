"""
Position Manager

Utilities for tracking and managing trading positions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from vectorforge.engine.event_driven import Fill


@dataclass
class Trade:
    """Represents a completed trade (entry + exit)."""
    symbol: str
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    quantity: float
    side: str  # "long" or "short"
    pnl: float
    pnl_pct: float
    commission: float
    holding_period: int  # in bars


@dataclass
class OpenPosition:
    """Represents an open position."""
    symbol: str
    quantity: float
    avg_entry_price: float
    entry_time: datetime
    unrealized_pnl: float = 0.0
    current_price: float = 0.0

    def update_price(self, price: float) -> None:
        """Update current price and unrealized P&L."""
        self.current_price = price
        self.unrealized_pnl = (price - self.avg_entry_price) * self.quantity


class PositionManager:
    """
    Manages position tracking for event-driven strategies.

    Handles:
    - Position entry/exit tracking
    - Average price calculation
    - Realized/unrealized P&L
    - Trade history

    Example:
        >>> pm = PositionManager()
        >>> pm.update(fill)  # Update on each fill
        >>> print(pm.get_position("AAPL"))
        >>> print(pm.realized_pnl)
    """

    def __init__(self):
        self._positions: dict[str, OpenPosition] = {}
        self._trades: list[Trade] = []
        self._pending_entries: dict[str, list[Fill]] = {}
        self._realized_pnl: float = 0.0
        self._total_commission: float = 0.0

    def update(self, fill: "Fill") -> Trade | None:
        """
        Update positions based on a fill.

        Args:
            fill: Order fill to process

        Returns:
            Completed Trade if position was closed, else None
        """
        symbol = fill.symbol
        quantity = fill.quantity if fill.side.value == "buy" else -fill.quantity

        self._total_commission += fill.commission

        if symbol not in self._positions:
            # New position
            self._positions[symbol] = OpenPosition(
                symbol=symbol,
                quantity=quantity,
                avg_entry_price=fill.fill_price,
                entry_time=fill.timestamp,
            )
            return None

        position = self._positions[symbol]
        old_qty = position.quantity

        # Check if this closes or adds to position
        if old_qty * quantity > 0:
            # Same direction - add to position
            total_cost = position.avg_entry_price * abs(old_qty) + fill.fill_price * abs(quantity)
            position.quantity += quantity
            position.avg_entry_price = total_cost / abs(position.quantity)
            return None

        else:
            # Opposite direction - close or reverse
            if abs(quantity) >= abs(old_qty):
                # Fully close (and possibly reverse)
                closed_qty = old_qty
                remaining = quantity + old_qty

                # Calculate P&L
                if old_qty > 0:  # Was long
                    pnl = (fill.fill_price - position.avg_entry_price) * abs(closed_qty)
                else:  # Was short
                    pnl = (position.avg_entry_price - fill.fill_price) * abs(closed_qty)

                self._realized_pnl += pnl

                # Create trade record
                trade = Trade(
                    symbol=symbol,
                    entry_time=position.entry_time,
                    exit_time=fill.timestamp,
                    entry_price=position.avg_entry_price,
                    exit_price=fill.fill_price,
                    quantity=abs(closed_qty),
                    side="long" if old_qty > 0 else "short",
                    pnl=pnl,
                    pnl_pct=pnl / (position.avg_entry_price * abs(closed_qty)),
                    commission=fill.commission,
                    holding_period=0,  # Would need bar counting
                )
                self._trades.append(trade)

                if abs(remaining) > 0.0001:
                    # Reverse position
                    position.quantity = remaining
                    position.avg_entry_price = fill.fill_price
                    position.entry_time = fill.timestamp
                else:
                    # Flat
                    del self._positions[symbol]

                return trade

            else:
                # Partial close
                closed_qty = quantity
                if old_qty > 0:
                    pnl = (fill.fill_price - position.avg_entry_price) * abs(closed_qty)
                else:
                    pnl = (position.avg_entry_price - fill.fill_price) * abs(closed_qty)

                self._realized_pnl += pnl
                position.quantity += quantity

                trade = Trade(
                    symbol=symbol,
                    entry_time=position.entry_time,
                    exit_time=fill.timestamp,
                    entry_price=position.avg_entry_price,
                    exit_price=fill.fill_price,
                    quantity=abs(closed_qty),
                    side="long" if old_qty > 0 else "short",
                    pnl=pnl,
                    pnl_pct=pnl / (position.avg_entry_price * abs(closed_qty)),
                    commission=fill.commission,
                    holding_period=0,
                )
                self._trades.append(trade)

                return trade

    def get_position(self, symbol: str) -> OpenPosition | None:
        """Get current position for a symbol."""
        return self._positions.get(symbol)

    def get_all_positions(self) -> dict[str, OpenPosition]:
        """Get all open positions."""
        return self._positions.copy()

    @property
    def realized_pnl(self) -> float:
        """Total realized P&L."""
        return self._realized_pnl

    @property
    def unrealized_pnl(self) -> float:
        """Total unrealized P&L across all positions."""
        return sum(p.unrealized_pnl for p in self._positions.values())

    @property
    def total_pnl(self) -> float:
        """Total P&L (realized + unrealized)."""
        return self._realized_pnl + self.unrealized_pnl

    @property
    def total_commission(self) -> float:
        """Total commission paid."""
        return self._total_commission

    @property
    def trades(self) -> list[Trade]:
        """List of completed trades."""
        return self._trades.copy()

    def reset(self) -> None:
        """Reset all position tracking."""
        self._positions.clear()
        self._trades.clear()
        self._pending_entries.clear()
        self._realized_pnl = 0.0
        self._total_commission = 0.0
