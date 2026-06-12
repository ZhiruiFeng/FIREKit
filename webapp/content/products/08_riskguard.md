# RiskGuard: Risk Management & Position Sizing

## Product Overview

**RiskGuard** is the risk management layer of the FIREKit ecosystem, providing position sizing algorithms, risk limits, and real-time monitoring to protect capital. It implements Kelly Criterion sizing, drawdown controls, and circuit breakers to ensure disciplined risk management.

### Key Value Propositions

- **Kelly-Based Sizing**: Mathematically optimal position sizing with configurable fractions
- **Real-Time Monitoring**: Track exposure, drawdown, and risk metrics continuously
- **Circuit Breakers**: Automatic risk reduction when thresholds are breached
- **Multi-Layer Controls**: Position, strategy, and portfolio-level limits
- **Regulatory Awareness**: PDT rules, margin requirements, concentration limits

## Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                          RiskGuard                                   │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                    Pre-Trade Checks                          │    │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐     │    │
│  │  │ Position │  │ Exposure │  │ Margin   │  │Concentration│    │    │
│  │  │  Limits  │  │  Limits  │  │  Check   │  │   Check   │    │    │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────┘     │    │
│  └───────────────────────────┬─────────────────────────────────┘    │
│                              │                                       │
│  ┌───────────────────────────▼─────────────────────────────────┐    │
│  │                    Position Sizing                           │    │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐     │    │
│  │  │  Kelly   │  │  Fixed   │  │ Volatility│  │  Equal  │     │    │
│  │  │ Criterion│  │Fractional│  │ Targeting │  │  Weight │     │    │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────┘     │    │
│  └───────────────────────────┬─────────────────────────────────┘    │
│                              │                                       │
│  ┌───────────────────────────▼─────────────────────────────────┐    │
│  │                   Real-Time Monitor                          │    │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐     │    │
│  │  │ Drawdown │  │  Daily   │  │   VaR    │  │ Exposure │     │    │
│  │  │ Tracker  │  │   P&L    │  │  Monitor │  │  Tracker │     │    │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────┘     │    │
│  └───────────────────────────┬─────────────────────────────────┘    │
│                              │                                       │
│  ┌───────────────────────────▼─────────────────────────────────┐    │
│  │                   Circuit Breakers                           │    │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐                   │    │
│  │  │ Drawdown │  │   Loss   │  │Emergency │                   │    │
│  │  │  Brake   │  │  Limit   │  │  Stop    │                   │    │
│  │  └──────────┘  └──────────┘  └──────────┘                   │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Risk Metrics

| Metric | Description | Typical Limit |
|--------|-------------|---------------|
| Max Position Size | % of portfolio per position | 5-10% |
| Max Sector Exposure | Total exposure to one sector | 25-30% |
| Max Drawdown | Peak-to-trough decline | 10-20% |
| Daily Loss Limit | Max loss in single day | 2-3% |
| VaR (95%) | Value at Risk | 2-5% daily |
| Beta Exposure | Portfolio beta to market | 0.5-1.5 |

## Technical Specification

### Position Sizing Algorithms

```python
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple
from abc import ABC, abstractmethod

@dataclass
class SizingInput:
    """Input for position sizing calculation."""
    signal_strength: float  # -1 to 1
    win_rate: float  # Historical win rate
    avg_win: float  # Average winning trade return
    avg_loss: float  # Average losing trade return (positive)
    volatility: float  # Asset volatility
    current_price: float
    portfolio_value: float
    current_position: float = 0


@dataclass
class SizingOutput:
    """Output from position sizing."""
    target_position: float
    position_value: float
    risk_amount: float
    sizing_method: str
    rationale: str


class PositionSizer(ABC):
    """Abstract base for position sizing algorithms."""

    @abstractmethod
    def calculate(self, input: SizingInput) -> SizingOutput:
        pass


class KellyCriterion(PositionSizer):
    """
    Kelly Criterion position sizing.

    Full Kelly: f* = (p * b - q) / b
    where:
        p = probability of winning
        q = probability of losing (1 - p)
        b = win/loss ratio

    Full Kelly maximizes geometric growth but creates high volatility.
    Use fractional Kelly (typically 0.25-0.5) for practical trading.
    """

    def __init__(
        self,
        fraction: float = 0.25,  # Quarter Kelly
        max_position_pct: float = 0.10,
        min_position_pct: float = 0.01
    ):
        self.fraction = fraction
        self.max_position_pct = max_position_pct
        self.min_position_pct = min_position_pct

    def calculate(self, input: SizingInput) -> SizingOutput:
        """Calculate Kelly-optimal position size."""
        p = input.win_rate
        q = 1 - p
        b = input.avg_win / input.avg_loss if input.avg_loss > 0 else 1

        # Full Kelly fraction
        kelly = (p * b - q) / b if b > 0 else 0

        # Apply fractional Kelly
        kelly_fraction = kelly * self.fraction

        # Clamp to limits
        kelly_fraction = max(0, kelly_fraction)
        kelly_fraction = min(kelly_fraction, self.max_position_pct)

        if kelly_fraction < self.min_position_pct:
            kelly_fraction = 0  # Don't take tiny positions

        # Apply signal strength
        position_pct = kelly_fraction * abs(input.signal_strength)

        # Calculate actual position
        position_value = input.portfolio_value * position_pct
        target_shares = position_value / input.current_price

        # Adjust for signal direction
        if input.signal_strength < 0:
            target_shares = -target_shares

        return SizingOutput(
            target_position=target_shares,
            position_value=position_value,
            risk_amount=position_value * input.volatility,
            sizing_method='kelly',
            rationale=f"Kelly={kelly:.2%}, Fractional={kelly_fraction:.2%}, Signal={input.signal_strength:.2f}"
        )


class FixedFractional(PositionSizer):
    """
    Fixed fractional position sizing.

    Risk a fixed percentage of portfolio per trade.
    Position size = (Portfolio * Risk%) / (Entry - Stop)
    """

    def __init__(
        self,
        risk_per_trade: float = 0.01,  # 1% risk per trade
        max_position_pct: float = 0.10
    ):
        self.risk_per_trade = risk_per_trade
        self.max_position_pct = max_position_pct

    def calculate(
        self,
        input: SizingInput,
        stop_loss_pct: float = 0.02
    ) -> SizingOutput:
        """Calculate position size based on stop loss."""
        risk_amount = input.portfolio_value * self.risk_per_trade
        position_value = risk_amount / stop_loss_pct

        # Apply max limit
        max_value = input.portfolio_value * self.max_position_pct
        position_value = min(position_value, max_value)

        target_shares = position_value / input.current_price

        if input.signal_strength < 0:
            target_shares = -target_shares

        return SizingOutput(
            target_position=target_shares,
            position_value=position_value,
            risk_amount=risk_amount,
            sizing_method='fixed_fractional',
            rationale=f"Risk={self.risk_per_trade:.2%}, StopLoss={stop_loss_pct:.2%}"
        )


class VolatilityTargeting(PositionSizer):
    """
    Volatility-targeted position sizing.

    Adjust position size to achieve target portfolio volatility.
    """

    def __init__(
        self,
        target_volatility: float = 0.15,  # 15% annualized
        lookback_days: int = 20,
        max_position_pct: float = 0.25
    ):
        self.target_volatility = target_volatility
        self.lookback_days = lookback_days
        self.max_position_pct = max_position_pct

    def calculate(self, input: SizingInput) -> SizingOutput:
        """Calculate position size for target volatility."""
        # Annualized to daily volatility
        daily_target = self.target_volatility / np.sqrt(252)
        asset_daily_vol = input.volatility / np.sqrt(252)

        # Weight to achieve target
        if asset_daily_vol > 0:
            weight = daily_target / asset_daily_vol
        else:
            weight = 0

        # Apply limits
        weight = min(weight, self.max_position_pct)

        position_value = input.portfolio_value * weight * abs(input.signal_strength)
        target_shares = position_value / input.current_price

        if input.signal_strength < 0:
            target_shares = -target_shares

        return SizingOutput(
            target_position=target_shares,
            position_value=position_value,
            risk_amount=position_value * asset_daily_vol,
            sizing_method='volatility_targeting',
            rationale=f"TargetVol={self.target_volatility:.1%}, AssetVol={input.volatility:.1%}, Weight={weight:.1%}"
        )


class EqualWeight(PositionSizer):
    """Simple equal-weight position sizing."""

    def __init__(
        self,
        num_positions: int = 20,
        max_position_pct: float = 0.10
    ):
        self.num_positions = num_positions
        self.max_position_pct = max_position_pct

    def calculate(self, input: SizingInput) -> SizingOutput:
        """Equal weight across positions."""
        weight = min(1.0 / self.num_positions, self.max_position_pct)

        position_value = input.portfolio_value * weight * abs(input.signal_strength)
        target_shares = position_value / input.current_price

        if input.signal_strength < 0:
            target_shares = -target_shares

        return SizingOutput(
            target_position=target_shares,
            position_value=position_value,
            risk_amount=position_value * input.volatility,
            sizing_method='equal_weight',
            rationale=f"Weight={weight:.1%}, NumPositions={self.num_positions}"
        )
```

### Risk Limits and Checks

```python
from dataclasses import dataclass
from typing import List, Dict, Optional
from enum import Enum

class RiskCheckResult(Enum):
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"


@dataclass
class RiskLimit:
    """Definition of a risk limit."""
    name: str
    limit_type: str  # 'hard' or 'soft'
    threshold: float
    current_value: float
    status: RiskCheckResult

    @property
    def utilization(self) -> float:
        return self.current_value / self.threshold if self.threshold > 0 else 0


class RiskLimits:
    """Collection of risk limits with enforcement."""

    def __init__(
        self,
        max_position_pct: float = 0.10,
        max_sector_pct: float = 0.25,
        max_exposure_pct: float = 1.0,
        max_drawdown_pct: float = 0.15,
        daily_loss_limit_pct: float = 0.03,
        max_orders_per_day: int = 100,
        max_leverage: float = 1.0
    ):
        self.max_position_pct = max_position_pct
        self.max_sector_pct = max_sector_pct
        self.max_exposure_pct = max_exposure_pct
        self.max_drawdown_pct = max_drawdown_pct
        self.daily_loss_limit_pct = daily_loss_limit_pct
        self.max_orders_per_day = max_orders_per_day
        self.max_leverage = max_leverage


class PreTradeRiskCheck:
    """Pre-trade risk validation."""

    def __init__(
        self,
        limits: RiskLimits,
        position_manager,  # From ExecutionCore
        portfolio_value: float
    ):
        self.limits = limits
        self.position_manager = position_manager
        self.portfolio_value = portfolio_value
        self.daily_orders = 0
        self.daily_pnl = 0

    def check_order(self, order) -> Tuple[bool, str]:
        """
        Validate order against all risk limits.

        Returns:
            Tuple of (allowed: bool, reason: str)
        """
        checks = [
            self._check_position_limit(order),
            self._check_exposure_limit(order),
            self._check_concentration(order),
            self._check_daily_orders(),
            self._check_daily_loss(),
            self._check_margin(order),
        ]

        for passed, reason in checks:
            if not passed:
                return False, reason

        return True, "All checks passed"

    def _check_position_limit(self, order) -> Tuple[bool, str]:
        """Check if order exceeds position limit."""
        order_value = order.quantity * order.limit_price if order.limit_price else 0

        # Estimate with current price if limit not set
        if order_value == 0:
            current_pos = self.position_manager.positions.get(order.symbol)
            if current_pos:
                order_value = order.quantity * current_pos.current_price

        position_pct = order_value / self.portfolio_value

        if position_pct > self.limits.max_position_pct:
            return False, f"Position size {position_pct:.1%} exceeds limit {self.limits.max_position_pct:.1%}"

        return True, ""

    def _check_exposure_limit(self, order) -> Tuple[bool, str]:
        """Check total portfolio exposure."""
        current_exposure = self.position_manager.get_total_exposure()
        order_exposure = order.quantity * (order.limit_price or 0)

        total_exposure_pct = (current_exposure + order_exposure) / self.portfolio_value

        if total_exposure_pct > self.limits.max_exposure_pct:
            return False, f"Total exposure {total_exposure_pct:.1%} would exceed limit"

        return True, ""

    def _check_concentration(self, order) -> Tuple[bool, str]:
        """Check sector/industry concentration."""
        # Simplified - would need sector data
        return True, ""

    def _check_daily_orders(self) -> Tuple[bool, str]:
        """Check daily order count."""
        if self.daily_orders >= self.limits.max_orders_per_day:
            return False, f"Daily order limit ({self.limits.max_orders_per_day}) reached"
        return True, ""

    def _check_daily_loss(self) -> Tuple[bool, str]:
        """Check daily loss limit."""
        loss_pct = -self.daily_pnl / self.portfolio_value if self.daily_pnl < 0 else 0

        if loss_pct > self.limits.daily_loss_limit_pct:
            return False, f"Daily loss limit reached ({loss_pct:.2%})"

        return True, ""

    def _check_margin(self, order) -> Tuple[bool, str]:
        """Check margin requirements."""
        # Would integrate with broker for actual margin check
        return True, ""

    def _check_pdt_rules(self, order) -> Tuple[bool, str]:
        """Check Pattern Day Trader rules (if applicable)."""
        # Track day trades in rolling 5-day window
        # Alert if approaching 4 day trades with <$25k equity
        return True, ""
```

### Real-Time Monitoring

```python
import asyncio
from datetime import datetime, timedelta
from collections import deque

class DrawdownMonitor:
    """Track and alert on portfolio drawdown."""

    def __init__(
        self,
        max_drawdown: float = 0.15,
        alert_threshold: float = 0.10
    ):
        self.max_drawdown = max_drawdown
        self.alert_threshold = alert_threshold

        self.peak_value = None
        self.current_value = None
        self.current_drawdown = 0
        self.drawdown_history = deque(maxlen=1000)

    def update(self, portfolio_value: float) -> Optional[str]:
        """Update with new portfolio value."""
        self.current_value = portfolio_value

        if self.peak_value is None:
            self.peak_value = portfolio_value
            return None

        # Update peak
        if portfolio_value > self.peak_value:
            self.peak_value = portfolio_value

        # Calculate drawdown
        self.current_drawdown = (self.peak_value - portfolio_value) / self.peak_value
        self.drawdown_history.append({
            'timestamp': datetime.now(),
            'drawdown': self.current_drawdown,
            'portfolio_value': portfolio_value,
            'peak': self.peak_value
        })

        # Check thresholds
        if self.current_drawdown >= self.max_drawdown:
            return f"CRITICAL: Max drawdown reached ({self.current_drawdown:.2%})"
        elif self.current_drawdown >= self.alert_threshold:
            return f"WARNING: Drawdown at {self.current_drawdown:.2%}"

        return None


class VaRMonitor:
    """Value at Risk monitoring."""

    def __init__(
        self,
        confidence_level: float = 0.95,
        lookback_days: int = 252,
        var_limit: float = 0.05
    ):
        self.confidence_level = confidence_level
        self.lookback_days = lookback_days
        self.var_limit = var_limit

        self.returns_history = deque(maxlen=lookback_days)

    def update(self, daily_return: float) -> Optional[str]:
        """Update with daily return."""
        self.returns_history.append(daily_return)

        if len(self.returns_history) < 20:
            return None

        var = self.calculate_var()

        if abs(var) > self.var_limit:
            return f"WARNING: VaR ({var:.2%}) exceeds limit ({self.var_limit:.2%})"

        return None

    def calculate_var(self) -> float:
        """Calculate historical VaR."""
        returns = np.array(self.returns_history)
        var = np.percentile(returns, (1 - self.confidence_level) * 100)
        return var

    def calculate_cvar(self) -> float:
        """Calculate Conditional VaR (Expected Shortfall)."""
        returns = np.array(self.returns_history)
        var = self.calculate_var()
        cvar = returns[returns <= var].mean()
        return cvar


class RealTimeRiskMonitor:
    """Aggregated real-time risk monitoring."""

    def __init__(
        self,
        limits: RiskLimits,
        position_manager,
        initial_portfolio_value: float
    ):
        self.limits = limits
        self.position_manager = position_manager
        self.initial_value = initial_portfolio_value

        self.drawdown_monitor = DrawdownMonitor(
            max_drawdown=limits.max_drawdown_pct
        )
        self.var_monitor = VaRMonitor()

        self.last_portfolio_value = initial_portfolio_value
        self.daily_start_value = initial_portfolio_value
        self.alerts = []

    async def run(self, update_interval: float = 1.0):
        """Continuous monitoring loop."""
        while True:
            await self.update()
            await asyncio.sleep(update_interval)

    async def update(self):
        """Update all monitors."""
        # Get current portfolio value
        await self.position_manager.sync_positions()
        current_value = self._calculate_portfolio_value()

        # Drawdown check
        dd_alert = self.drawdown_monitor.update(current_value)
        if dd_alert:
            self.alerts.append(dd_alert)
            await self._handle_alert(dd_alert)

        # Daily P&L check
        daily_return = (current_value - self.daily_start_value) / self.daily_start_value
        if daily_return < -self.limits.daily_loss_limit_pct:
            alert = f"CRITICAL: Daily loss limit breached ({daily_return:.2%})"
            self.alerts.append(alert)
            await self._handle_alert(alert)

        # VaR check (if we have position-level data)
        var_alert = self.var_monitor.update(
            (current_value - self.last_portfolio_value) / self.last_portfolio_value
        )
        if var_alert:
            self.alerts.append(var_alert)

        self.last_portfolio_value = current_value

    def _calculate_portfolio_value(self) -> float:
        """Calculate current portfolio value."""
        total = 0
        for pos in self.position_manager.positions.values():
            total += pos.market_value
        return total

    async def _handle_alert(self, alert: str):
        """Handle risk alert."""
        print(f"[RISK ALERT] {alert}")
        # Could integrate with notification system
```

### Circuit Breakers

```python
class CircuitBreaker:
    """Automated risk reduction when thresholds are breached."""

    def __init__(
        self,
        execution_engine,  # From ExecutionCore
        limits: RiskLimits
    ):
        self.execution = execution_engine
        self.limits = limits

        self.triggered = False
        self.trigger_time = None
        self.cooldown_minutes = 60

    async def check_and_act(
        self,
        current_drawdown: float,
        daily_loss: float
    ) -> bool:
        """Check conditions and trigger if needed."""
        if self.triggered:
            # Check if cooldown has passed
            if datetime.now() - self.trigger_time > timedelta(minutes=self.cooldown_minutes):
                self.triggered = False
            else:
                return True  # Still in cooldown

        # Check trigger conditions
        should_trigger = False
        reason = ""

        if current_drawdown >= self.limits.max_drawdown_pct:
            should_trigger = True
            reason = f"Max drawdown breached: {current_drawdown:.2%}"

        if daily_loss >= self.limits.daily_loss_limit_pct * 1.5:  # 150% of limit
            should_trigger = True
            reason = f"Severe daily loss: {daily_loss:.2%}"

        if should_trigger:
            await self._trigger(reason)
            return True

        return False

    async def _trigger(self, reason: str):
        """Trigger circuit breaker."""
        print(f"[CIRCUIT BREAKER] Triggered: {reason}")

        self.triggered = True
        self.trigger_time = datetime.now()

        # Action 1: Cancel all pending orders
        await self._cancel_all_orders()

        # Action 2: Reduce positions (not flatten completely)
        await self._reduce_positions(reduction_pct=0.5)

        # Action 3: Enter cooldown mode
        print(f"[CIRCUIT BREAKER] Cooldown for {self.cooldown_minutes} minutes")

    async def _cancel_all_orders(self):
        """Cancel all pending orders."""
        for order_id in list(self.execution.order_queue.active.keys()):
            await self.execution.broker.cancel_order(order_id)

    async def _reduce_positions(self, reduction_pct: float = 0.5):
        """Reduce all positions by percentage."""
        for symbol, position in self.execution.position_manager.positions.items():
            if position.quantity != 0:
                reduce_qty = abs(position.quantity) * reduction_pct

                order = Order(
                    symbol=symbol,
                    side=OrderSide.SELL if position.quantity > 0 else OrderSide.BUY,
                    quantity=reduce_qty,
                    order_type=OrderType.MARKET
                )

                await self.execution.submit_order(order)


class GradualDerisking:
    """Gradual position reduction based on drawdown level."""

    def __init__(self, limits: RiskLimits):
        self.limits = limits

        # Drawdown -> max exposure mapping
        self.derisk_schedule = [
            (0.05, 1.0),   # 5% DD -> 100% exposure allowed
            (0.10, 0.75),  # 10% DD -> 75% exposure
            (0.15, 0.50),  # 15% DD -> 50% exposure
            (0.20, 0.25),  # 20% DD -> 25% exposure
        ]

    def get_max_exposure(self, current_drawdown: float) -> float:
        """Get maximum allowed exposure for current drawdown."""
        for dd_threshold, max_exp in self.derisk_schedule:
            if current_drawdown < dd_threshold:
                return max_exp

        # Beyond all thresholds
        return 0.0

    def get_position_multiplier(self, current_drawdown: float) -> float:
        """Get multiplier for new position sizes."""
        max_exp = self.get_max_exposure(current_drawdown)
        return max_exp
```

## Configuration

```yaml
# riskguard.yaml
limits:
  position:
    max_single_position_pct: 0.10
    min_position_pct: 0.01

  exposure:
    max_total_exposure_pct: 1.0
    max_long_exposure_pct: 1.0
    max_short_exposure_pct: 0.5
    max_sector_pct: 0.25

  drawdown:
    alert_threshold_pct: 0.10
    max_drawdown_pct: 0.15

  daily:
    loss_limit_pct: 0.03
    max_orders: 100

  margin:
    max_leverage: 1.0
    maintenance_buffer: 0.25

sizing:
  method: kelly  # kelly, fixed_fractional, volatility_targeting, equal_weight
  kelly:
    fraction: 0.25
  fixed_fractional:
    risk_per_trade: 0.01
  volatility_targeting:
    target_vol: 0.15

circuit_breaker:
  enabled: true
  cooldown_minutes: 60
  actions:
    - cancel_pending_orders
    - reduce_positions_50pct

gradual_derisk:
  enabled: true
  schedule:
    - drawdown: 0.05
      max_exposure: 1.0
    - drawdown: 0.10
      max_exposure: 0.75
    - drawdown: 0.15
      max_exposure: 0.50

monitoring:
  update_interval_seconds: 1
  alert_channels:
    - console
    - email
```

## Roadmap

### v1.0 (Core)
- [x] Kelly Criterion sizing
- [x] Fixed fractional sizing
- [x] Position limits
- [x] Drawdown monitoring

### v1.1 (Advanced)
- [ ] Volatility targeting
- [ ] VaR monitoring
- [ ] Circuit breakers
- [ ] Gradual derisking

### v1.2 (Compliance)
- [ ] PDT rules tracking
- [ ] Margin monitoring
- [ ] Regulatory reporting

### v2.0 (Enterprise)
- [ ] Risk dashboard
- [ ] Historical risk analytics
- [ ] Stress testing
- [ ] Scenario analysis
