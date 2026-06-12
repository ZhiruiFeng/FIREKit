# ExecutionCore: Live Trading & Order Management System

## Product Overview

**ExecutionCore** is the production trading infrastructure of the FIREKit ecosystem, handling live order execution, position management, and broker connectivity. It bridges the gap between backtested strategies and real-world trading with robust order handling, smart execution, and comprehensive monitoring.

### Key Value Propositions

- **Multi-Broker Support**: Alpaca, Interactive Brokers, crypto exchanges via CCXT
- **Smart Order Routing**: Optimal execution algorithms for various order sizes
- **Paper Trading First**: Identical API for paper and live trading
- **Real-Time Monitoring**: Order status, fills, P&L tracking
- **Failsafe Design**: Circuit breakers, position limits, emergency stops

## Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        ExecutionCore                                 │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                    Strategy Interface                        │    │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐                   │    │
│  │  │ SignalML │  │DeepTrader│  │  Manual  │                   │    │
│  │  │ Signals  │  │ Actions  │  │  Orders  │                   │    │
│  │  └────┬─────┘  └────┬─────┘  └────┬─────┘                   │    │
│  └───────┴─────────────┴─────────────┴─────────────────────────┘    │
│                              │                                       │
│  ┌───────────────────────────▼─────────────────────────────────┐    │
│  │                    Order Management                          │    │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐     │    │
│  │  │  Order   │  │ Position │  │   Risk   │  │Execution │     │    │
│  │  │  Queue   │  │ Manager  │  │  Check   │  │  Algo    │     │    │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────┘     │    │
│  └───────────────────────────┬─────────────────────────────────┘    │
│                              │                                       │
│  ┌───────────────────────────▼─────────────────────────────────┐    │
│  │                    Broker Adapters                           │    │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐     │    │
│  │  │  Alpaca  │  │   IBKR   │  │  Binance │  │   CCXT   │     │    │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────┘     │    │
│  └───────────────────────────┬─────────────────────────────────┘    │
│                              │                                       │
│  ┌───────────────────────────▼─────────────────────────────────┐    │
│  │                    Monitoring Layer                          │    │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐                   │    │
│  │  │  Order   │  │   P&L    │  │  Alert   │                   │    │
│  │  │ Tracker  │  │ Monitor  │  │  System  │                   │    │
│  │  └──────────┘  └──────────┘  └──────────┘                   │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Broker Comparison

| Broker | Asset Types | Latency | Commission | Best For |
|--------|-------------|---------|------------|----------|
| Alpaca | US Stocks, Crypto | ~1.5ms | $0 | Beginners, API-first |
| IBKR | Global, All Assets | ~5ms | $0.005/share | Professional |
| Binance | Crypto | <10ms | 0.1% | Crypto trading |
| Coinbase | Crypto | ~50ms | 0.5% | US Crypto |

## Technical Specification

### Order Types and Management

```python
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, List, Dict
import uuid

class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"

class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"

class OrderStatus(Enum):
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIAL = "partial"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"

class TimeInForce(Enum):
    DAY = "day"
    GTC = "gtc"  # Good till cancelled
    IOC = "ioc"  # Immediate or cancel
    FOK = "fok"  # Fill or kill
    OPG = "opg"  # At the open
    CLS = "cls"  # At the close

@dataclass
class Order:
    """Order representation."""
    symbol: str
    side: OrderSide
    quantity: float
    order_type: OrderType = OrderType.MARKET
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: TimeInForce = TimeInForce.DAY
    strategy_id: Optional[str] = None

    # Filled by system
    order_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0
    filled_price: float = 0
    commission: float = 0
    submitted_at: Optional[datetime] = None
    filled_at: Optional[datetime] = None

    @property
    def remaining_quantity(self) -> float:
        return self.quantity - self.filled_quantity

    @property
    def is_complete(self) -> bool:
        return self.status in [
            OrderStatus.FILLED,
            OrderStatus.CANCELLED,
            OrderStatus.REJECTED,
            OrderStatus.EXPIRED
        ]

@dataclass
class Position:
    """Current position in an asset."""
    symbol: str
    quantity: float
    avg_entry_price: float
    current_price: float
    unrealized_pnl: float
    realized_pnl: float
    side: str  # 'long' or 'short'

    @property
    def market_value(self) -> float:
        return abs(self.quantity) * self.current_price

    @property
    def cost_basis(self) -> float:
        return abs(self.quantity) * self.avg_entry_price


class OrderQueue:
    """Thread-safe order queue with priority handling."""

    def __init__(self):
        self.pending: List[Order] = []
        self.active: Dict[str, Order] = {}
        self.completed: List[Order] = []
        self._lock = asyncio.Lock()

    async def submit(self, order: Order):
        """Add order to queue."""
        async with self._lock:
            order.status = OrderStatus.PENDING
            self.pending.append(order)

    async def process_next(self) -> Optional[Order]:
        """Get next order to process."""
        async with self._lock:
            if not self.pending:
                return None
            order = self.pending.pop(0)
            order.status = OrderStatus.SUBMITTED
            self.active[order.order_id] = order
            return order

    async def update_fill(
        self,
        order_id: str,
        filled_qty: float,
        filled_price: float,
        commission: float
    ):
        """Update order with fill information."""
        async with self._lock:
            if order_id not in self.active:
                return

            order = self.active[order_id]
            order.filled_quantity = filled_qty
            order.filled_price = filled_price
            order.commission = commission

            if filled_qty >= order.quantity:
                order.status = OrderStatus.FILLED
                order.filled_at = datetime.now()
                self.completed.append(order)
                del self.active[order_id]
            else:
                order.status = OrderStatus.PARTIAL
```

### Broker Adapters

```python
from abc import ABC, abstractmethod
import asyncio
import aiohttp

class BrokerAdapter(ABC):
    """Abstract base class for broker integrations."""

    @abstractmethod
    async def connect(self):
        """Establish connection to broker."""
        pass

    @abstractmethod
    async def disconnect(self):
        """Close broker connection."""
        pass

    @abstractmethod
    async def submit_order(self, order: Order) -> str:
        """Submit order and return broker order ID."""
        pass

    @abstractmethod
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order."""
        pass

    @abstractmethod
    async def get_positions(self) -> List[Position]:
        """Get current positions."""
        pass

    @abstractmethod
    async def get_account(self) -> dict:
        """Get account information."""
        pass

    @abstractmethod
    async def stream_orders(self):
        """Stream order updates."""
        pass


class AlpacaAdapter(BrokerAdapter):
    """Alpaca Markets broker adapter."""

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        paper: bool = True
    ):
        self.api_key = api_key
        self.api_secret = api_secret
        self.paper = paper

        self.base_url = (
            "https://paper-api.alpaca.markets" if paper
            else "https://api.alpaca.markets"
        )
        self.data_url = "https://data.alpaca.markets"
        self.stream_url = (
            "wss://paper-api.alpaca.markets/stream" if paper
            else "wss://api.alpaca.markets/stream"
        )

        self._session = None
        self._ws = None

    async def connect(self):
        """Initialize connection."""
        self._session = aiohttp.ClientSession(
            headers={
                "APCA-API-KEY-ID": self.api_key,
                "APCA-API-SECRET-KEY": self.api_secret
            }
        )

    async def disconnect(self):
        """Close connections."""
        if self._ws:
            await self._ws.close()
        if self._session:
            await self._session.close()

    async def submit_order(self, order: Order) -> str:
        """Submit order to Alpaca."""
        payload = {
            "symbol": order.symbol,
            "qty": str(order.quantity),
            "side": order.side.value,
            "type": order.order_type.value,
            "time_in_force": order.time_in_force.value
        }

        if order.limit_price:
            payload["limit_price"] = str(order.limit_price)
        if order.stop_price:
            payload["stop_price"] = str(order.stop_price)

        async with self._session.post(
            f"{self.base_url}/v2/orders",
            json=payload
        ) as resp:
            if resp.status == 200:
                data = await resp.json()
                return data["id"]
            else:
                error = await resp.text()
                raise Exception(f"Order submission failed: {error}")

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel order."""
        async with self._session.delete(
            f"{self.base_url}/v2/orders/{order_id}"
        ) as resp:
            return resp.status == 204

    async def get_positions(self) -> List[Position]:
        """Get all positions."""
        async with self._session.get(
            f"{self.base_url}/v2/positions"
        ) as resp:
            data = await resp.json()
            return [
                Position(
                    symbol=p["symbol"],
                    quantity=float(p["qty"]),
                    avg_entry_price=float(p["avg_entry_price"]),
                    current_price=float(p["current_price"]),
                    unrealized_pnl=float(p["unrealized_pl"]),
                    realized_pnl=0,  # Not provided directly
                    side="long" if float(p["qty"]) > 0 else "short"
                )
                for p in data
            ]

    async def get_account(self) -> dict:
        """Get account info."""
        async with self._session.get(
            f"{self.base_url}/v2/account"
        ) as resp:
            data = await resp.json()
            return {
                "cash": float(data["cash"]),
                "buying_power": float(data["buying_power"]),
                "portfolio_value": float(data["portfolio_value"]),
                "equity": float(data["equity"]),
                "last_equity": float(data["last_equity"]),
                "daytrade_count": int(data.get("daytrade_count", 0)),
                "pattern_day_trader": data.get("pattern_day_trader", False)
            }

    async def stream_orders(self):
        """Stream order updates via WebSocket."""
        import websockets

        async with websockets.connect(self.stream_url) as ws:
            # Authenticate
            auth_msg = {
                "action": "auth",
                "key": self.api_key,
                "secret": self.api_secret
            }
            await ws.send(json.dumps(auth_msg))

            # Subscribe to trade updates
            subscribe_msg = {
                "action": "listen",
                "data": {"streams": ["trade_updates"]}
            }
            await ws.send(json.dumps(subscribe_msg))

            # Stream updates
            async for msg in ws:
                data = json.loads(msg)
                if data.get("stream") == "trade_updates":
                    yield data["data"]


class IBKRAdapter(BrokerAdapter):
    """Interactive Brokers adapter using ib_insync."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 7497,  # 7497 for paper, 7496 for live
        client_id: int = 1
    ):
        self.host = host
        self.port = port
        self.client_id = client_id
        self.ib = None

    async def connect(self):
        """Connect to TWS/Gateway."""
        from ib_insync import IB
        self.ib = IB()
        await self.ib.connectAsync(self.host, self.port, self.client_id)

    async def disconnect(self):
        """Disconnect from TWS."""
        if self.ib:
            self.ib.disconnect()

    async def submit_order(self, order: Order) -> str:
        """Submit order to IBKR."""
        from ib_insync import Stock, MarketOrder, LimitOrder

        # Create contract
        contract = Stock(order.symbol, 'SMART', 'USD')

        # Create order
        if order.order_type == OrderType.MARKET:
            ib_order = MarketOrder(
                order.side.value.upper(),
                order.quantity
            )
        elif order.order_type == OrderType.LIMIT:
            ib_order = LimitOrder(
                order.side.value.upper(),
                order.quantity,
                order.limit_price
            )

        # Submit
        trade = self.ib.placeOrder(contract, ib_order)
        return str(trade.order.orderId)


class CCXTAdapter(BrokerAdapter):
    """Unified crypto exchange adapter using CCXT."""

    def __init__(
        self,
        exchange_id: str,
        api_key: str,
        api_secret: str,
        sandbox: bool = True
    ):
        import ccxt
        self.exchange_class = getattr(ccxt, exchange_id)
        self.exchange = self.exchange_class({
            'apiKey': api_key,
            'secret': api_secret,
            'sandbox': sandbox,
            'enableRateLimit': True
        })

    async def connect(self):
        """Load markets."""
        await asyncio.to_thread(self.exchange.load_markets)

    async def disconnect(self):
        """Close exchange connection."""
        pass

    async def submit_order(self, order: Order) -> str:
        """Submit order to exchange."""
        order_func = (
            self.exchange.create_market_order if order.order_type == OrderType.MARKET
            else self.exchange.create_limit_order
        )

        result = await asyncio.to_thread(
            order_func,
            order.symbol,
            order.side.value,
            order.quantity,
            order.limit_price if order.order_type == OrderType.LIMIT else None
        )

        return result['id']

    async def get_positions(self) -> List[Position]:
        """Get positions from exchange balance."""
        balance = await asyncio.to_thread(self.exchange.fetch_balance)

        positions = []
        for symbol, data in balance.get('total', {}).items():
            if data > 0 and symbol != 'USDT':
                # Get current price
                ticker = await asyncio.to_thread(
                    self.exchange.fetch_ticker,
                    f"{symbol}/USDT"
                )
                positions.append(Position(
                    symbol=f"{symbol}/USDT",
                    quantity=data,
                    avg_entry_price=0,  # Not tracked by exchange
                    current_price=ticker['last'],
                    unrealized_pnl=0,
                    realized_pnl=0,
                    side='long'
                ))

        return positions
```

### Execution Algorithms

```python
class ExecutionAlgorithm(ABC):
    """Base class for execution algorithms."""

    @abstractmethod
    async def execute(
        self,
        order: Order,
        broker: BrokerAdapter
    ) -> List[str]:
        """Execute order and return list of child order IDs."""
        pass


class TWAPExecution(ExecutionAlgorithm):
    """Time-Weighted Average Price execution."""

    def __init__(
        self,
        duration_minutes: int = 30,
        slice_count: int = 10
    ):
        self.duration_minutes = duration_minutes
        self.slice_count = slice_count

    async def execute(
        self,
        order: Order,
        broker: BrokerAdapter
    ) -> List[str]:
        """Execute order using TWAP."""
        slice_qty = order.quantity / self.slice_count
        interval = (self.duration_minutes * 60) / self.slice_count

        order_ids = []

        for i in range(self.slice_count):
            child_order = Order(
                symbol=order.symbol,
                side=order.side,
                quantity=slice_qty,
                order_type=OrderType.MARKET,
                strategy_id=order.order_id
            )

            order_id = await broker.submit_order(child_order)
            order_ids.append(order_id)

            if i < self.slice_count - 1:
                await asyncio.sleep(interval)

        return order_ids


class VWAPExecution(ExecutionAlgorithm):
    """Volume-Weighted Average Price execution."""

    def __init__(
        self,
        volume_profile: List[float] = None,
        duration_minutes: int = 60
    ):
        # Default: typical intraday volume profile
        self.volume_profile = volume_profile or [
            0.15, 0.10, 0.08, 0.07, 0.06,  # Morning
            0.05, 0.05, 0.05, 0.05, 0.05,  # Midday
            0.06, 0.08, 0.10, 0.15         # Afternoon
        ]
        self.duration_minutes = duration_minutes

    async def execute(
        self,
        order: Order,
        broker: BrokerAdapter
    ) -> List[str]:
        """Execute order following volume profile."""
        total_profile = sum(self.volume_profile)
        normalized = [v / total_profile for v in self.volume_profile]

        interval = (self.duration_minutes * 60) / len(normalized)
        order_ids = []

        for i, weight in enumerate(normalized):
            slice_qty = order.quantity * weight

            if slice_qty < 1:
                continue

            child_order = Order(
                symbol=order.symbol,
                side=order.side,
                quantity=slice_qty,
                order_type=OrderType.MARKET,
                strategy_id=order.order_id
            )

            order_id = await broker.submit_order(child_order)
            order_ids.append(order_id)

            if i < len(normalized) - 1:
                await asyncio.sleep(interval)

        return order_ids


class IcebergExecution(ExecutionAlgorithm):
    """Iceberg order execution - hide large order size."""

    def __init__(
        self,
        visible_qty: float,
        price_tolerance: float = 0.001
    ):
        self.visible_qty = visible_qty
        self.price_tolerance = price_tolerance

    async def execute(
        self,
        order: Order,
        broker: BrokerAdapter
    ) -> List[str]:
        """Execute as iceberg order."""
        remaining = order.quantity
        order_ids = []

        while remaining > 0:
            slice_qty = min(self.visible_qty, remaining)

            child_order = Order(
                symbol=order.symbol,
                side=order.side,
                quantity=slice_qty,
                order_type=OrderType.LIMIT,
                limit_price=order.limit_price,
                time_in_force=TimeInForce.IOC,
                strategy_id=order.order_id
            )

            order_id = await broker.submit_order(child_order)
            order_ids.append(order_id)

            # Wait for fill confirmation
            await asyncio.sleep(1)

            # Check if filled (simplified)
            remaining -= slice_qty

        return order_ids
```

### Position Manager

```python
class PositionManager:
    """Manage positions across strategies and brokers."""

    def __init__(self, broker: BrokerAdapter):
        self.broker = broker
        self.positions: Dict[str, Position] = {}
        self.strategy_positions: Dict[str, Dict[str, float]] = {}

    async def sync_positions(self):
        """Sync positions with broker."""
        broker_positions = await self.broker.get_positions()
        self.positions = {p.symbol: p for p in broker_positions}

    def update_from_fill(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        price: float,
        strategy_id: str = None
    ):
        """Update positions from order fill."""
        if symbol not in self.positions:
            self.positions[symbol] = Position(
                symbol=symbol,
                quantity=0,
                avg_entry_price=0,
                current_price=price,
                unrealized_pnl=0,
                realized_pnl=0,
                side='long'
            )

        pos = self.positions[symbol]

        if side == OrderSide.BUY:
            # Increase position
            new_qty = pos.quantity + quantity
            if pos.quantity > 0:
                pos.avg_entry_price = (
                    (pos.avg_entry_price * pos.quantity + price * quantity) / new_qty
                )
            else:
                pos.avg_entry_price = price
            pos.quantity = new_qty
        else:
            # Decrease/close position
            if pos.quantity > 0:
                realized = (price - pos.avg_entry_price) * min(quantity, pos.quantity)
                pos.realized_pnl += realized
            pos.quantity -= quantity

        # Track by strategy
        if strategy_id:
            if strategy_id not in self.strategy_positions:
                self.strategy_positions[strategy_id] = {}
            if symbol not in self.strategy_positions[strategy_id]:
                self.strategy_positions[strategy_id][symbol] = 0

            delta = quantity if side == OrderSide.BUY else -quantity
            self.strategy_positions[strategy_id][symbol] += delta

    def get_strategy_exposure(self, strategy_id: str) -> float:
        """Get total exposure for a strategy."""
        if strategy_id not in self.strategy_positions:
            return 0

        total = 0
        for symbol, qty in self.strategy_positions[strategy_id].items():
            if symbol in self.positions:
                total += abs(qty * self.positions[symbol].current_price)
        return total

    def get_total_exposure(self) -> float:
        """Get total portfolio exposure."""
        return sum(p.market_value for p in self.positions.values())
```

### Execution Engine

```python
class ExecutionEngine:
    """Main execution orchestrator."""

    def __init__(
        self,
        broker: BrokerAdapter,
        risk_guard,  # RiskGuard instance
        paper_mode: bool = True
    ):
        self.broker = broker
        self.risk_guard = risk_guard
        self.paper_mode = paper_mode

        self.order_queue = OrderQueue()
        self.position_manager = PositionManager(broker)
        self.execution_algos = {
            'market': None,  # Direct execution
            'twap': TWAPExecution(),
            'vwap': VWAPExecution(),
            'iceberg': IcebergExecution(visible_qty=100)
        }

        self._running = False

    async def start(self):
        """Start execution engine."""
        await self.broker.connect()
        await self.position_manager.sync_positions()
        self._running = True

        # Start background tasks
        asyncio.create_task(self._process_orders())
        asyncio.create_task(self._stream_updates())

    async def stop(self):
        """Stop execution engine."""
        self._running = False
        await self.broker.disconnect()

    async def submit_order(
        self,
        order: Order,
        execution_algo: str = 'market'
    ) -> str:
        """Submit order through risk checks."""
        # Risk check
        allowed, reason = await self.risk_guard.check_order(order)
        if not allowed:
            order.status = OrderStatus.REJECTED
            raise Exception(f"Order rejected: {reason}")

        order.submitted_at = datetime.now()
        await self.order_queue.submit(order)

        return order.order_id

    async def _process_orders(self):
        """Background order processing loop."""
        while self._running:
            order = await self.order_queue.process_next()

            if order:
                try:
                    algo = self.execution_algos.get(order.strategy_id)
                    if algo:
                        await algo.execute(order, self.broker)
                    else:
                        await self.broker.submit_order(order)
                except Exception as e:
                    order.status = OrderStatus.REJECTED
                    print(f"Order execution failed: {e}")

            await asyncio.sleep(0.1)

    async def _stream_updates(self):
        """Stream and process order updates."""
        async for update in self.broker.stream_orders():
            order_id = update.get('order', {}).get('id')
            event = update.get('event')

            if event == 'fill':
                filled_qty = float(update['order']['filled_qty'])
                filled_price = float(update['order']['filled_avg_price'])

                await self.order_queue.update_fill(
                    order_id, filled_qty, filled_price, 0
                )

                # Update positions
                self.position_manager.update_from_fill(
                    update['order']['symbol'],
                    OrderSide(update['order']['side']),
                    filled_qty,
                    filled_price
                )

    async def flatten_all(self):
        """Emergency: close all positions."""
        for symbol, position in self.position_manager.positions.items():
            if position.quantity != 0:
                order = Order(
                    symbol=symbol,
                    side=OrderSide.SELL if position.quantity > 0 else OrderSide.BUY,
                    quantity=abs(position.quantity),
                    order_type=OrderType.MARKET
                )
                await self.broker.submit_order(order)
```

## Configuration

```yaml
# executioncore.yaml
brokers:
  primary:
    type: alpaca
    api_key: ${ALPACA_API_KEY}
    api_secret: ${ALPACA_API_SECRET}
    paper: true

  backup:
    type: ibkr
    host: 127.0.0.1
    port: 7497

  crypto:
    type: ccxt
    exchange: binance
    api_key: ${BINANCE_API_KEY}
    api_secret: ${BINANCE_API_SECRET}
    sandbox: true

execution:
  default_algo: market
  algos:
    twap:
      duration_minutes: 30
      slice_count: 10
    vwap:
      duration_minutes: 60
    iceberg:
      visible_qty: 100

orders:
  default_time_in_force: day
  max_order_value: 50000
  max_daily_orders: 100

monitoring:
  log_all_orders: true
  alert_on_rejection: true
  heartbeat_interval: 60

failsafes:
  max_position_pct: 0.20
  daily_loss_limit: 0.05
  emergency_flatten: true
```

## Roadmap

### v1.0 (Core)
- [x] Alpaca adapter
- [x] Order management
- [x] Position tracking
- [x] Basic execution

### v1.1 (Multi-Broker)
- [ ] IBKR adapter
- [ ] CCXT integration
- [ ] Broker failover

### v1.2 (Smart Execution)
- [ ] TWAP/VWAP algorithms
- [ ] Iceberg orders
- [ ] Smart order routing

### v2.0 (Production)
- [ ] Real-time monitoring dashboard
- [ ] Alerting system
- [ ] Audit logging
- [ ] Performance analytics
