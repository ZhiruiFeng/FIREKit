# ExecutionCore

Order management and trading execution for FIREKit: a validated order
lifecycle, a deterministic paper-broker fill simulator, TWAP/VWAP execution
algos, pre-trade risk checks, and implementation-shortfall analytics.

## What's inside

| Module | Contents |
|---|---|
| `executioncore.orders` | `Order` (market / limit / stop / stop-limit), `TimeInForce` (DAY, GTC, IOC, FOK), validated state machine `NEW -> ACCEPTED -> PARTIALLY_FILLED -> FILLED / CANCELLED / REJECTED / EXPIRED` |
| `executioncore.market` | `Bar`, `PriceFeed`, deterministic synthetic intraday feed generator |
| `executioncore.costs` | Slippage models (fixed bps, square-root volume impact) and commission models (per-share, percentage, tiered) |
| `executioncore.broker` | `Broker` ABC + `PaperBroker`: bar-by-bar matching with configurable latency (in bars), volume-participation partial fills, TIF semantics |
| `executioncore.oms` | `OrderManager`: submit/cancel/replace, average-cost positions (long & short), realized/unrealized PnL, cash accounting, blotter audit trail |
| `executioncore.algos` | `TWAP` / `VWAP` parent-order slicing and `AlgoExecutor` scheduling |
| `executioncore.risk` | Pluggable pre-trade validators: `MaxOrderSize`, `MaxNotional`, `PriceBand` (fat-finger guard) |
| `executioncore.analytics` | Fill rate, slippage vs arrival, Perold implementation shortfall (delay / trading / opportunity), per-algo comparison |

## Quick start

```python
from executioncore import (
    FixedBpsSlippage, Order, OrderManager, OrderSide, PaperBroker,
    PerShareCommission, synthetic_intraday_feed,
)

feed = synthetic_intraday_feed(n_bars=390, seed=7)
broker = PaperBroker(feed, slippage=FixedBpsSlippage(2.0),
                     commission=PerShareCommission(0.005), participation_cap=0.1)
oms = OrderManager(broker, cash=1_000_000)

oms.submit(Order("SYNTH", OrderSide.BUY, 500))
broker.advance()                      # process the next bar -> fill
print(oms.snapshot())
```

## Demo

```bash
cd executioncore/ && python3 -m executioncore.demo
```

Writes `hub/data/executioncore.json` (hub schema v1): a full simulated
session of order flow with fill-rate / slippage / shortfall metrics, a
TWAP-vs-VWAP comparison, slippage distribution, and a blotter excerpt.

## Tests

```bash
cd executioncore/ && python3 -m pytest tests -q
```

## Design-doc substitutions (offline MVP)

- Live broker adapters (Alpaca / IBKR / CCXT) are replaced by `PaperBroker`
  behind a `Broker` ABC, so real adapters can plug in later.
- The async event-streaming engine is replaced by a deterministic
  bar-clock simulation (`advance()` / `run_session`).
- Iceberg execution is out of scope for the MVP; TWAP and VWAP are included.
