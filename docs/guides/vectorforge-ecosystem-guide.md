# VectorForge: Why We Need It and How to Use It in FIREKit

## Introduction

VectorForge is the foundational backtesting engine at the core of the FIREKit ecosystem. This guide explains why VectorForge exists, what problems it solves, and how it integrates with other FIREKit products.

## Why Do We Need VectorForge?

### The Problem: Traditional Backtesting is Broken

Quantitative trading faces four critical challenges that existing tools fail to address:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      Traditional Backtesting Pain Points                 │
├─────────────────────────────────────────────────────────────────────────┤
│  🐌 SPEED: Event-driven frameworks take hours for parameter sweeps      │
│  🎭 ACCURACY: Research code differs from production, causing drift      │
│  🔮 BIAS: Lookahead and survivorship bias corrupt backtest results      │
│  🔧 COMPLEXITY: Steep learning curves for institutional-grade tools     │
└─────────────────────────────────────────────────────────────────────────┘
```

#### 1. The Speed Problem

Traditional event-driven backtesting is painfully slow:

| Operation | Traditional | VectorForge | Improvement |
|-----------|-------------|-------------|-------------|
| 10-year daily backtest | 2.3 seconds | 0.05 seconds | **46x faster** |
| 1000 parameter sweep | 38 minutes | 1.2 seconds | **1917x faster** |
| Monte Carlo 10k paths | 6+ hours | 8 seconds | **2700x faster** |

When testing 1000 parameter combinations takes 38 minutes, you can't iterate fast enough. Research becomes a bottleneck.

#### 2. The Accuracy Problem

Most backtesting tools optimize for speed OR accuracy, forcing you to choose:

```
Traditional Approach:
┌────────────────┐     ┌────────────────┐
│ Research Code  │ ──▶ │ Production Code │  ← Code rewrite required
│ (Vectorized)   │     │ (Event-driven)  │  ← Different logic paths
└────────────────┘     └────────────────┘  ← Bugs introduced
```

This causes "deployment drift" - strategies that work in research fail in production because the code is fundamentally different.

#### 3. The Bias Problem

Backtesting biases destroy returns:

- **Lookahead Bias**: Accidentally using future data (e.g., tomorrow's close to decide today's trade)
- **Survivorship Bias**: Only testing on stocks that still exist (ignoring bankruptcies)
- **Data Snooping**: Overfitting parameters to historical data

A strategy with 50% Sharpe from lookahead bias will have 0% Sharpe in production.

#### 4. The Complexity Problem

Institutional tools like QuantConnect, Zipline, or Backtrader have steep learning curves and don't integrate well with modern ML workflows (JAX, PyTorch, LLMs).

### The Solution: VectorForge's Hybrid Architecture

VectorForge solves all four problems with a dual-mode design:

```
┌─────────────────────────────────────────────────────────────────┐
│                        VectorForge                               │
├────────────────────────────┬────────────────────────────────────┤
│      Vectorized Mode       │       Event-Driven Mode            │
├────────────────────────────┼────────────────────────────────────┤
│  ✓ NumPy/JAX operations    │  ✓ Queue-based architecture        │
│  ✓ Parallel parameter sweep│  ✓ Realistic market simulation     │
│  ✓ 1M+ trades/second       │  ✓ Slippage & commission models    │
│  ✓ GPU acceleration        │  ✓ Order book dynamics             │
│  ✓ Fast research iteration │  ✓ Identical to live trading code  │
└────────────────────────────┴────────────────────────────────────┘

                    ↓ HybridRunner ↓

        Research fast → Validate accurately → Deploy confidently
```

## How VectorForge Fits in the FIREKit Ecosystem

### Architectural Position

VectorForge sits at the base of the FIREKit pyramid, providing the essential backtesting capability that all other products build upon:

```
┌─────────────────────────────────────────────────────────────────┐
│                     Portfolio Dashboard                          │  ← Visualization
├─────────────────────────────────────────────────────────────────┤
│  PortfolioEngine  │   RiskGuard   │   ExecutionCore              │  ← Deployment
├─────────────────────────────────────────────────────────────────┤
│  SignalML  │  AlphaLab  │  SentimentPulse  │  DeepTrader         │  ← Intelligence
├─────────────────────────────────────────────────────────────────┤
│                        DataStream                                │  ← Data
├─────────────────────────────────────────────────────────────────┤
│                       VectorForge ★                              │  ← Backtesting Core
└─────────────────────────────────────────────────────────────────┘
```

### Data Flow Through the Ecosystem

```
┌──────────────────────────────────────────────────────────────────────────┐
│                         Complete FIREKit Data Flow                        │
└──────────────────────────────────────────────────────────────────────────┘

  External APIs (Alpaca, Polygon, CoinGecko)
           │
           ▼
  ┌─────────────────┐
  │   DataStream    │ ──── Clean, normalized OHLCV data
  └────────┬────────┘
           │
           ▼
  ┌─────────────────┐
  │   VectorForge   │ ──── Backtest strategies with this data
  └────────┬────────┘
           │
     ┌─────┴─────┐
     ▼           ▼
┌─────────┐ ┌─────────┐
│AlphaLab │ │SignalML │ ──── Generate signals/features
└────┬────┘ └────┬────┘
     │           │
     └─────┬─────┘
           ▼
  ┌─────────────────┐
  │   RiskGuard     │ ──── Apply position sizing & risk controls
  └────────┬────────┘
           │
           ▼
  ┌─────────────────┐
  │  ExecutionCore  │ ──── Execute trades in production
  └────────┬────────┘
           │
           ▼
  ┌─────────────────┐
  │ PortfolioEngine │ ──── Manage multi-strategy portfolio
  └─────────────────┘
```

## Integration Examples

### 1. DataStream → VectorForge (Data to Backtest)

DataStream provides clean, normalized data that VectorForge consumes:

```python
from datastream import DataStream
from vectorforge import VectorizedBacktester, MomentumStrategy

# Load clean data from DataStream
data = DataStream.load(
    symbol='AAPL',
    start='2020-01-01',
    end='2024-01-01',
    adjust='split_and_dividend'  # Point-in-time adjusted
)

# Backtest with VectorForge
backtester = VectorizedBacktester()
strategy = MomentumStrategy(lookback=20)
results = backtester.run(strategy, data, initial_capital=100000)

print(results.summary())
# Total Return: 45.2%
# Sharpe Ratio: 1.35
# Max Drawdown: -12.4%
```

### 2. VectorForge + SignalML (ML-Powered Strategies)

Use ML models from SignalML within VectorForge backtests:

```python
from signalml import load_ensemble
from vectorforge import EventDrivenBacktester, BaseStrategy

# Load pre-trained ML model
model = load_ensemble("momentum_classifier")

class MLStrategy(BaseStrategy):
    """Strategy that uses ML predictions for trading signals."""

    def __init__(self, model, threshold=0.6):
        super().__init__()
        self.model = model
        self.threshold = threshold

    def on_bar(self, event):
        # Extract features from current bar
        features = self.extract_features(event.bar)

        # Get ML prediction
        prob_up = self.model.predict_proba(features)[0, 1]

        # Generate order based on prediction
        if prob_up > self.threshold and self.position <= 0:
            return self.create_order('BUY', quantity=100)
        elif prob_up < (1 - self.threshold) and self.position >= 0:
            return self.create_order('SELL', quantity=abs(self.position))

        return None

# Backtest the ML strategy
backtester = EventDrivenBacktester(
    slippage_model='volume_dependent',
    commission_model='tiered'
)
results = backtester.run(MLStrategy(model), data)
```

### 3. VectorForge + AlphaLab (Factor Research)

Research alpha factors with AlphaLab, then backtest with VectorForge:

```python
from alphalab import FactorLibrary, Alpha101
from vectorforge import VectorizedBacktester

# Generate alpha factors
factor_lib = FactorLibrary()
momentum_factor = factor_lib.momentum(lookback=20)
mean_reversion_factor = Alpha101.alpha_042()  # Intraday reversal

# Combine factors
combined_alpha = 0.6 * momentum_factor + 0.4 * mean_reversion_factor

# Create factor-based strategy
class FactorStrategy(BaseStrategy):
    def generate_signals(self, close, **kwargs):
        alpha = combined_alpha.compute(close)
        # Long top decile, short bottom decile
        return np.where(alpha > np.percentile(alpha, 90), 1,
                       np.where(alpha < np.percentile(alpha, 10), -1, 0))

# Fast parameter sweep
from vectorforge import HybridRunner

runner = HybridRunner()
param_results = runner.run_batch(
    strategy_class=FactorStrategy,
    param_grid={
        'lookback': range(10, 60, 5),
        'threshold': [0.5, 0.6, 0.7, 0.8]
    },
    data=data
)
```

### 4. VectorForge + RiskGuard (Risk-Controlled Backtesting)

Apply risk management during backtesting:

```python
from vectorforge import EventDrivenBacktester
from riskguard import RiskManager

# Configure risk manager
risk_manager = RiskManager(
    max_position_pct=0.10,      # Max 10% in any single position
    max_drawdown=0.20,          # Stop trading at 20% drawdown
    daily_loss_limit=0.03,      # Max 3% daily loss
    portfolio_heat=0.02         # 2% portfolio risk per trade
)

# Backtest with risk controls
backtester = EventDrivenBacktester()
backtester.set_risk_manager(risk_manager)

results = backtester.run(strategy, data)

# Results now include risk-adjusted metrics
print(f"Risk-adjusted Sharpe: {results.risk_adjusted_sharpe}")
print(f"Times stopped by risk: {results.risk_stops}")
```

### 5. VectorForge → ExecutionCore (Research to Production)

The same strategy code works in both VectorForge (backtest) and ExecutionCore (live):

```python
from vectorforge import EventDrivenBacktester
from executioncore import LiveExecutor

# Define strategy ONCE
class MyStrategy(BaseStrategy):
    def __init__(self, fast_period=10, slow_period=30):
        self.fast_period = fast_period
        self.slow_period = slow_period

    def on_bar(self, event):
        fast_ma = self.data.close[-self.fast_period:].mean()
        slow_ma = self.data.close[-self.slow_period:].mean()

        if fast_ma > slow_ma and self.position <= 0:
            return self.create_order('BUY', quantity=100)
        elif fast_ma < slow_ma and self.position >= 0:
            return self.create_order('SELL', quantity=abs(self.position))
        return None

# Step 1: Backtest with VectorForge
backtester = EventDrivenBacktester()
backtest_results = backtester.run(MyStrategy(), historical_data)

if backtest_results.sharpe > 1.5:
    # Step 2: Paper trade with ExecutionCore
    executor = LiveExecutor(broker='alpaca', mode='paper')
    executor.run(MyStrategy(), live_data_stream)

    # Step 3: Go live with the SAME code
    executor = LiveExecutor(broker='alpaca', mode='live')
    executor.run(MyStrategy(), live_data_stream)
```

## The Complete Workflow

Here's how VectorForge fits into a complete quantitative trading workflow:

```
┌──────────────────────────────────────────────────────────────────────────┐
│                      Complete Quant Workflow with FIREKit                 │
└──────────────────────────────────────────────────────────────────────────┘

Phase 1: Data Preparation
─────────────────────────
DataStream → Load and clean historical data
           → Handle corporate actions (splits, dividends)
           → Create point-in-time universe (no survivorship bias)

Phase 2: Research (Fast Iteration)                    ← VectorForge
──────────────────────────────────                      Vectorized Mode
VectorForge (Vectorized) → Test 1000s of parameters in seconds
                         → Explore factor combinations
                         → Find promising strategy candidates

Phase 3: Validation (Production Accuracy)             ← VectorForge
─────────────────────────────────────────               Event-Driven Mode
VectorForge (Event-Driven) → Validate with realistic execution
                           → Model slippage and commissions
                           → Walk-forward optimization

Phase 4: Risk Assessment
────────────────────────
RiskGuard → Size positions using Kelly Criterion
          → Set drawdown limits
          → Configure circuit breakers

Phase 5: Deployment
───────────────────
ExecutionCore → Paper trade first
              → Monitor for 2-4 weeks
              → Go live with confidence

Phase 6: Monitoring
───────────────────
PortfolioEngine → Track multi-strategy portfolio
                → Rebalance as needed
                → Generate performance reports
```

## Best Practices

### 1. Always Use the Hybrid Approach

```python
from vectorforge import HybridRunner

runner = HybridRunner()

# Fast: Test 1000 parameter combinations
params = runner.run_batch(strategy_class, param_grid, data)

# Accurate: Validate top 10 with realistic execution
for param in params[:10]:
    validated = runner.validate(strategy_class(**param), data)
```

### 2. Enable Bias Protection

```python
from vectorforge.data import DataGuard, PointInTimeUniverse

# Prevent lookahead bias
guarded_data = DataGuard(data, current_idx)

# Prevent survivorship bias
universe = PointInTimeUniverse.sp500(date='2020-01-01')
```

### 3. Use Walk-Forward Optimization

```python
from vectorforge.optimization import WalkForwardOptimizer

wfo = WalkForwardOptimizer(
    train_period=252,  # 1 year train
    test_period=63,    # 1 quarter test
    anchored=False     # Rolling window
)

results = wfo.run(strategy_class, param_grid, data)
print(f"Average degradation: {results.avg_degradation:.2%}")
```

### 4. Match Backtest to Production Execution

```python
# Use the SAME execution models in backtest and live
execution_config = {
    'slippage_model': 'volume_dependent',
    'slippage_bps': 5,
    'commission_model': 'ibkr_tiered',
}

# Backtest
backtester = EventDrivenBacktester(**execution_config)

# Live (same config)
executor = LiveExecutor(broker='ibkr', **execution_config)
```

## Summary

VectorForge is essential to FIREKit because it:

1. **Enables Fast Research**: 1000x speedup lets you iterate quickly
2. **Ensures Production Accuracy**: Event-driven mode mirrors live trading
3. **Prevents Costly Biases**: Built-in protection against lookahead and survivorship bias
4. **Integrates Seamlessly**: Works with DataStream, SignalML, RiskGuard, and ExecutionCore
5. **Supports Modern ML**: JAX/GPU acceleration for ML-based strategies

Without VectorForge, you can't validate strategies before risking real capital. It's the foundation that makes everything else in FIREKit possible.

## Next Steps

1. **Install VectorForge**: `pip install vectorforge`
2. **Run your first backtest**: See the [Quick Start Guide](../vectorforge/README.md)
3. **Connect DataStream**: Set up data ingestion
4. **Explore AlphaLab**: Research alpha factors
5. **Train SignalML models**: Build ML-powered strategies
6. **Deploy with ExecutionCore**: Go live with confidence

---

*For detailed API documentation, see [VectorForge Technical Specification](../products/01_vectorforge.md)*
