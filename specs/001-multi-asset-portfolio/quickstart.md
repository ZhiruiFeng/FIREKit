# Quickstart: VectorForge v0.3.0 Multi-Asset Portfolio Support

**Date**: 2026-01-07

This guide demonstrates the key features of VectorForge v0.3.0 multi-asset portfolio support.

## 1. Loading Multi-Asset Data

```python
import pandas as pd
from vectorforge.portfolio import PortfolioData, MissingDataPolicy

# Load individual symbol DataFrames (OHLCV format)
aapl = pd.read_parquet("data/AAPL.parquet")
goog = pd.read_parquet("data/GOOG.parquet")
msft = pd.read_parquet("data/MSFT.parquet")
# ... load more symbols

# Create PortfolioData from dictionary
portfolio_data = PortfolioData.from_dict({
    "AAPL": aapl,
    "GOOG": goog,
    "MSFT": msft,
    # ... add all symbols
})

# Align to common dates with forward-fill for missing values
aligned = portfolio_data.align(
    policy=MissingDataPolicy.FORWARD_FILL,
    start_date="2020-01-01",
    end_date="2025-12-31",
)

# Validate data consistency
validated = aligned.validate(min_dates=252, max_gap_days=5)

print(f"Universe: {validated.n_symbols} symbols × {validated.n_dates} dates")
```

## 2. Generating Cross-Sectional Signals

```python
from vectorforge.portfolio import CrossSectionalSignal

# Create momentum signal (12-month lookback, skip last month)
momentum = CrossSectionalSignal.momentum(lookback=252, skip_recent=21)

# Generate signals across universe
signals = momentum.generate(validated)

# Filter to top 10% (decile) stocks
top_decile = signals.top_percentile(10)

# Convert to equal-weighted portfolio
weights = top_decile.to_weights(method="equal")

print(f"Selected {len(weights.get_weights_at(weights.dates[0]))} assets")
```

## 3. Sector-Neutral Signal Generation

```python
from vectorforge.portfolio import SymbolMetadata

# Add sector metadata
validated = validated.set_metadata("AAPL", SymbolMetadata(
    symbol="AAPL",
    sector="Technology",
    market_cap=3_000_000_000_000,
))
validated = validated.set_metadata("GOOG", SymbolMetadata(
    symbol="GOOG",
    sector="Technology",
    market_cap=2_000_000_000_000,
))
# ... add metadata for all symbols

# Generate sector-neutral signals (rank within each sector)
sector_neutral_signals = momentum.generate(
    validated,
    group_field="sector",  # Rank within sectors
)

# Top 20% within each sector
sector_neutral_weights = (
    sector_neutral_signals
    .top_percentile(20)
    .to_weights(method="equal")
)
```

## 4. Configuring Portfolio Rebalancing

```python
from vectorforge.portfolio import Rebalancer, RebalanceFrequency

# Monthly rebalancing with 20% turnover limit
rebalancer = Rebalancer.calendar(
    frequency=RebalanceFrequency.MONTHLY,
    turnover_limit=0.20,  # Max 20% portfolio turnover per rebalance
)

# Or use drift-based rebalancing
drift_rebalancer = Rebalancer.drift(
    threshold=0.05,  # Rebalance when any position drifts 5%+
    turnover_limit=0.15,
)

# Or hybrid: monthly + 10% drift override
hybrid_rebalancer = Rebalancer.hybrid(
    calendar_frequency=RebalanceFrequency.MONTHLY,
    drift_threshold=0.10,
    turnover_limit=0.25,
)
```

## 5. Running a Portfolio Backtest

```python
from vectorforge import VectorizedEngine

# Create engine
engine = VectorizedEngine()

# Run portfolio backtest
result = engine.run_portfolio(
    strategy=weights,  # Target weights from signal generation
    data=validated,
    rebalancer=rebalancer,
    initial_capital=1_000_000,
)

# View results
print(result)
print(f"Total Return: {result.total_return:.2%}")
print(f"Sharpe Ratio: {result.sharpe_ratio:.2f}")
print(f"Max Drawdown: {result.max_drawdown:.2%}")
```

## 6. Analyzing Portfolio Metrics

```python
from vectorforge.portfolio import PortfolioMetrics

# Create metrics analyzer
metrics = PortfolioMetrics.from_backtest_result(
    result,
    sector_map={"AAPL": "Technology", "GOOG": "Technology", ...},
    risk_free_rate=0.04,
)

# Concentration metrics
concentration = metrics.concentration_metrics()
print(f"HHI: {concentration.herfindahl_index:.4f}")
print(f"Effective N: {concentration.effective_n:.1f}")
print(f"Top 5 Weight: {concentration.top_5_weight:.2%}")

# Diversification metrics
diversification = metrics.diversification_metrics()
print(f"Diversification Ratio: {diversification.diversification_ratio:.2f}")
print(f"Avg Correlation: {diversification.avg_correlation:.2f}")

# Sector exposure over time
sector_exp = metrics.sector_exposure()
print(sector_exp.tail())

# Return attribution by asset
attribution = metrics.return_attribution()
for attr in attribution[:5]:  # Top 5 contributors
    print(f"{attr.symbol}: {attr.contribution:.2%} contribution")
```

## 7. Handling Corporate Actions

```python
from vectorforge.portfolio import CorporateAction
from datetime import date

# Define corporate actions
actions = [
    CorporateAction(
        symbol="AAPL",
        action_type="split",
        effective_date=date(2022, 6, 6),
        adjustment_factor=4.0,  # 4-for-1 split
    ),
    CorporateAction(
        symbol="AAPL",
        action_type="dividend",
        effective_date=date(2023, 2, 10),
        cash_amount=0.23,
        reinvest=True,  # Reinvest dividends
    ),
]

# Apply to data
adjusted = validated.apply_corporate_actions(actions)
```

## 8. Complete Example: Momentum Strategy

```python
from vectorforge import VectorizedEngine
from vectorforge.portfolio import (
    PortfolioData,
    CrossSectionalSignal,
    Rebalancer,
    RebalanceFrequency,
    PortfolioMetrics,
    MissingDataPolicy,
)

# 1. Load and prepare data
data = PortfolioData.from_parquet("sp500_universe.parquet")
aligned = data.align(policy=MissingDataPolicy.FORWARD_FILL).validate()

# 2. Generate momentum signals
momentum = CrossSectionalSignal.momentum(lookback=252, skip_recent=21)
signals = momentum.generate(aligned)

# 3. Create portfolio: top 10% momentum, equal weighted
weights = signals.top_percentile(10).to_weights(method="equal")

# 4. Configure rebalancing: monthly, 25% turnover limit
rebalancer = Rebalancer.calendar(
    frequency=RebalanceFrequency.MONTHLY,
    turnover_limit=0.25,
)

# 5. Run backtest
engine = VectorizedEngine()
result = engine.run_portfolio(
    strategy=weights,
    data=aligned,
    rebalancer=rebalancer,
    initial_capital=1_000_000,
)

# 6. Analyze results
metrics = PortfolioMetrics.from_backtest_result(result)
report = metrics.generate_report()

print("=== Portfolio Backtest Results ===")
for key, value in report.items():
    print(f"{key}: {value}")
```

## Key Concepts

| Concept | Description |
|---------|-------------|
| **PortfolioData** | Container for multi-symbol OHLCV with alignment |
| **CrossSectionalSignal** | Rank-based signal across universe |
| **TargetWeights** | Portfolio allocation targets |
| **Rebalancer** | When/how to adjust positions |
| **PortfolioMetrics** | Analytics: HHI, diversification, attribution |

## Performance Tips

1. **Use vectorized mode** for research (fastest)
2. **Memory**: Use `float32` for large universes
3. **Batch operations**: Generate all signals before filtering
4. **Caching**: Reuse aligned PortfolioData across strategies
