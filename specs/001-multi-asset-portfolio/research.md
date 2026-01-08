# Research: VectorForge v0.3.0 Multi-Asset Portfolio Support

**Date**: 2026-01-07
**Status**: Complete

## Overview

This document captures design decisions, best practices research, and integration patterns for implementing multi-asset portfolio support in VectorForge.

---

## 1. Portfolio Data Container Design

### Decision: Panel-like Structure with Symbol × Time Axes

**Rationale**: Multi-asset OHLCV data requires efficient storage and access patterns. A 3D structure (symbol × time × field) enables vectorized cross-sectional operations.

**Alternatives Considered**:

| Option | Description | Rejected Because |
|--------|-------------|------------------|
| Dict of DataFrames | `{symbol: df}` mapping | O(n) iteration for cross-sectional ops; no alignment guarantee |
| MultiIndex DataFrame | Pandas hierarchical index | Complex API; memory overhead for sparse data |
| xarray DataArray | N-dimensional labeled arrays | Adds dependency; overkill for OHLCV structure |

**Chosen Approach**:
- Primary: NumPy 3D array `(n_symbols, n_times, n_fields)` for computation
- Metadata: Separate symbol list and DatetimeIndex
- Polars for I/O and lazy alignment operations
- Memory-mapped backing for datasets > 100k rows (existing pattern from v0.2.0)

---

## 2. Data Alignment Strategy

### Decision: Forward-Fill with Configurable Fallbacks

**Rationale**: Assets trade on different calendars (NYSE vs LSE, crypto 24/7). Alignment to a common timeline is required for cross-sectional comparisons.

**Best Practices Researched**:
- **Forward-fill (LOCF)**: Industry standard for price data; maintains last known state
- **Linear interpolation**: Inappropriate for financial prices; creates artificial data points
- **Mark-as-NaN**: Valid for excluding non-tradeable periods from signal calculation

**Chosen Approach**:
```python
class MissingDataPolicy(Enum):
    FORWARD_FILL = "ffill"      # Default: carry last known value
    INTERPOLATE = "interpolate"  # Linear interpolation (volume only)
    DROP = "drop"                # Exclude from universe for that period
    ZERO = "zero"                # Fill with 0 (for returns/signals)
```

**Calendar Handling**:
- Union of all symbol trading days forms master timeline
- Each symbol carries a tradeable mask: `bool[n_times]`
- Cross-sectional signals skip non-tradeable assets in rankings

---

## 3. Cross-Sectional Signal Generation

### Decision: Percentile Ranking with Group Neutralization

**Rationale**: Cross-sectional signals rank assets relative to peers rather than absolute thresholds. This is standard in factor investing.

**Best Practices Researched**:
- **Percentile ranking (0-100)**: Robust to outliers; uniform distribution
- **Z-score normalization**: Sensitive to outliers; assumes normal distribution
- **Fractional ranking**: Raw ranks divided by count; equivalent to percentile

**Chosen Approach**:
```python
def rank_signal(values: np.ndarray, method: str = "percentile") -> np.ndarray:
    """
    Rank values cross-sectionally.

    Args:
        values: (n_symbols,) array of raw signal values
        method: "percentile" (0-100), "fractional" (0-1), "ordinal" (1-N)

    Returns:
        Ranked signal values
    """
```

**Sector Neutralization**:
- Optional grouping field (e.g., GICS sector)
- Ranking computed within each group
- Prevents sector bets from dominating signal

---

## 4. Rebalancing Architecture

### Decision: Composable Trigger System

**Rationale**: Different strategies require calendar-based, drift-based, or hybrid rebalancing. A strategy pattern enables clean composition.

**Alternatives Considered**:

| Option | Description | Rejected Because |
|--------|-------------|------------------|
| Single Rebalancer class | Monolithic with mode flags | Poor extensibility; complex conditionals |
| Inheritance hierarchy | CalendarRebalancer extends BaseRebalancer | Rigid; doesn't support hybrid triggers |

**Chosen Approach**:
```python
class RebalanceTrigger(ABC):
    """Abstract trigger that determines when to rebalance."""
    @abstractmethod
    def should_rebalance(self, date: date, portfolio: Portfolio) -> bool: ...

class CalendarTrigger(RebalanceTrigger):
    """Rebalance on schedule: daily, weekly, monthly, etc."""

class DriftTrigger(RebalanceTrigger):
    """Rebalance when position drifts beyond threshold."""

class HybridTrigger(RebalanceTrigger):
    """Combine multiple triggers with OR logic."""
```

**Turnover Optimization**:
- When constrained, prioritize trades by deviation magnitude
- Use greedy allocation: largest deviations first until turnover budget exhausted
- Track actual vs target weights for next period's drift calculation

---

## 5. Portfolio Metrics Extension

### Decision: Extend Existing PerformanceMetrics Class

**Rationale**: Avoid code duplication. Portfolio metrics build on single-asset metrics with additional aggregate calculations.

**New Metrics Required**:

| Metric | Formula | Purpose |
|--------|---------|---------|
| Diversification Ratio | σ_weighted / σ_portfolio | Measures correlation benefit |
| HHI (Concentration) | Σ(weight²) | 0 = uniform, 1 = single asset |
| Sector Exposure | Σ(weight_i) per sector | Allocation breakdown |
| Portfolio Beta | Cov(r_p, r_m) / Var(r_m) | Market sensitivity |
| Contribution to Return | weight_i × return_i | Per-asset attribution |

**Implementation**:
```python
class PortfolioMetrics(PerformanceMetrics):
    """Extends PerformanceMetrics with portfolio-specific analytics."""

    def __init__(self, returns: pd.Series, weights: pd.DataFrame, ...):
        super().__init__(returns, ...)
        self.weights = weights  # (n_times, n_symbols)
```

---

## 6. Corporate Actions Handling

### Decision: Adjustment Factor Approach

**Rationale**: Stock splits and dividends affect historical prices and positions. Using adjustment factors is cleaner than modifying raw data.

**Best Practices Researched**:
- **Split adjustment**: Multiply historical prices by factor; divide position quantities
- **Dividend adjustment**: Subtract dividend from price (for total return calc)
- **Point-in-time**: Apply adjustments as of event date, not retroactively

**Chosen Approach**:
```python
@dataclass
class CorporateAction:
    symbol: str
    action_type: Literal["split", "dividend"]
    effective_date: date
    adjustment_factor: float  # For splits: new_shares / old_shares
    cash_amount: float        # For dividends: per-share amount
```

**Dividend Reinvestment**:
- Optional DRIP (Dividend Reinvestment Plan) setting
- If enabled: cash dividend buys fractional shares at close price
- If disabled: cash credited to portfolio

---

## 7. Engine Integration Pattern

### Decision: Parallel Methods, Not Parallel Classes

**Rationale**: Existing engines (Vectorized, EventDriven, Hybrid) should gain portfolio methods rather than creating separate PortfolioVectorizedEngine classes.

**Integration Points**:

```python
class VectorizedEngine:
    def run(self, strategy, data, ...):  # Existing single-asset
        ...

    def run_portfolio(self, strategy, portfolio_data, ...):  # NEW
        """Run multi-asset portfolio backtest."""
        ...

class PortfolioStrategy(BaseStrategy):
    """Strategy that generates portfolio weights instead of signals."""

    def generate_weights(
        self,
        portfolio_data: PortfolioData,
        current_weights: np.ndarray,
    ) -> np.ndarray:
        """Generate target weights for next period."""
        ...
```

---

## 8. Performance Optimization Strategies

### Decision: Vectorized Cross-Sectional Operations

**Rationale**: Cross-sectional operations (ranking, weighting) must be vectorized to meet performance targets.

**Optimization Techniques**:

| Operation | Technique | Expected Speedup |
|-----------|-----------|------------------|
| Percentile ranking | `scipy.stats.rankdata` with broadcasting | 100x vs loop |
| Weight normalization | NumPy vectorized division | 1000x vs loop |
| Rebalance trades | NumPy diff + mask | 50x vs loop |
| Correlation matrix | Polars lazy with streaming | Memory-efficient |

**Memory Optimization**:
- Use `float32` for price data (sufficient precision, half memory)
- Memory-map for >100k rows (existing v0.2.0 pattern)
- Streaming correlation for large universes (chunk processing)

---

## 9. API Design Principles

### Decision: Consistent with Existing VectorForge Patterns

**Guiding Principles**:
1. **Explicit over implicit**: No hidden state; all inputs/outputs documented
2. **Chainable methods**: Return self or new objects for fluent API
3. **Type-safe**: Full type hints; Pydantic for config validation
4. **Fail fast**: Validate inputs early; clear error messages

**Example API**:
```python
# Load and align portfolio data
portfolio = (
    PortfolioData.from_dict({"AAPL": aapl_df, "GOOG": goog_df})
    .align(method="ffill")
    .validate()
)

# Generate cross-sectional signals
momentum = CrossSectionalSignal.momentum(lookback=252)
weights = momentum.generate(portfolio).top_percentile(10).equal_weight()

# Run backtest with rebalancing
result = (
    VectorizedEngine()
    .run_portfolio(
        strategy=weights,
        data=portfolio,
        rebalancer=CalendarRebalancer("monthly", turnover_limit=0.2),
    )
)
```

---

## Summary of Key Decisions

| Area | Decision | Key Benefit |
|------|----------|-------------|
| Data structure | NumPy 3D array + metadata | Fast cross-sectional ops |
| Alignment | Forward-fill default | Industry standard |
| Signals | Percentile ranking | Robust to outliers |
| Rebalancing | Composable triggers | Flexible strategies |
| Metrics | Extend existing class | Code reuse |
| Corporate actions | Adjustment factors | Clean data handling |
| Engine integration | Add methods, not classes | Backward compatible |
| Performance | Vectorized everywhere | Meet performance targets |
