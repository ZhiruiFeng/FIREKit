# Data Model: VectorForge v0.3.0 Multi-Asset Portfolio Support

**Date**: 2026-01-07
**Status**: Complete

## Entity Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         PortfolioData                                │
│  (Container for multi-symbol price data with alignment)             │
├─────────────────────────────────────────────────────────────────────┤
│ • symbols: list[str]                                                 │
│ • dates: DatetimeIndex                                              │
│ • prices: ndarray[float32] (symbols × dates × OHLCV)                │
│ • tradeable_mask: ndarray[bool] (symbols × dates)                   │
│ • metadata: dict[str, SymbolMetadata]                               │
└───────────────────┬─────────────────────────────────────────────────┘
                    │ uses
                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      CorporateAction                                 │
│  (Stock splits, dividends, and adjustments)                         │
├─────────────────────────────────────────────────────────────────────┤
│ • symbol: str                                                        │
│ • action_type: "split" | "dividend"                                 │
│ • effective_date: date                                              │
│ • adjustment_factor: float                                          │
│ • cash_amount: float                                                │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                    CrossSectionalSignal                              │
│  (Generator for relative/ranked signals)                            │
├─────────────────────────────────────────────────────────────────────┤
│ • signal_type: str ("momentum", "mean_reversion", "custom")         │
│ • lookback: int                                                      │
│ • rank_method: "percentile" | "fractional" | "ordinal"              │
│ • group_field: str | None (for sector neutralization)              │
└───────────────────┬─────────────────────────────────────────────────┘
                    │ generates
                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       SignalResult                                   │
│  (Cross-sectional signal output)                                    │
├─────────────────────────────────────────────────────────────────────┤
│ • values: ndarray[float] (symbols × dates)                          │
│ • ranks: ndarray[float] (symbols × dates)                           │
│ • symbols: list[str]                                                │
│ • dates: DatetimeIndex                                              │
└───────────────────┬─────────────────────────────────────────────────┘
                    │ converted to
                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       TargetWeights                                  │
│  (Portfolio target allocation)                                      │
├─────────────────────────────────────────────────────────────────────┤
│ • weights: ndarray[float] (symbols × dates)                         │
│ • symbols: list[str]                                                │
│ • dates: DatetimeIndex                                              │
└───────────────────┬─────────────────────────────────────────────────┘
                    │ input to
                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        Rebalancer                                    │
│  (Determines when and how to rebalance)                             │
├─────────────────────────────────────────────────────────────────────┤
│ • trigger: RebalanceTrigger                                         │
│ • turnover_limit: float | None                                      │
│ • cost_aware: bool                                                  │
└───────────────────┬─────────────────────────────────────────────────┘
                    │ computes
                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      RebalanceOrders                                 │
│  (Trade instructions for a rebalance event)                         │
├─────────────────────────────────────────────────────────────────────┤
│ • date: date                                                        │
│ • trades: dict[str, float] (symbol → delta_weight)                  │
│ • turnover: float                                                   │
│ • constrained: bool                                                 │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                     PortfolioBacktestResult                          │
│  (Extended backtest result for portfolios)                          │
├─────────────────────────────────────────────────────────────────────┤
│ • (inherits from BacktestResult)                                    │
│ • weights_history: DataFrame (dates × symbols)                      │
│ • asset_returns: DataFrame (dates × symbols)                        │
│ • rebalance_dates: list[date]                                       │
│ • turnover_history: Series                                          │
│ • sector_exposure: DataFrame (dates × sectors)                      │
└───────────────────┬─────────────────────────────────────────────────┘
                    │ analyzed by
                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     PortfolioMetrics                                 │
│  (Extended metrics for portfolios)                                  │
├─────────────────────────────────────────────────────────────────────┤
│ • (inherits from PerformanceMetrics)                                │
│ • weights: DataFrame                                                │
│ • asset_returns: DataFrame                                          │
│ • sector_map: dict[str, str] | None                                 │
│ • benchmark_weights: DataFrame | None                               │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Entity Definitions

### 1. PortfolioData

**Purpose**: Container for aligned multi-symbol OHLCV data.

**Fields**:

| Field | Type | Description | Validation |
|-------|------|-------------|------------|
| `symbols` | `list[str]` | Ordered list of ticker symbols | Non-empty, unique |
| `dates` | `pd.DatetimeIndex` | Aligned trading dates | Sorted ascending, no dups |
| `prices` | `np.ndarray[float32]` | 3D array: (symbols, dates, 5) for OHLCV | No NaN after alignment |
| `tradeable_mask` | `np.ndarray[bool]` | 2D array: (symbols, dates) | True if asset traded |
| `metadata` | `dict[str, SymbolMetadata]` | Per-symbol info (sector, market cap) | Optional |

**State Transitions**:
```
Raw → Aligned → Validated → Ready
```

**Validation Rules**:
- V1: All symbols must have at least one valid price
- V2: Date range must have at least 20 trading days
- V3: Gap threshold: no gaps > configurable limit (default 5 days)
- V4: No future dates relative to "as of" date

---

### 2. SymbolMetadata

**Purpose**: Per-symbol metadata for grouping and weighting.

**Fields**:

| Field | Type | Description | Validation |
|-------|------|-------------|------------|
| `symbol` | `str` | Ticker symbol | Required |
| `sector` | `str \| None` | GICS sector or custom | Optional |
| `industry` | `str \| None` | GICS industry | Optional |
| `market_cap` | `float \| None` | Market capitalization | > 0 if present |
| `exchange` | `str \| None` | Exchange code (NYSE, NASDAQ) | Optional |

---

### 3. CorporateAction

**Purpose**: Represents stock splits and dividends.

**Fields**:

| Field | Type | Description | Validation |
|-------|------|-------------|------------|
| `symbol` | `str` | Affected symbol | Must exist in portfolio |
| `action_type` | `Literal["split", "dividend"]` | Action category | Required |
| `effective_date` | `date` | When action takes effect | Within data range |
| `adjustment_factor` | `float` | Split ratio (e.g., 2.0 for 2:1) | > 0 for splits |
| `cash_amount` | `float` | Dividend per share | >= 0 for dividends |
| `reinvest` | `bool` | Whether to reinvest dividends | Default: False |

**State Transitions**:
```
Pending → Applied
```

---

### 4. CrossSectionalSignal

**Purpose**: Generator for relative/ranked signals across universe.

**Fields**:

| Field | Type | Description | Validation |
|-------|------|-------------|------------|
| `signal_type` | `str` | Signal category | "momentum", "mean_reversion", "volatility", "custom" |
| `lookback` | `int` | Lookback period in days | > 0 |
| `rank_method` | `str` | Ranking method | "percentile", "fractional", "ordinal" |
| `group_field` | `str \| None` | Field for sector neutralization | Must exist in metadata |
| `ascending` | `bool` | Rank direction | Default: False (higher = better) |

**Factory Methods**:
- `momentum(lookback=252)` → Price momentum signal
- `mean_reversion(lookback=20)` → Short-term reversal signal
- `volatility(lookback=60)` → Inverse volatility signal

---

### 5. SignalResult

**Purpose**: Output of cross-sectional signal generation.

**Fields**:

| Field | Type | Description | Validation |
|-------|------|-------------|------------|
| `values` | `np.ndarray[float]` | Raw signal values (symbols × dates) | Required |
| `ranks` | `np.ndarray[float]` | Percentile ranks (symbols × dates) | 0-100 |
| `symbols` | `list[str]` | Symbol order | Matches values rows |
| `dates` | `pd.DatetimeIndex` | Date order | Matches values cols |

**Methods**:
- `top_percentile(n: int) → SignalResult` — Filter to top n%
- `bottom_percentile(n: int) → SignalResult` — Filter to bottom n%
- `between_percentile(low: int, high: int) → SignalResult` — Range filter

---

### 6. TargetWeights

**Purpose**: Portfolio target allocation at each rebalance date.

**Fields**:

| Field | Type | Description | Validation |
|-------|------|-------------|------------|
| `weights` | `np.ndarray[float]` | Target weights (symbols × dates) | Sum to 1.0 (or leverage) |
| `symbols` | `list[str]` | Symbol order | Required |
| `dates` | `pd.DatetimeIndex` | Rebalance dates | Required |

**Construction Methods**:
- `equal_weight()` → Equal weight across selected assets
- `market_cap_weight()` → Weight by market cap
- `inverse_volatility_weight()` → Risk parity weighting
- `custom_weight(weights_dict)` → User-defined weights

---

### 7. RebalanceTrigger (Abstract)

**Purpose**: Determines when to rebalance.

**Implementations**:

| Class | Trigger Condition |
|-------|-------------------|
| `CalendarTrigger` | First trading day of period (daily, weekly, monthly, quarterly, annual) |
| `DriftTrigger` | Any position weight deviates > threshold from target |
| `HybridTrigger` | Calendar OR drift (whichever occurs first) |

---

### 8. Rebalancer

**Purpose**: Computes trades to reach target weights.

**Fields**:

| Field | Type | Description | Validation |
|-------|------|-------------|------------|
| `trigger` | `RebalanceTrigger` | When to rebalance | Required |
| `turnover_limit` | `float \| None` | Max turnover per rebalance | 0-1 if set |
| `cost_aware` | `bool` | Optimize for transaction costs | Default: False |
| `min_trade_size` | `float` | Minimum trade as % of portfolio | Default: 0.001 |

---

### 9. RebalanceOrders

**Purpose**: Output of rebalancing calculation.

**Fields**:

| Field | Type | Description | Validation |
|-------|------|-------------|------------|
| `date` | `date` | Rebalance date | Required |
| `trades` | `dict[str, float]` | Symbol → weight change | Required |
| `from_weights` | `dict[str, float]` | Current weights | Required |
| `to_weights` | `dict[str, float]` | Target weights after trades | Required |
| `turnover` | `float` | Total turnover (sum of absolute deltas / 2) | 0-1+ |
| `constrained` | `bool` | True if turnover was limited | Required |

---

### 10. PortfolioBacktestResult

**Purpose**: Extended backtest result for multi-asset portfolios.

**Inherits**: `BacktestResult`

**Additional Fields**:

| Field | Type | Description |
|-------|------|-------------|
| `weights_history` | `pd.DataFrame` | Daily weights (dates × symbols) |
| `asset_returns` | `pd.DataFrame` | Per-asset returns (dates × symbols) |
| `asset_contributions` | `pd.DataFrame` | Return contribution (dates × symbols) |
| `rebalance_dates` | `list[date]` | When rebalances occurred |
| `turnover_history` | `pd.Series` | Turnover at each rebalance |
| `sector_exposure` | `pd.DataFrame` | Sector weights over time |
| `cash_balance` | `pd.Series` | Cash position over time |

---

### 11. PortfolioMetrics

**Purpose**: Portfolio-specific analytics.

**Inherits**: `PerformanceMetrics`

**Additional Fields**:

| Field | Type | Description |
|-------|------|-------------|
| `weights` | `pd.DataFrame` | Weight history |
| `asset_returns` | `pd.DataFrame` | Per-asset returns |
| `sector_map` | `dict[str, str]` | Symbol → sector mapping |
| `benchmark_weights` | `pd.DataFrame \| None` | Benchmark allocation |

**New Methods**:

| Method | Returns | Description |
|--------|---------|-------------|
| `diversification_ratio()` | `float` | σ_weighted / σ_portfolio |
| `herfindahl_index()` | `pd.Series` | Concentration over time |
| `sector_exposure()` | `pd.DataFrame` | Allocation by sector |
| `portfolio_beta(benchmark)` | `float` | Market sensitivity |
| `contribution_to_return()` | `pd.DataFrame` | Per-asset attribution |
| `correlation_matrix(window)` | `pd.DataFrame` | Rolling correlations |
| `top_n_concentration(n)` | `pd.Series` | Weight in top N holdings |

---

## Relationships

```
PortfolioData
    │
    ├──[1:N]──► SymbolMetadata
    │
    ├──[1:N]──► CorporateAction
    │
    └──[input]──► CrossSectionalSignal
                        │
                        └──[generates]──► SignalResult
                                              │
                                              └──[converts]──► TargetWeights
                                                                   │
                                                                   └──[input]──► Rebalancer
                                                                                     │
                                                                                     └──[outputs]──► RebalanceOrders
                                                                                                        │
                                                                                                        └──[executed in]──► PortfolioBacktestResult
                                                                                                                                │
                                                                                                                                └──[analyzed by]──► PortfolioMetrics
```

---

## Validation Summary

| Entity | Key Validations |
|--------|-----------------|
| PortfolioData | Non-empty symbols, sorted dates, no NaN after alignment |
| CorporateAction | Symbol exists, date in range, factor > 0 |
| CrossSectionalSignal | Lookback > 0, valid rank method |
| TargetWeights | Weights sum constraint (configurable) |
| Rebalancer | Turnover limit 0-1, valid trigger |
