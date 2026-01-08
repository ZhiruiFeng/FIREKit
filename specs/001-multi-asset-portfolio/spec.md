# Feature Specification: VectorForge v0.3.0 Multi-Asset Portfolio Support

**Feature Branch**: `001-multi-asset-portfolio`
**Created**: 2026-01-07
**Status**: Draft
**Input**: User description: "VectorForge v0.3.0 Multi-Asset Portfolio Support - Enable backtesting strategies across multiple assets simultaneously with proper portfolio-level metrics and rebalancing logic"

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Backtest a Multi-Asset Strategy (Priority: P1)

A quantitative researcher wants to backtest a momentum strategy across a universe of 50 stocks to evaluate portfolio-level performance, including how assets interact and contribute to overall returns.

**Why this priority**: This is the core use case for multi-asset support. Without the ability to run strategies across multiple assets and see aggregated results, the entire feature has no value. This enables the transition from single-asset to portfolio-based research.

**Independent Test**: Can be fully tested by loading price data for multiple symbols, running a simple equal-weight strategy, and verifying portfolio-level returns are calculated correctly.

**Acceptance Scenarios**:

1. **Given** price data for 50 symbols over 5 years, **When** I run a momentum-based portfolio strategy, **Then** I receive portfolio-level equity curve, returns, and performance metrics aggregated across all positions.

2. **Given** a multi-asset backtest has completed, **When** I request the results, **Then** I can see individual asset contributions to portfolio P&L alongside aggregate metrics.

3. **Given** symbols with different trading calendars (e.g., NYSE and LSE stocks), **When** I run a portfolio backtest, **Then** the system correctly aligns data using forward-fill for missing days.

---

### User Story 2 - Implement Cross-Sectional Signal Generation (Priority: P1)

A portfolio manager wants to generate relative signals by ranking assets against each other (e.g., top decile momentum, bottom quartile value) rather than using absolute thresholds.

**Why this priority**: Cross-sectional signals are fundamental to most institutional portfolio strategies. Without ranking and relative comparisons, users cannot implement common strategies like momentum, value, or quality factor portfolios.

**Independent Test**: Can be fully tested by providing historical returns for 20 assets, generating a momentum ranking signal, and verifying that the top 5 assets are correctly identified each period.

**Acceptance Scenarios**:

1. **Given** a universe of 100 stocks with 12-month return data, **When** I apply a momentum ranking signal, **Then** each stock receives a percentile rank (0-100) based on its relative performance.

2. **Given** a cross-sectional signal generator, **When** I request "top 10%" assets, **Then** I receive only the assets in the highest decile for that signal at each point in time.

3. **Given** stocks with sector classifications, **When** I generate sector-neutral signals, **Then** rankings are computed within each sector to neutralize sector effects.

---

### User Story 3 - Configure Portfolio Rebalancing (Priority: P1)

A fund manager needs to rebalance a portfolio on a monthly schedule while respecting turnover constraints to manage transaction costs.

**Why this priority**: Rebalancing logic determines when and how portfolios change. Without this, users cannot simulate realistic portfolio management where positions are adjusted periodically based on new signals.

**Independent Test**: Can be fully tested by setting up a monthly rebalancing schedule, running a backtest, and verifying that trades only execute at month boundaries.

**Acceptance Scenarios**:

1. **Given** a calendar-based rebalancer set to monthly, **When** the backtest runs, **Then** position adjustments occur only on the first trading day of each month.

2. **Given** a drift-threshold rebalancer set to 5%, **When** any position drifts more than 5% from target weight, **Then** a rebalance is triggered regardless of calendar.

3. **Given** a turnover constraint of 20% per rebalance, **When** target weights would require 35% turnover, **Then** the system partially rebalances to respect the constraint while prioritizing the largest deviations.

---

### User Story 4 - Analyze Portfolio-Level Metrics (Priority: P2)

A risk analyst wants to understand portfolio concentration, sector exposure, and diversification metrics beyond simple return statistics.

**Why this priority**: While basic P&L metrics are essential (covered in Story 1), deeper portfolio analytics help users understand risk characteristics. This is important but secondary to core functionality.

**Independent Test**: Can be fully tested by running a completed backtest and requesting concentration metrics (HHI, top-5 holdings weight) and verifying calculations match expected formulas.

**Acceptance Scenarios**:

1. **Given** a completed portfolio backtest, **When** I request diversification analysis, **Then** I receive diversification ratio, correlation matrix, and concentration metrics (HHI).

2. **Given** a portfolio with sector classifications, **When** I request sector exposure, **Then** I see time-series of allocation percentages by sector.

3. **Given** a multi-asset portfolio, **When** I request rolling correlation analysis, **Then** I receive pairwise correlation matrices computed over specified lookback windows.

---

### User Story 5 - Handle Corporate Actions (Priority: P2)

A data engineer needs to backtest strategies with accurate handling of stock splits and dividends to ensure historical P&L calculations are correct.

**Why this priority**: Corporate actions affect historical prices and positions. Without proper handling, backtests produce incorrect results. This is important for accuracy but many users work with already-adjusted data.

**Independent Test**: Can be fully tested by providing split-adjusted and unadjusted data for a stock that had a 2:1 split, running a backtest, and verifying position quantities and values are correctly adjusted.

**Acceptance Scenarios**:

1. **Given** a 2:1 stock split on a held position, **When** the split date passes, **Then** position quantity doubles and price basis is halved, maintaining total value.

2. **Given** a $1.00 cash dividend on a 100-share position, **When** the ex-dividend date passes, **Then** $100 cash is credited to the portfolio (if reinvestment is disabled).

3. **Given** the user enables dividend reinvestment, **When** a dividend is paid, **Then** additional fractional shares are purchased at the current price.

---

### Edge Cases

- What happens when an asset in the universe is delisted mid-backtest? The system handles the delisting by closing the position at the last available price and excluding the asset from future signal calculations.

- How does the system handle assets with different trading hours (e.g., US and Asian markets)? Data is aligned to a common time axis using forward-fill, with configurable handling for timezone differences.

- What happens when target weights sum to more than 100% (leverage) or less than 100% (cash position)? The system respects the provided weights, with leverage resulting in margin usage tracking and under-allocation maintaining a cash balance.

- How are missing prices handled for thinly-traded assets? Configurable options: forward-fill (default), linear interpolation, or flag as untradeable for that period.

## Requirements *(mandatory)*

### Functional Requirements

#### Portfolio Data Management
- **FR-001**: System MUST support loading and storing OHLCV data for multiple symbols simultaneously in a single data structure
- **FR-002**: System MUST align data from different trading calendars to a common timeline, with forward-fill as the default for missing values
- **FR-003**: System MUST provide configurable missing data handling: forward-fill, linear interpolation, or mark-as-invalid
- **FR-004**: System MUST apply stock split adjustments to historical prices and position quantities
- **FR-005**: System MUST handle cash dividends with options for cash payout or reinvestment
- **FR-006**: System MUST validate data consistency across symbols (no future dates, no gaps exceeding configurable threshold)

#### Cross-Sectional Signals
- **FR-007**: System MUST support ranking assets by any numeric signal to produce percentile scores (0-100)
- **FR-008**: System MUST support filtering assets by rank threshold (e.g., top 10%, bottom quartile)
- **FR-009**: System MUST support sector/group-neutral signal generation where rankings are computed within groups
- **FR-010**: System MUST support market-cap weighting for constructing capitalization-weighted portfolios
- **FR-011**: System MUST support relative strength calculations comparing asset momentum to universe average

#### Rebalancing Logic
- **FR-012**: System MUST support calendar-based rebalancing (daily, weekly, monthly, quarterly, annual)
- **FR-013**: System MUST support drift-based rebalancing triggered when any position deviates beyond a threshold from target weight
- **FR-014**: System MUST support turnover constraints that limit the percentage of portfolio traded per rebalance
- **FR-015**: System MUST optimize trade execution to minimize transaction costs when turnover-constrained
- **FR-016**: System MUST support combining calendar and drift triggers (e.g., monthly rebalance with 10% drift override)

#### Portfolio Metrics
- **FR-017**: System MUST calculate portfolio-level Sharpe and Sortino ratios from aggregate returns
- **FR-018**: System MUST track rolling correlation matrices across portfolio holdings
- **FR-019**: System MUST calculate sector/group exposure as time-series of allocation percentages
- **FR-020**: System MUST calculate diversification ratio and portfolio-level beta
- **FR-021**: System MUST calculate concentration metrics: HHI (Herfindahl-Hirschman Index) and top-N holding percentages
- **FR-022**: System MUST provide per-asset contribution to portfolio return and risk

### Key Entities

- **PortfolioData**: Container for multi-symbol OHLCV data with alignment, validation, and corporate action support. Key attributes: symbol list, aligned price matrix, volume matrix, adjustment factors.

- **CrossSectionalSignal**: Generator for relative/ranked signals across a universe. Key attributes: signal type, lookback period, grouping field (for sector-neutralization).

- **Rebalancer**: Logic for determining when and how to rebalance. Key attributes: trigger type (calendar/drift/hybrid), frequency, turnover limit, cost optimization flag.

- **PortfolioMetrics**: Calculator for aggregate and decomposed portfolio analytics. Key attributes: returns aggregation method, benchmark (optional), risk-free rate.

- **CorporateAction**: Representation of splits, dividends, and other adjustments. Key attributes: action type, effective date, adjustment factor, cash amount.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Users can backtest a 100-asset portfolio over 10 years of daily data in under 5 seconds (vectorized mode)
- **SC-002**: Portfolio-level metrics (Sharpe, drawdown, sector exposure) are computed within 1 second of backtest completion
- **SC-003**: Cross-sectional signal generation for 500 assets completes in under 500 milliseconds per period
- **SC-004**: System handles portfolios with up to 1,000 assets without performance degradation greater than linear scaling
- **SC-005**: Rebalancing logic correctly respects constraints (verified: target trades match constraint limits within 0.1% tolerance)
- **SC-006**: Corporate action adjustments produce identical P&L results compared to pre-adjusted data (within floating-point tolerance)
- **SC-007**: 90% of quantitative researchers can configure and run a multi-asset backtest using only documentation (usability target)
- **SC-008**: Memory usage for 1,000 assets over 10 years remains under 4GB

## Assumptions

- Users will provide data in standard OHLCV format; the system is not responsible for data sourcing or cleaning beyond basic alignment
- Sector/industry classifications are provided by the user as metadata; the system does not infer classifications
- Transaction costs (slippage, commission) use existing VectorForge v0.2.0 models; no new cost model types are required
- The existing event-driven and vectorized engine architecture will be extended, not replaced
- Cash is always available for purchases (no margin call simulation in this version)
- All assets are assumed to be in the same currency; multi-currency support is out of scope for v0.3.0

## Out of Scope

- Multi-currency support and FX hedging
- Options, futures, and other derivatives (equity-like assets only)
- Real-time/live trading integration (backtest only)
- Visual portfolio dashboard (v1.0.0 scope)
- Tax-lot optimization (deferred to future version based on P2 priority)
- Distributed backtesting across multiple machines
