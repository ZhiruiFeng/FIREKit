# PortfolioEngine: Asset Allocation & Portfolio Optimization

## Product Overview

**PortfolioEngine** is the portfolio management component of the FIREKit ecosystem, providing asset allocation strategies, rebalancing algorithms, and multi-strategy portfolio construction. It implements mean-variance optimization, risk parity, and dynamic rebalancing.

### Key Value Propositions

- **Multiple Allocation Methods**: Mean-variance, risk parity, equal weight, Black-Litterman
- **Dynamic Rebalancing**: Calendar, threshold, and cost-aware rebalancing strategies
- **Multi-Strategy Support**: Combine multiple alpha strategies with risk budgeting
- **Tax Efficiency**: Tax-loss harvesting and lot selection optimization
- **Performance Attribution**: Understand sources of returns and risk

## Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        PortfolioEngine                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                    Allocation Methods                        │    │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐     │    │
│  │  │   Mean   │  │   Risk   │  │  Black   │  │Hierarchical│    │    │
│  │  │ Variance │  │  Parity  │  │Litterman │  │   Risk    │    │    │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────┘     │    │
│  └───────────────────────────┬─────────────────────────────────┘    │
│                              │                                       │
│  ┌───────────────────────────▼─────────────────────────────────┐    │
│  │                   Rebalancing Engine                         │    │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐     │    │
│  │  │ Calendar │  │Threshold │  │  Cost    │  │ Optimal  │     │    │
│  │  │  Based   │  │  Based   │  │  Aware   │  │  Trade   │     │    │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────┘     │    │
│  └───────────────────────────┬─────────────────────────────────┘    │
│                              │                                       │
│  ┌───────────────────────────▼─────────────────────────────────┐    │
│  │                Multi-Strategy Manager                        │    │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐                   │    │
│  │  │  Risk    │  │ Strategy │  │ Correlation│                  │    │
│  │  │ Budgeting│  │ Blending │  │  Monitor  │                  │    │
│  │  └──────────┘  └──────────┘  └──────────┘                   │    │
│  └───────────────────────────┬─────────────────────────────────┘    │
│                              │                                       │
│  ┌───────────────────────────▼─────────────────────────────────┐    │
│  │                   Analytics Layer                            │    │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐                   │    │
│  │  │Performance│  │   Risk   │  │   Tax    │                   │    │
│  │  │Attribution│  │ Analytics│  │ Reporting│                   │    │
│  │  └──────────┘  └──────────┘  └──────────┘                   │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Allocation Methods Comparison

| Method | Best For | Complexity | Inputs Required |
|--------|----------|------------|-----------------|
| Mean-Variance | Traditional optimization | Medium | Returns, Covariance |
| Risk Parity | Equal risk contribution | Low | Covariance |
| Black-Litterman | Incorporating views | High | Returns, Covariance, Views |
| HRP | Robust allocation | Medium | Covariance |
| Equal Weight | Simplicity | Low | None |

## Technical Specification

### Allocation Algorithms

```python
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class AllocationResult:
    """Result of portfolio allocation."""
    weights: Dict[str, float]
    expected_return: float
    expected_volatility: float
    sharpe_ratio: float
    method: str


class MeanVarianceOptimizer:
    """
    Classic Markowitz mean-variance optimization.

    Maximize: w'μ - (λ/2) * w'Σw
    Subject to: sum(w) = 1, w >= 0 (optional)
    """

    def __init__(
        self,
        risk_aversion: float = 1.0,
        allow_short: bool = False,
        max_weight: float = 0.20
    ):
        self.risk_aversion = risk_aversion
        self.allow_short = allow_short
        self.max_weight = max_weight

    def optimize(
        self,
        expected_returns: pd.Series,
        covariance_matrix: pd.DataFrame,
        target_return: Optional[float] = None,
        target_volatility: Optional[float] = None
    ) -> AllocationResult:
        """
        Find optimal portfolio weights.

        Args:
            expected_returns: Expected returns for each asset
            covariance_matrix: Covariance matrix of returns
            target_return: Optional target return constraint
            target_volatility: Optional target volatility constraint
        """
        n_assets = len(expected_returns)
        symbols = expected_returns.index.tolist()

        mu = expected_returns.values
        sigma = covariance_matrix.values

        # Objective: maximize Sharpe-like utility
        def objective(w):
            ret = np.dot(w, mu)
            vol = np.sqrt(np.dot(w, np.dot(sigma, w)))
            return -(ret - self.risk_aversion * vol)  # Minimize negative

        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}  # Weights sum to 1
        ]

        if target_return is not None:
            constraints.append({
                'type': 'eq',
                'fun': lambda w: np.dot(w, mu) - target_return
            })

        if target_volatility is not None:
            constraints.append({
                'type': 'eq',
                'fun': lambda w: np.sqrt(np.dot(w, np.dot(sigma, w))) - target_volatility
            })

        # Bounds
        if self.allow_short:
            bounds = [(-self.max_weight, self.max_weight) for _ in range(n_assets)]
        else:
            bounds = [(0, self.max_weight) for _ in range(n_assets)]

        # Initial guess
        w0 = np.ones(n_assets) / n_assets

        # Optimize
        result = minimize(
            objective,
            w0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )

        weights = result.x
        exp_ret = np.dot(weights, mu)
        exp_vol = np.sqrt(np.dot(weights, np.dot(sigma, weights)))

        return AllocationResult(
            weights={s: w for s, w in zip(symbols, weights)},
            expected_return=exp_ret,
            expected_volatility=exp_vol,
            sharpe_ratio=exp_ret / exp_vol if exp_vol > 0 else 0,
            method='mean_variance'
        )

    def efficient_frontier(
        self,
        expected_returns: pd.Series,
        covariance_matrix: pd.DataFrame,
        n_points: int = 50
    ) -> List[AllocationResult]:
        """Generate points on the efficient frontier."""
        min_ret = expected_returns.min()
        max_ret = expected_returns.max()

        frontier = []
        for target_ret in np.linspace(min_ret, max_ret, n_points):
            try:
                result = self.optimize(
                    expected_returns,
                    covariance_matrix,
                    target_return=target_ret
                )
                frontier.append(result)
            except:
                continue

        return frontier


class RiskParity:
    """
    Risk parity allocation - equal risk contribution from each asset.

    Each asset contributes equally to portfolio volatility.
    """

    def __init__(self, target_volatility: Optional[float] = None):
        self.target_volatility = target_volatility

    def optimize(
        self,
        covariance_matrix: pd.DataFrame
    ) -> AllocationResult:
        """Find risk parity weights."""
        n_assets = len(covariance_matrix)
        symbols = covariance_matrix.index.tolist()
        sigma = covariance_matrix.values

        def risk_contribution(w):
            """Calculate each asset's contribution to portfolio risk."""
            portfolio_vol = np.sqrt(np.dot(w, np.dot(sigma, w)))
            marginal_contrib = np.dot(sigma, w) / portfolio_vol
            risk_contrib = w * marginal_contrib
            return risk_contrib

        def objective(w):
            """Minimize squared differences in risk contributions."""
            rc = risk_contribution(w)
            target_rc = np.ones(n_assets) / n_assets
            return np.sum((rc - target_rc) ** 2)

        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        ]

        # Bounds (long only)
        bounds = [(0.01, 1.0) for _ in range(n_assets)]

        # Initial guess
        w0 = np.ones(n_assets) / n_assets

        result = minimize(
            objective,
            w0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )

        weights = result.x
        portfolio_vol = np.sqrt(np.dot(weights, np.dot(sigma, weights)))

        # Scale to target volatility if specified
        if self.target_volatility:
            scale = self.target_volatility / portfolio_vol
            weights = weights * scale
            portfolio_vol = self.target_volatility

        return AllocationResult(
            weights={s: w for s, w in zip(symbols, weights)},
            expected_return=0,  # Risk parity doesn't use expected returns
            expected_volatility=portfolio_vol,
            sharpe_ratio=0,
            method='risk_parity'
        )


class HierarchicalRiskParity:
    """
    Hierarchical Risk Parity (HRP) by López de Prado.

    Uses hierarchical clustering to build a more robust allocation.
    """

    def optimize(
        self,
        returns: pd.DataFrame
    ) -> AllocationResult:
        """Calculate HRP weights."""
        from scipy.cluster.hierarchy import linkage, leaves_list
        from scipy.spatial.distance import squareform

        # Correlation and covariance
        corr = returns.corr()
        cov = returns.cov()
        symbols = returns.columns.tolist()

        # Distance matrix
        dist = np.sqrt((1 - corr) / 2)
        dist_condensed = squareform(dist.values)

        # Hierarchical clustering
        link = linkage(dist_condensed, method='single')
        sort_idx = leaves_list(link)

        # Recursive bisection for weights
        weights = self._recursive_bisection(cov.values, sort_idx)

        portfolio_vol = np.sqrt(np.dot(weights, np.dot(cov.values, weights)))

        return AllocationResult(
            weights={s: w for s, w in zip(symbols, weights)},
            expected_return=0,
            expected_volatility=portfolio_vol,
            sharpe_ratio=0,
            method='hrp'
        )

    def _recursive_bisection(
        self,
        cov: np.ndarray,
        sort_idx: np.ndarray
    ) -> np.ndarray:
        """Recursive bisection for HRP weights."""
        weights = np.ones(len(sort_idx))

        clusters = [sort_idx]

        while clusters:
            clusters_new = []
            for cluster in clusters:
                if len(cluster) > 1:
                    mid = len(cluster) // 2
                    left = cluster[:mid]
                    right = cluster[mid:]

                    # Calculate variance of each cluster
                    left_var = self._cluster_var(cov, left)
                    right_var = self._cluster_var(cov, right)

                    # Allocate inversely proportional to variance
                    alpha = 1 - left_var / (left_var + right_var)

                    weights[left] *= alpha
                    weights[right] *= (1 - alpha)

                    clusters_new.extend([left, right])

            clusters = clusters_new

        return weights

    def _cluster_var(self, cov: np.ndarray, indices: np.ndarray) -> float:
        """Calculate variance of a cluster."""
        sub_cov = cov[np.ix_(indices, indices)]
        ivp = 1 / np.diag(sub_cov)
        ivp /= ivp.sum()
        return np.dot(ivp, np.dot(sub_cov, ivp))


class BlackLitterman:
    """
    Black-Litterman model for incorporating views into allocation.
    """

    def __init__(
        self,
        tau: float = 0.05,  # Uncertainty of prior
        risk_aversion: float = 2.5
    ):
        self.tau = tau
        self.risk_aversion = risk_aversion

    def optimize(
        self,
        market_weights: pd.Series,
        covariance_matrix: pd.DataFrame,
        views: Dict[str, float],
        view_confidence: Dict[str, float]
    ) -> AllocationResult:
        """
        Calculate Black-Litterman posterior weights.

        Args:
            market_weights: Market cap weights (equilibrium)
            covariance_matrix: Covariance matrix
            views: Dict of asset -> expected return view
            view_confidence: Dict of asset -> confidence (0-1)
        """
        symbols = market_weights.index.tolist()
        w_mkt = market_weights.values
        sigma = covariance_matrix.values
        n = len(symbols)

        # Equilibrium returns (reverse optimization)
        pi = self.risk_aversion * np.dot(sigma, w_mkt)

        # View matrix P and view vector Q
        n_views = len(views)
        P = np.zeros((n_views, n))
        Q = np.zeros(n_views)
        omega = np.zeros((n_views, n_views))

        for i, (asset, view_return) in enumerate(views.items()):
            asset_idx = symbols.index(asset)
            P[i, asset_idx] = 1
            Q[i] = view_return
            confidence = view_confidence.get(asset, 0.5)
            omega[i, i] = (1 - confidence) * sigma[asset_idx, asset_idx]

        # Black-Litterman formula
        tau_sigma = self.tau * sigma
        M = np.linalg.inv(np.linalg.inv(tau_sigma) + np.dot(P.T, np.dot(np.linalg.inv(omega), P)))
        posterior_mean = np.dot(M, np.dot(np.linalg.inv(tau_sigma), pi) +
                                np.dot(P.T, np.dot(np.linalg.inv(omega), Q)))

        # Optimize with posterior
        mv = MeanVarianceOptimizer(risk_aversion=self.risk_aversion)
        result = mv.optimize(
            pd.Series(posterior_mean, index=symbols),
            covariance_matrix
        )
        result.method = 'black_litterman'

        return result
```

### Rebalancing Strategies

```python
from datetime import datetime, timedelta
from enum import Enum

class RebalanceTrigger(Enum):
    CALENDAR = "calendar"
    THRESHOLD = "threshold"
    SIGNAL = "signal"


@dataclass
class RebalanceDecision:
    """Decision to rebalance or not."""
    should_rebalance: bool
    trigger: Optional[RebalanceTrigger]
    target_weights: Dict[str, float]
    current_weights: Dict[str, float]
    trades: Dict[str, float]
    estimated_cost: float


class RebalancingEngine:
    """Manage portfolio rebalancing decisions."""

    def __init__(
        self,
        calendar_frequency: str = 'monthly',  # daily, weekly, monthly, quarterly
        drift_threshold: float = 0.05,  # 5% drift triggers rebalance
        min_trade_value: float = 100,
        transaction_cost_pct: float = 0.001
    ):
        self.calendar_frequency = calendar_frequency
        self.drift_threshold = drift_threshold
        self.min_trade_value = min_trade_value
        self.transaction_cost_pct = transaction_cost_pct

        self.last_rebalance = None

    def check_rebalance(
        self,
        current_weights: Dict[str, float],
        target_weights: Dict[str, float],
        portfolio_value: float
    ) -> RebalanceDecision:
        """Check if rebalancing is needed."""
        # Check calendar trigger
        calendar_trigger = self._check_calendar_trigger()

        # Check threshold trigger
        max_drift = self._calculate_max_drift(current_weights, target_weights)
        threshold_trigger = max_drift > self.drift_threshold

        should_rebalance = calendar_trigger or threshold_trigger

        # Calculate trades needed
        trades = {}
        estimated_cost = 0

        if should_rebalance:
            for symbol in set(current_weights.keys()) | set(target_weights.keys()):
                current = current_weights.get(symbol, 0)
                target = target_weights.get(symbol, 0)
                diff = target - current

                trade_value = diff * portfolio_value

                if abs(trade_value) >= self.min_trade_value:
                    trades[symbol] = diff
                    estimated_cost += abs(trade_value) * self.transaction_cost_pct

        trigger = None
        if calendar_trigger:
            trigger = RebalanceTrigger.CALENDAR
        elif threshold_trigger:
            trigger = RebalanceTrigger.THRESHOLD

        return RebalanceDecision(
            should_rebalance=should_rebalance and len(trades) > 0,
            trigger=trigger,
            target_weights=target_weights,
            current_weights=current_weights,
            trades=trades,
            estimated_cost=estimated_cost
        )

    def _check_calendar_trigger(self) -> bool:
        """Check if calendar-based rebalance is due."""
        if self.last_rebalance is None:
            return True

        now = datetime.now()
        days_since = (now - self.last_rebalance).days

        if self.calendar_frequency == 'daily':
            return days_since >= 1
        elif self.calendar_frequency == 'weekly':
            return days_since >= 7
        elif self.calendar_frequency == 'monthly':
            return days_since >= 30
        elif self.calendar_frequency == 'quarterly':
            return days_since >= 90

        return False

    def _calculate_max_drift(
        self,
        current: Dict[str, float],
        target: Dict[str, float]
    ) -> float:
        """Calculate maximum weight drift from target."""
        max_drift = 0
        for symbol in set(current.keys()) | set(target.keys()):
            curr = current.get(symbol, 0)
            tgt = target.get(symbol, 0)
            drift = abs(curr - tgt)
            max_drift = max(max_drift, drift)
        return max_drift

    def record_rebalance(self):
        """Record that rebalancing occurred."""
        self.last_rebalance = datetime.now()


class CostAwareRebalancer:
    """Rebalance considering transaction costs vs. tracking error."""

    def __init__(
        self,
        transaction_cost_pct: float = 0.001,
        tracking_error_aversion: float = 10.0
    ):
        self.transaction_cost_pct = transaction_cost_pct
        self.tracking_error_aversion = tracking_error_aversion

    def optimize_trades(
        self,
        current_weights: Dict[str, float],
        target_weights: Dict[str, float],
        covariance_matrix: pd.DataFrame,
        portfolio_value: float
    ) -> Dict[str, float]:
        """
        Find optimal trades balancing cost vs. tracking error.

        Minimize: transaction_cost + λ * tracking_error²
        """
        symbols = list(set(current_weights.keys()) | set(target_weights.keys()))
        n = len(symbols)

        current = np.array([current_weights.get(s, 0) for s in symbols])
        target = np.array([target_weights.get(s, 0) for s in symbols])
        sigma = covariance_matrix.loc[symbols, symbols].values

        def objective(new_weights):
            # Transaction cost
            trades = np.abs(new_weights - current)
            cost = np.sum(trades) * self.transaction_cost_pct * portfolio_value

            # Tracking error vs. target
            diff = new_weights - target
            te_squared = np.dot(diff, np.dot(sigma, diff))

            return cost + self.tracking_error_aversion * te_squared

        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        ]

        # Bounds
        bounds = [(0, 1) for _ in range(n)]

        result = minimize(
            objective,
            current,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )

        optimal_weights = result.x
        trades = {s: optimal_weights[i] - current[i]
                 for i, s in enumerate(symbols)
                 if abs(optimal_weights[i] - current[i]) > 0.001}

        return trades
```

### Multi-Strategy Management

```python
@dataclass
class StrategyAllocation:
    """Allocation to a single strategy."""
    strategy_id: str
    weight: float
    risk_budget: float
    positions: Dict[str, float]


class MultiStrategyManager:
    """Manage portfolio of multiple trading strategies."""

    def __init__(
        self,
        strategies: List[str],
        initial_weights: Dict[str, float] = None
    ):
        self.strategies = strategies
        self.weights = initial_weights or {s: 1/len(strategies) for s in strategies}
        self.strategy_returns = {s: [] for s in strategies}

    def update_weights(
        self,
        method: str = 'risk_parity',
        lookback_days: int = 60
    ):
        """Update strategy weights based on performance."""
        if method == 'equal':
            self.weights = {s: 1/len(self.strategies) for s in self.strategies}

        elif method == 'risk_parity':
            # Calculate strategy volatilities
            vols = {}
            for s in self.strategies:
                if len(self.strategy_returns[s]) >= lookback_days:
                    returns = np.array(self.strategy_returns[s][-lookback_days:])
                    vols[s] = np.std(returns) * np.sqrt(252)
                else:
                    vols[s] = 0.20  # Default volatility

            # Inverse volatility weighting
            inv_vols = {s: 1/v for s, v in vols.items()}
            total = sum(inv_vols.values())
            self.weights = {s: iv/total for s, iv in inv_vols.items()}

        elif method == 'momentum':
            # Weight by recent performance
            returns_3m = {}
            for s in self.strategies:
                if len(self.strategy_returns[s]) >= 63:
                    cum_ret = np.prod([1 + r for r in self.strategy_returns[s][-63:]]) - 1
                    returns_3m[s] = max(cum_ret, 0)  # Only positive momentum
                else:
                    returns_3m[s] = 0

            total = sum(returns_3m.values())
            if total > 0:
                self.weights = {s: r/total for s, r in returns_3m.items()}
            else:
                self.weights = {s: 1/len(self.strategies) for s in self.strategies}

    def record_strategy_return(self, strategy_id: str, daily_return: float):
        """Record daily return for a strategy."""
        if strategy_id in self.strategy_returns:
            self.strategy_returns[strategy_id].append(daily_return)

    def get_combined_positions(
        self,
        strategy_positions: Dict[str, Dict[str, float]]
    ) -> Dict[str, float]:
        """Combine positions from all strategies weighted by allocation."""
        combined = {}

        for strategy_id, positions in strategy_positions.items():
            weight = self.weights.get(strategy_id, 0)

            for symbol, pos in positions.items():
                if symbol not in combined:
                    combined[symbol] = 0
                combined[symbol] += pos * weight

        return combined

    def get_strategy_correlation(self, lookback_days: int = 60) -> pd.DataFrame:
        """Calculate correlation between strategies."""
        data = {}
        for s in self.strategies:
            if len(self.strategy_returns[s]) >= lookback_days:
                data[s] = self.strategy_returns[s][-lookback_days:]

        if len(data) < 2:
            return pd.DataFrame()

        df = pd.DataFrame(data)
        return df.corr()
```

## Configuration

```yaml
# portfolioengine.yaml
allocation:
  method: risk_parity  # mean_variance, risk_parity, hrp, black_litterman, equal
  mean_variance:
    risk_aversion: 1.0
    allow_short: false
    max_weight: 0.20
  risk_parity:
    target_volatility: 0.15
  black_litterman:
    tau: 0.05
    risk_aversion: 2.5

rebalancing:
  frequency: monthly  # daily, weekly, monthly, quarterly
  drift_threshold: 0.05
  min_trade_value: 100
  cost_aware: true
  tracking_error_aversion: 10.0

constraints:
  max_position: 0.20
  min_position: 0.01
  max_sector: 0.30
  max_turnover: 0.50  # Monthly turnover limit

multi_strategy:
  weight_method: risk_parity  # equal, risk_parity, momentum
  reweight_frequency: monthly
  correlation_threshold: 0.80  # Alert if strategies too correlated

tax:
  enabled: false
  harvest_threshold: 0.03  # 3% loss to harvest
  wash_sale_days: 31

reporting:
  performance_attribution: true
  risk_decomposition: true
```

## Roadmap

### v1.0 (Core)
- [x] Mean-variance optimization
- [x] Risk parity
- [x] Basic rebalancing
- [x] Multi-strategy weights

### v1.1 (Advanced Allocation)
- [ ] Black-Litterman
- [ ] Hierarchical Risk Parity
- [ ] Factor-based allocation

### v1.2 (Rebalancing)
- [ ] Cost-aware rebalancing
- [ ] Tax-loss harvesting
- [ ] Optimal trade scheduling

### v2.0 (Analytics)
- [ ] Performance attribution
- [ ] Risk decomposition
- [ ] What-if scenarios
- [ ] Portfolio dashboard
