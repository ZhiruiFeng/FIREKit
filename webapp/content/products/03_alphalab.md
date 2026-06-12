# AlphaLab: Factor Research & Feature Engineering Platform

## Product Overview

**AlphaLab** is the research workbench of the FIREKit ecosystem, designed for discovering, testing, and validating alpha factors. It provides tools for systematic factor mining, feature engineering, and rigorous statistical validation to prevent overfitting.

### Key Value Propositions

- **Alpha101 Library**: Complete implementation of WorldQuant's 101 formulaic alphas
- **Custom Factor Builder**: Domain-specific language for rapid factor prototyping
- **Statistical Rigor**: Purged K-Fold CV, CPCV, and multiple testing correction
- **Factor Decay Analysis**: Understand how quickly alpha signals degrade
- **Combination Engine**: Optimal factor weighting and ensemble methods

## Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                          AlphaLab                                    │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                    Factor Universe                           │    │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐     │    │
│  │  │ Alpha101 │  │Technical │  │Fundamental│  │  Custom  │     │    │
│  │  │  Factors │  │Indicators│  │  Ratios   │  │ Factors  │     │    │
│  │  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘     │    │
│  │       └─────────────┴─────────────┴─────────────┘           │    │
│  └───────────────────────────┬─────────────────────────────────┘    │
│                              │                                       │
│  ┌───────────────────────────▼─────────────────────────────────┐    │
│  │                   Validation Engine                          │    │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐     │    │
│  │  │Information│  │Purged CV │  │  Factor  │  │  Multi   │     │    │
│  │  │Coefficient│  │  /CPCV   │  │  Decay   │  │  Test    │     │    │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────┘     │    │
│  └───────────────────────────┬─────────────────────────────────┘    │
│                              │                                       │
│  ┌───────────────────────────▼─────────────────────────────────┐    │
│  │                 Combination & Output                         │    │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐                   │    │
│  │  │  Factor  │  │ Ensemble │  │  Signal  │                   │    │
│  │  │ Weighting│  │  Methods │  │  Export  │                   │    │
│  │  └──────────┘  └──────────┘  └──────────┘                   │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Factor Categories

| Category | Examples | Expected IC | Typical Decay |
|----------|----------|-------------|---------------|
| Momentum | 12M-1M returns, 52-week high | 0.02-0.04 | 1-3 months |
| Value | P/E, P/B, EV/EBITDA | 0.01-0.03 | 6-12 months |
| Quality | ROE, debt/equity, earnings stability | 0.01-0.02 | 12+ months |
| Size | Market cap, log(market cap) | 0.01-0.02 | Long-term |
| Volatility | Historical vol, beta | 0.02-0.03 | 1-3 months |
| Technical | RSI, MACD, Bollinger bands | 0.01-0.03 | Days-weeks |

## Technical Specification

### Alpha101 Implementation

```python
import numpy as np
import pandas as pd
from typing import Callable

class Alpha101:
    """
    WorldQuant's 101 Formulaic Alphas implementation.
    Reference: https://arxiv.org/abs/1601.00991
    """

    def __init__(self, data: pd.DataFrame):
        """
        Initialize with OHLCV data.

        Args:
            data: DataFrame with columns [open, high, low, close, volume, vwap]
                  and MultiIndex (date, symbol)
        """
        self.data = data
        self.open = data['open']
        self.high = data['high']
        self.low = data['low']
        self.close = data['close']
        self.volume = data['volume']
        self.vwap = data.get('vwap', (self.high + self.low + self.close) / 3)
        self.returns = self.close.groupby('symbol').pct_change()

    # Utility functions
    def rank(self, x: pd.Series) -> pd.Series:
        """Cross-sectional rank (0 to 1)."""
        return x.groupby('date').rank(pct=True)

    def delta(self, x: pd.Series, d: int = 1) -> pd.Series:
        """Time-series difference."""
        return x.groupby('symbol').diff(d)

    def delay(self, x: pd.Series, d: int = 1) -> pd.Series:
        """Time-series lag."""
        return x.groupby('symbol').shift(d)

    def ts_sum(self, x: pd.Series, d: int) -> pd.Series:
        """Rolling sum."""
        return x.groupby('symbol').rolling(d).sum().droplevel(0)

    def ts_mean(self, x: pd.Series, d: int) -> pd.Series:
        """Rolling mean."""
        return x.groupby('symbol').rolling(d).mean().droplevel(0)

    def ts_std(self, x: pd.Series, d: int) -> pd.Series:
        """Rolling standard deviation."""
        return x.groupby('symbol').rolling(d).std().droplevel(0)

    def ts_max(self, x: pd.Series, d: int) -> pd.Series:
        """Rolling maximum."""
        return x.groupby('symbol').rolling(d).max().droplevel(0)

    def ts_min(self, x: pd.Series, d: int) -> pd.Series:
        """Rolling minimum."""
        return x.groupby('symbol').rolling(d).min().droplevel(0)

    def ts_argmax(self, x: pd.Series, d: int) -> pd.Series:
        """Days since rolling maximum."""
        return x.groupby('symbol').rolling(d).apply(np.argmax).droplevel(0)

    def ts_argmin(self, x: pd.Series, d: int) -> pd.Series:
        """Days since rolling minimum."""
        return x.groupby('symbol').rolling(d).apply(np.argmin).droplevel(0)

    def ts_corr(self, x: pd.Series, y: pd.Series, d: int) -> pd.Series:
        """Rolling correlation."""
        return x.groupby('symbol').rolling(d).corr(y).droplevel(0)

    def ts_cov(self, x: pd.Series, y: pd.Series, d: int) -> pd.Series:
        """Rolling covariance."""
        return x.groupby('symbol').rolling(d).cov(y).droplevel(0)

    def ts_rank(self, x: pd.Series, d: int) -> pd.Series:
        """Time-series rank (position in lookback window)."""
        return x.groupby('symbol').rolling(d).apply(
            lambda arr: pd.Series(arr).rank().iloc[-1] / len(arr)
        ).droplevel(0)

    # Alpha implementations
    def alpha001(self) -> pd.Series:
        """(rank(Ts_ArgMax(SignedPower(((returns < 0) ? stddev(returns, 20) : close), 2.), 5)) - 0.5)"""
        cond = self.returns < 0
        inner = pd.Series(np.where(cond, self.ts_std(self.returns, 20), self.close))
        return self.rank(self.ts_argmax(inner ** 2, 5)) - 0.5

    def alpha002(self) -> pd.Series:
        """(-1 * correlation(rank(delta(log(volume), 2)), rank(((close - open) / open)), 6))"""
        return -1 * self.ts_corr(
            self.rank(self.delta(np.log(self.volume), 2)),
            self.rank((self.close - self.open) / self.open),
            6
        )

    def alpha003(self) -> pd.Series:
        """(-1 * correlation(rank(open), rank(volume), 10))"""
        return -1 * self.ts_corr(self.rank(self.open), self.rank(self.volume), 10)

    def alpha004(self) -> pd.Series:
        """(-1 * Ts_Rank(rank(low), 9))"""
        return -1 * self.ts_rank(self.rank(self.low), 9)

    def alpha005(self) -> pd.Series:
        """(rank((open - (sum(vwap, 10) / 10))) * (-1 * abs(rank((close - vwap)))))"""
        return self.rank(self.open - self.ts_mean(self.vwap, 10)) * \
               (-1 * abs(self.rank(self.close - self.vwap)))

    def alpha006(self) -> pd.Series:
        """(-1 * correlation(open, volume, 10))"""
        return -1 * self.ts_corr(self.open, self.volume, 10)

    # ... alphas 7-100 follow similar patterns ...

    def alpha042(self) -> pd.Series:
        """(rank((vwap - close)) / rank((vwap + close)))"""
        return self.rank(self.vwap - self.close) / self.rank(self.vwap + self.close)

    def alpha053(self) -> pd.Series:
        """(-1 * delta((((close - low) - (high - close)) / (close - low)), 9))"""
        inner = ((self.close - self.low) - (self.high - self.close)) / (self.close - self.low)
        return -1 * self.delta(inner, 9)

    def alpha101(self) -> pd.Series:
        """((close - open) / ((high - low) + .001))"""
        return (self.close - self.open) / ((self.high - self.low) + 0.001)

    def compute_all(self) -> pd.DataFrame:
        """Compute all 101 alphas."""
        alphas = {}
        for i in range(1, 102):
            method_name = f'alpha{str(i).zfill(3)}'
            if hasattr(self, method_name):
                try:
                    alphas[f'alpha_{i}'] = getattr(self, method_name)()
                except Exception as e:
                    print(f"Error computing {method_name}: {e}")
        return pd.DataFrame(alphas)
```

### Custom Factor Builder

```python
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class FactorDefinition:
    """Declarative factor definition."""
    name: str
    formula: str
    category: str
    lookback: int
    description: str = ""

class FactorBuilder:
    """Build custom factors with a domain-specific language."""

    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.alpha101 = Alpha101(data)
        self._namespace = self._build_namespace()

    def _build_namespace(self) -> Dict[str, Any]:
        """Build evaluation namespace with available functions."""
        return {
            # Data columns
            'open': self.data['open'],
            'high': self.data['high'],
            'low': self.data['low'],
            'close': self.data['close'],
            'volume': self.data['volume'],
            'vwap': self.data.get('vwap'),
            'returns': self.data['close'].groupby('symbol').pct_change(),

            # Functions
            'rank': self.alpha101.rank,
            'delta': self.alpha101.delta,
            'delay': self.alpha101.delay,
            'ts_sum': self.alpha101.ts_sum,
            'ts_mean': self.alpha101.ts_mean,
            'ts_std': self.alpha101.ts_std,
            'ts_max': self.alpha101.ts_max,
            'ts_min': self.alpha101.ts_min,
            'ts_corr': self.alpha101.ts_corr,
            'ts_rank': self.alpha101.ts_rank,

            # NumPy
            'np': np,
            'log': np.log,
            'sqrt': np.sqrt,
            'abs': np.abs,
            'sign': np.sign,
        }

    def build(self, definition: FactorDefinition) -> pd.Series:
        """Build factor from definition."""
        return eval(definition.formula, {"__builtins__": {}}, self._namespace)

    def build_from_yaml(self, yaml_str: str) -> Dict[str, pd.Series]:
        """Build multiple factors from YAML definition."""
        import yaml
        definitions = yaml.safe_load(yaml_str)

        factors = {}
        for name, spec in definitions.items():
            factor_def = FactorDefinition(
                name=name,
                formula=spec['formula'],
                category=spec.get('category', 'custom'),
                lookback=spec.get('lookback', 20),
                description=spec.get('description', '')
            )
            factors[name] = self.build(factor_def)

        return factors


# Example YAML factor definitions
CUSTOM_FACTORS_YAML = """
momentum_12m_1m:
  formula: "delay(returns, 21).groupby('symbol').rolling(252-21).sum().droplevel(0)"
  category: momentum
  lookback: 252
  description: "12-month momentum excluding most recent month"

reversal_5d:
  formula: "-1 * ts_sum(returns, 5)"
  category: reversal
  lookback: 5
  description: "5-day mean reversion signal"

volume_momentum:
  formula: "ts_mean(volume, 5) / ts_mean(volume, 20) - 1"
  category: volume
  lookback: 20
  description: "Short-term volume relative to medium-term"

price_range:
  formula: "(ts_max(high, 20) - ts_min(low, 20)) / close"
  category: volatility
  lookback: 20
  description: "20-day price range as percentage of close"

overnight_return:
  formula: "open / delay(close, 1) - 1"
  category: technical
  lookback: 1
  description: "Overnight return (gap)"
"""
```

### Validation Engine

```python
class FactorValidator:
    """Rigorous factor validation to prevent overfitting."""

    def __init__(self, factors: pd.DataFrame, forward_returns: pd.DataFrame):
        """
        Args:
            factors: DataFrame of factor values (date, symbol) index
            forward_returns: DataFrame of forward returns at various horizons
        """
        self.factors = factors
        self.forward_returns = forward_returns

    def information_coefficient(
        self,
        factor_name: str,
        horizon: str = '1D'
    ) -> pd.Series:
        """
        Compute Information Coefficient (IC) - Spearman correlation with forward returns.

        Returns:
            Series of daily IC values
        """
        factor = self.factors[factor_name]
        returns = self.forward_returns[horizon]

        # Compute rank correlation for each date
        ic = factor.groupby('date').apply(
            lambda x: x.corr(returns.loc[x.index], method='spearman')
        )
        return ic

    def ic_summary(self, factor_name: str, horizon: str = '1D') -> dict:
        """Compute IC summary statistics."""
        ic = self.information_coefficient(factor_name, horizon)

        return {
            'mean_ic': ic.mean(),
            'std_ic': ic.std(),
            'ir': ic.mean() / ic.std(),  # Information Ratio
            'pct_positive': (ic > 0).mean(),
            't_stat': ic.mean() / (ic.std() / np.sqrt(len(ic))),
            'significant': abs(ic.mean() / (ic.std() / np.sqrt(len(ic)))) > 2
        }

    def purged_kfold_cv(
        self,
        factor_name: str,
        n_splits: int = 5,
        embargo_days: int = 5
    ) -> List[dict]:
        """
        Purged K-Fold cross-validation to prevent leakage.

        Args:
            factor_name: Factor to validate
            n_splits: Number of CV folds
            embargo_days: Gap between train and test to prevent leakage

        Returns:
            List of metrics for each fold
        """
        dates = self.factors.index.get_level_values('date').unique().sort_values()
        fold_size = len(dates) // n_splits

        results = []

        for i in range(n_splits):
            # Define test period
            test_start_idx = i * fold_size
            test_end_idx = (i + 1) * fold_size if i < n_splits - 1 else len(dates)
            test_dates = dates[test_start_idx:test_end_idx]

            # Define train period with purging
            train_dates = dates[
                ~dates.isin(test_dates) &
                ~dates.isin(dates[max(0, test_start_idx - embargo_days):test_start_idx]) &
                ~dates.isin(dates[test_end_idx:min(len(dates), test_end_idx + embargo_days)])
            ]

            # Compute IC on test set only
            test_factor = self.factors.loc[test_dates, factor_name]
            test_returns = self.forward_returns.loc[test_dates, '1D']

            ic = test_factor.groupby('date').apply(
                lambda x: x.corr(test_returns.loc[x.index], method='spearman')
            )

            results.append({
                'fold': i,
                'train_days': len(train_dates),
                'test_days': len(test_dates),
                'test_ic': ic.mean(),
                'test_ir': ic.mean() / ic.std() if ic.std() > 0 else 0
            })

        return results

    def combinatorial_purged_cv(
        self,
        factor_name: str,
        n_splits: int = 5,
        n_test_splits: int = 2,
        embargo_days: int = 5
    ) -> dict:
        """
        Combinatorial Purged Cross-Validation (CPCV).
        Tests multiple historical paths to reduce overfitting probability.

        Reference: López de Prado, "Advances in Financial Machine Learning"
        """
        from itertools import combinations

        dates = self.factors.index.get_level_values('date').unique().sort_values()
        fold_size = len(dates) // n_splits

        # Create fold boundaries
        folds = []
        for i in range(n_splits):
            start = i * fold_size
            end = (i + 1) * fold_size if i < n_splits - 1 else len(dates)
            folds.append(dates[start:end])

        # Generate all combinations of test folds
        test_combinations = list(combinations(range(n_splits), n_test_splits))

        results = []
        for test_fold_indices in test_combinations:
            # Combine test folds
            test_dates = pd.DatetimeIndex([])
            for idx in test_fold_indices:
                test_dates = test_dates.union(folds[idx])

            # Train on remaining folds with embargo
            train_dates = dates[~dates.isin(test_dates)]

            # Apply embargo around test boundaries
            for idx in test_fold_indices:
                fold_start = folds[idx][0]
                fold_end = folds[idx][-1]
                train_dates = train_dates[
                    (train_dates < fold_start - pd.Timedelta(days=embargo_days)) |
                    (train_dates > fold_end + pd.Timedelta(days=embargo_days))
                ]

            # Compute metrics
            test_factor = self.factors.loc[test_dates, factor_name]
            test_returns = self.forward_returns.loc[test_dates, '1D']

            ic = test_factor.groupby('date').apply(
                lambda x: x.corr(test_returns.loc[x.index], method='spearman')
            )

            results.append({
                'test_folds': test_fold_indices,
                'test_ic': ic.mean(),
                'test_ir': ic.mean() / ic.std() if ic.std() > 0 else 0
            })

        # Aggregate across all paths
        all_ics = [r['test_ic'] for r in results]

        return {
            'mean_ic': np.mean(all_ics),
            'std_ic': np.std(all_ics),
            'min_ic': np.min(all_ics),
            'max_ic': np.max(all_ics),
            'num_paths': len(results),
            'pct_positive': np.mean([ic > 0 for ic in all_ics]),
            'details': results
        }

    def factor_decay(
        self,
        factor_name: str,
        horizons: List[int] = [1, 2, 3, 5, 10, 20, 40, 60]
    ) -> pd.DataFrame:
        """
        Analyze how factor signal decays over time.

        Returns:
            DataFrame with IC at various forward horizons
        """
        decay = {}

        for horizon in horizons:
            horizon_key = f'{horizon}D'
            if horizon_key in self.forward_returns.columns:
                ic = self.information_coefficient(factor_name, horizon_key)
                decay[horizon] = {
                    'mean_ic': ic.mean(),
                    'ir': ic.mean() / ic.std() if ic.std() > 0 else 0
                }

        return pd.DataFrame(decay).T

    def multiple_testing_correction(
        self,
        ic_results: Dict[str, float],
        method: str = 'bonferroni'
    ) -> Dict[str, bool]:
        """
        Apply multiple testing correction when screening many factors.

        Args:
            ic_results: Dict of factor_name -> t-statistic
            method: 'bonferroni', 'holm', or 'fdr' (Benjamini-Hochberg)

        Returns:
            Dict of factor_name -> still_significant
        """
        from scipy import stats

        n_tests = len(ic_results)
        factors = list(ic_results.keys())
        t_stats = np.array([ic_results[f] for f in factors])
        p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), df=250))  # Assume 250 obs

        if method == 'bonferroni':
            # Divide alpha by number of tests
            adjusted_alpha = 0.05 / n_tests
            significant = p_values < adjusted_alpha

        elif method == 'holm':
            # Holm-Bonferroni step-down
            sorted_indices = np.argsort(p_values)
            significant = np.zeros(n_tests, dtype=bool)
            for i, idx in enumerate(sorted_indices):
                adjusted_alpha = 0.05 / (n_tests - i)
                if p_values[idx] < adjusted_alpha:
                    significant[idx] = True
                else:
                    break

        elif method == 'fdr':
            # Benjamini-Hochberg FDR
            sorted_indices = np.argsort(p_values)
            significant = np.zeros(n_tests, dtype=bool)
            for i, idx in enumerate(sorted_indices[::-1]):
                threshold = 0.05 * (n_tests - i) / n_tests
                if p_values[idx] < threshold:
                    significant[sorted_indices[:n_tests-i]] = True
                    break

        return {f: sig for f, sig in zip(factors, significant)}
```

### Factor Combination

```python
class FactorCombiner:
    """Combine multiple factors into ensemble signals."""

    def __init__(self, factors: pd.DataFrame, forward_returns: pd.Series):
        self.factors = factors
        self.forward_returns = forward_returns

    def equal_weight(self) -> pd.Series:
        """Simple equal-weighted combination."""
        # Standardize each factor cross-sectionally
        standardized = self.factors.groupby('date').transform(
            lambda x: (x - x.mean()) / x.std()
        )
        return standardized.mean(axis=1)

    def ic_weighted(self, lookback: int = 60) -> pd.Series:
        """Weight factors by their recent IC."""
        combined = pd.Series(index=self.factors.index, dtype=float)

        dates = self.factors.index.get_level_values('date').unique().sort_values()

        for i, date in enumerate(dates):
            if i < lookback:
                # Not enough history, use equal weight
                weights = np.ones(len(self.factors.columns)) / len(self.factors.columns)
            else:
                # Compute IC for each factor over lookback period
                lookback_dates = dates[i-lookback:i]
                weights = []
                for col in self.factors.columns:
                    ic = self.factors.loc[lookback_dates, col].groupby('date').apply(
                        lambda x: x.corr(self.forward_returns.loc[x.index], method='spearman')
                    ).mean()
                    weights.append(max(ic, 0))  # Only positive IC factors

                weights = np.array(weights)
                if weights.sum() > 0:
                    weights = weights / weights.sum()
                else:
                    weights = np.ones(len(self.factors.columns)) / len(self.factors.columns)

            # Apply weights
            day_factors = self.factors.loc[date]
            standardized = (day_factors - day_factors.mean()) / day_factors.std()
            combined.loc[date] = (standardized * weights).sum(axis=1)

        return combined

    def optimal_weights_ols(
        self,
        train_end: pd.Timestamp,
        regularization: float = 0.01
    ) -> Tuple[np.ndarray, pd.Series]:
        """Find optimal factor weights using regularized OLS."""
        from sklearn.linear_model import Ridge

        train_factors = self.factors.loc[:train_end]
        train_returns = self.forward_returns.loc[:train_end]

        # Standardize
        X = train_factors.groupby('date').transform(
            lambda x: (x - x.mean()) / x.std()
        ).values
        y = train_returns.values

        # Fit with regularization
        model = Ridge(alpha=regularization)
        model.fit(X, y)

        weights = model.coef_

        # Apply to full dataset
        all_standardized = self.factors.groupby('date').transform(
            lambda x: (x - x.mean()) / x.std()
        )
        combined = (all_standardized * weights).sum(axis=1)

        return weights, combined
```

## Configuration

```yaml
# alphalab.yaml
factors:
  alpha101:
    enabled: true
    subset: null  # null = all, or list of alpha numbers

  technical:
    - rsi_14
    - macd
    - bollinger_bands
    - atr_14

  fundamental:
    - pe_ratio
    - pb_ratio
    - roe
    - debt_equity

validation:
  ic_threshold: 0.02
  t_stat_threshold: 3.0  # Stricter than 2.0
  min_observations: 252

  cross_validation:
    method: cpcv  # purged_kfold or cpcv
    n_splits: 5
    embargo_days: 5

  multiple_testing:
    method: fdr
    alpha: 0.05

combination:
  method: ic_weighted  # equal, ic_weighted, or optimal
  lookback_days: 60
  regularization: 0.01

output:
  format: parquet
  path: ./factors
```

## Roadmap

### v1.0 (Core)
- [x] Alpha101 implementation
- [x] Custom factor builder
- [x] IC calculation
- [x] Purged K-Fold CV

### v1.1 (Advanced Validation)
- [ ] CPCV implementation
- [ ] Factor decay analysis
- [ ] Multiple testing correction
- [ ] Turnover analysis

### v1.2 (Combination)
- [ ] IC-weighted combination
- [ ] Optimal weight estimation
- [ ] Factor neutralization
- [ ] Sector adjustment

### v2.0 (Research Platform)
- [ ] Interactive factor explorer UI
- [ ] Factor database with versioning
- [ ] Automated factor screening
- [ ] Factor correlation analysis
