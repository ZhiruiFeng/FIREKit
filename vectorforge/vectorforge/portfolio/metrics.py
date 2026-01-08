"""
Portfolio Metrics

Extended analytics for multi-asset portfolios including
concentration, diversification, and attribution.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    pass


@dataclass
class ConcentrationMetrics:
    """Portfolio concentration metrics."""

    herfindahl_index: float  # HHI: sum of squared weights (0-1)
    top_1_weight: float  # Weight of largest position
    top_5_weight: float  # Weight of top 5 positions
    top_10_weight: float  # Weight of top 10 positions
    effective_n: float  # Effective number of positions (1/HHI)

    def __repr__(self) -> str:
        return (
            f"ConcentrationMetrics(\n"
            f"  HHI={self.herfindahl_index:.4f},\n"
            f"  effective_n={self.effective_n:.1f},\n"
            f"  top_5_weight={self.top_5_weight:.2%}\n"
            f")"
        )


@dataclass
class DiversificationMetrics:
    """Portfolio diversification metrics."""

    diversification_ratio: float  # sigma_weighted / sigma_portfolio
    avg_correlation: float  # Average pairwise correlation
    max_correlation: float  # Maximum pairwise correlation
    min_correlation: float  # Minimum pairwise correlation

    def __repr__(self) -> str:
        return (
            f"DiversificationMetrics(\n"
            f"  diversification_ratio={self.diversification_ratio:.2f},\n"
            f"  avg_correlation={self.avg_correlation:.2f}\n"
            f")"
        )


@dataclass
class SectorExposure:
    """Portfolio sector exposure at a point in time."""

    weights: dict[str, float]  # Sector -> weight
    date: pd.Timestamp

    def top_sector(self) -> tuple[str, float]:
        """Get largest sector exposure."""
        if not self.weights:
            return ("Unknown", 0.0)
        max_sector = max(self.weights.items(), key=lambda x: x[1])
        return max_sector

    def to_series(self) -> pd.Series:
        """Convert to pandas Series."""
        return pd.Series(self.weights, name=self.date)


@dataclass
class ReturnAttribution:
    """Per-asset return attribution."""

    symbol: str
    weight: float  # Average weight
    asset_return: float  # Asset's return
    contribution: float  # Weight x return
    contribution_pct: float  # Contribution as % of total return


class PortfolioMetrics:
    """
    Extended metrics for multi-asset portfolios.

    Provides concentration, diversification, and attribution analytics
    from a portfolio backtest result.

    Example:
        >>> metrics = PortfolioMetrics(
        ...     returns=portfolio_returns,
        ...     weights=weight_history,
        ...     asset_returns=per_asset_returns,
        ... )
        >>> print(metrics.diversification_ratio())
        >>> print(metrics.sector_exposure())
    """

    def __init__(
        self,
        returns: pd.Series,
        weights_history: pd.DataFrame,
        asset_returns: pd.DataFrame,
        sector_map: dict[str, str] | None = None,
        risk_free_rate: float = 0.0,
    ):
        """
        Initialize portfolio metrics.

        Args:
            returns: Portfolio returns series
            weights_history: DataFrame of weights (dates x symbols)
            asset_returns: DataFrame of per-asset returns (dates x symbols)
            sector_map: Optional symbol -> sector mapping
            risk_free_rate: Annual risk-free rate for Sharpe calculation
        """
        self._returns = returns
        self._weights = weights_history
        self._asset_returns = asset_returns
        self._sector_map = sector_map or {}
        self._risk_free_rate = risk_free_rate

    @property
    def symbols(self) -> list[str]:
        """List of symbols in portfolio."""
        return list(self._weights.columns)

    @property
    def dates(self) -> pd.DatetimeIndex:
        """Date range of portfolio."""
        return self._weights.index

    # =========================================================================
    # Performance metrics (inherited from PerformanceMetrics interface)
    # =========================================================================

    def sharpe_ratio(self) -> float:
        """Annualized Sharpe ratio."""
        if len(self._returns) == 0:
            return 0.0
        excess_returns = self._returns - self._risk_free_rate / 252
        std = excess_returns.std()
        if std == 0 or np.isnan(std):
            return 0.0
        return float(excess_returns.mean() / std * np.sqrt(252))

    def sortino_ratio(self) -> float:
        """Sortino ratio using downside deviation."""
        if len(self._returns) == 0:
            return 0.0
        excess_returns = self._returns - self._risk_free_rate / 252
        downside = excess_returns[excess_returns < 0]
        if len(downside) == 0:
            return float("inf") if excess_returns.mean() > 0 else 0.0
        downside_std = downside.std()
        if downside_std == 0 or np.isnan(downside_std):
            return 0.0
        return float(excess_returns.mean() / downside_std * np.sqrt(252))

    def max_drawdown(self) -> float:
        """Maximum peak-to-trough decline."""
        if len(self._returns) == 0:
            return 0.0
        cumulative = (1 + self._returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = cumulative / running_max - 1
        return float(drawdown.min())

    def total_return(self) -> float:
        """Total cumulative return."""
        if len(self._returns) == 0:
            return 0.0
        return float((1 + self._returns).prod() - 1)

    def annual_return(self) -> float:
        """Annualized return."""
        total = self.total_return()
        n_years = len(self._returns) / 252
        if n_years <= 0:
            return 0.0
        return float((1 + total) ** (1 / n_years) - 1)

    def volatility(self) -> float:
        """Annualized volatility."""
        if len(self._returns) == 0:
            return 0.0
        return float(self._returns.std() * np.sqrt(252))

    # =========================================================================
    # Portfolio-specific metrics
    # =========================================================================

    def diversification_ratio(self) -> float:
        """
        Calculate diversification ratio.

        DR = sigma_weighted / sigma_portfolio

        Where sigma_weighted is the weighted average of individual volatilities
        and sigma_portfolio is the portfolio volatility.

        DR > 1 indicates diversification benefit.

        Returns:
            Diversification ratio
        """
        if len(self._asset_returns) < 2:
            return 1.0

        # Calculate individual asset volatilities
        asset_vols = self._asset_returns.std()

        # Get average weights
        avg_weights = self._weights.mean()

        # Weighted average volatility
        weighted_vol = (avg_weights * asset_vols).sum()

        # Portfolio volatility
        portfolio_vol = self.volatility() / np.sqrt(252)  # Daily

        if portfolio_vol == 0 or np.isnan(portfolio_vol):
            return 1.0

        return float(weighted_vol / portfolio_vol)

    def herfindahl_index(self) -> pd.Series:
        """
        Calculate Herfindahl-Hirschman Index over time.

        HHI = sum(weight_i^2)

        HHI = 1 means single asset, HHI -> 0 means highly diversified.

        Returns:
            Series of HHI values indexed by date
        """
        hhi = (self._weights**2).sum(axis=1)
        return hhi

    def concentration_metrics(self) -> ConcentrationMetrics:
        """
        Calculate portfolio concentration metrics.

        Returns average concentration over the backtest period.

        Returns:
            ConcentrationMetrics with HHI, top-N weights, effective N
        """
        # Average weights
        avg_weights = self._weights.mean()
        sorted_weights = avg_weights.sort_values(ascending=False)

        # HHI
        hhi = float((avg_weights**2).sum())

        # Top-N weights
        top_1 = float(sorted_weights.iloc[0]) if len(sorted_weights) > 0 else 0.0
        top_5 = (
            float(sorted_weights.iloc[:5].sum())
            if len(sorted_weights) >= 5
            else float(sorted_weights.sum())
        )
        top_10 = (
            float(sorted_weights.iloc[:10].sum())
            if len(sorted_weights) >= 10
            else float(sorted_weights.sum())
        )

        # Effective N
        effective_n = 1.0 / hhi if hhi > 0 else float(len(avg_weights))

        return ConcentrationMetrics(
            herfindahl_index=hhi,
            top_1_weight=top_1,
            top_5_weight=top_5,
            top_10_weight=top_10,
            effective_n=effective_n,
        )

    def diversification_metrics(self) -> DiversificationMetrics:
        """
        Calculate portfolio diversification metrics.

        Returns:
            DiversificationMetrics with DR, correlations
        """
        dr = self.diversification_ratio()

        # Correlation matrix
        corr_matrix = self.correlation_matrix()

        # Extract off-diagonal correlations
        n = len(corr_matrix)
        if n < 2:
            return DiversificationMetrics(
                diversification_ratio=dr,
                avg_correlation=0.0,
                max_correlation=0.0,
                min_correlation=0.0,
            )

        mask = ~np.eye(n, dtype=bool)
        off_diag = corr_matrix.values[mask]

        return DiversificationMetrics(
            diversification_ratio=dr,
            avg_correlation=float(off_diag.mean()),
            max_correlation=float(off_diag.max()),
            min_correlation=float(off_diag.min()),
        )

    def sector_exposure(self) -> pd.DataFrame:
        """
        Calculate sector exposure over time.

        Returns:
            DataFrame with dates as index, sectors as columns, weights as values
        """
        if not self._sector_map:
            return pd.DataFrame()

        # Get unique sectors
        sectors = list(set(self._sector_map.values()))

        # Calculate sector weights for each date
        sector_weights = pd.DataFrame(index=self._weights.index, columns=sectors, dtype=float)
        sector_weights.fillna(0.0, inplace=True)

        for symbol in self._weights.columns:
            sector = self._sector_map.get(symbol, "Unknown")
            if sector not in sector_weights.columns:
                sector_weights[sector] = 0.0
            sector_weights[sector] += self._weights[symbol]

        return sector_weights

    def sector_exposure_at(self, date: pd.Timestamp) -> SectorExposure:
        """Get sector exposure at a specific date."""
        sector_df = self.sector_exposure()

        if date not in sector_df.index:
            # Find closest date
            idx = sector_df.index.get_indexer([date], method="nearest")[0]
            date = sector_df.index[idx]

        weights_dict = sector_df.loc[date].to_dict()
        return SectorExposure(weights=weights_dict, date=date)

    def portfolio_beta(
        self,
        benchmark_returns: pd.Series,
    ) -> float:
        """
        Calculate portfolio beta relative to benchmark.

        beta = Cov(r_p, r_b) / Var(r_b)

        Args:
            benchmark_returns: Benchmark return series

        Returns:
            Portfolio beta
        """
        # Align returns
        common_idx = self._returns.index.intersection(benchmark_returns.index)
        if len(common_idx) < 10:
            return 1.0

        portfolio_ret = self._returns.loc[common_idx]
        benchmark_ret = benchmark_returns.loc[common_idx]

        cov = portfolio_ret.cov(benchmark_ret)
        var = benchmark_ret.var()

        if var == 0 or np.isnan(var):
            return 1.0

        return float(cov / var)

    def rolling_beta(
        self,
        benchmark_returns: pd.Series,
        window: int = 60,
    ) -> pd.Series:
        """
        Calculate rolling portfolio beta.

        Args:
            benchmark_returns: Benchmark return series
            window: Rolling window size

        Returns:
            Series of rolling beta values
        """
        common_idx = self._returns.index.intersection(benchmark_returns.index)
        if len(common_idx) < window:
            return pd.Series(dtype=float)

        portfolio_ret = self._returns.loc[common_idx]
        benchmark_ret = benchmark_returns.loc[common_idx]

        rolling_cov = portfolio_ret.rolling(window).cov(benchmark_ret)
        rolling_var = benchmark_ret.rolling(window).var()

        rolling_beta = rolling_cov / rolling_var
        return rolling_beta.dropna()

    def correlation_matrix(
        self,
        window: int | None = None,
    ) -> pd.DataFrame:
        """
        Calculate pairwise correlation matrix.

        Args:
            window: Rolling window (None = full period)

        Returns:
            Correlation matrix (symbols x symbols)
        """
        if window is None:
            return self._asset_returns.corr()
        else:
            return self._asset_returns.rolling(window).corr().iloc[-len(self.symbols) :]

    def rolling_correlation(
        self,
        window: int = 60,
    ) -> pd.Series:
        """
        Calculate rolling average pairwise correlation.

        Args:
            window: Rolling window size

        Returns:
            Series of average correlations over time
        """
        n = len(self.symbols)
        if n < 2:
            return pd.Series(dtype=float)

        # Rolling correlation for each pair
        rolling_corrs = []

        for i in range(n):
            for j in range(i + 1, n):
                sym1 = self.symbols[i]
                sym2 = self.symbols[j]
                rolling = self._asset_returns[sym1].rolling(window).corr(self._asset_returns[sym2])
                rolling_corrs.append(rolling)

        if not rolling_corrs:
            return pd.Series(dtype=float)

        avg_corr = pd.concat(rolling_corrs, axis=1).mean(axis=1)
        return avg_corr.dropna()

    def contribution_to_return(self) -> pd.DataFrame:
        """
        Calculate per-asset contribution to portfolio return.

        contribution_i = weight_i x return_i

        Returns:
            DataFrame of contributions (dates x symbols)
        """
        # Align weights with returns
        aligned_weights = self._weights.shift(1).iloc[1:]  # Use previous day weights
        aligned_returns = self._asset_returns.iloc[1:]

        common_idx = aligned_weights.index.intersection(aligned_returns.index)
        aligned_weights = aligned_weights.loc[common_idx]
        aligned_returns = aligned_returns.loc[common_idx]

        return aligned_weights * aligned_returns

    def return_attribution(self) -> list[ReturnAttribution]:
        """
        Get return attribution summary by asset.

        Returns:
            List of ReturnAttribution sorted by contribution
        """
        contributions = self.contribution_to_return()
        total_contrib = contributions.sum()
        total_return = self.total_return()

        attributions = []
        for symbol in self.symbols:
            avg_weight = float(self._weights[symbol].mean())
            asset_return = (
                float((1 + self._asset_returns[symbol]).prod() - 1)
                if len(self._asset_returns) > 0
                else 0.0
            )
            contribution = float(total_contrib.get(symbol, 0.0))
            contrib_pct = contribution / total_return if total_return != 0 else 0.0

            attributions.append(
                ReturnAttribution(
                    symbol=symbol,
                    weight=avg_weight,
                    asset_return=asset_return,
                    contribution=contribution,
                    contribution_pct=contrib_pct,
                )
            )

        # Sort by contribution
        attributions.sort(key=lambda x: x.contribution, reverse=True)
        return attributions

    def contribution_to_risk(self) -> pd.DataFrame:
        """
        Calculate per-asset contribution to portfolio risk.

        Uses marginal contribution to risk (MCTR).

        Returns:
            DataFrame of risk contributions (dates x symbols)
        """
        # Simplified: use weighted volatility contribution
        asset_vols = self._asset_returns.std()
        risk_contrib = self._weights * asset_vols
        return risk_contrib

    def turnover_analysis(self) -> dict[str, Any]:
        """
        Analyze portfolio turnover.

        Returns:
            Dictionary with total, average, max turnover and dates
        """
        # Calculate turnover from weight changes
        weight_changes = self._weights.diff().abs()
        turnover = weight_changes.sum(axis=1) / 2

        return {
            "total_turnover": float(turnover.sum()),
            "avg_turnover": float(turnover.mean()),
            "max_turnover": float(turnover.max()),
            "max_turnover_date": turnover.idxmax() if len(turnover) > 0 else None,
            "turnover_series": turnover,
        }

    def top_n_concentration(self, n: int = 5) -> pd.Series:
        """
        Calculate weight in top N holdings over time.

        Args:
            n: Number of top holdings

        Returns:
            Series of top-N concentration indexed by date
        """
        top_n_weights = []
        for idx in self._weights.index:
            row = self._weights.loc[idx].sort_values(ascending=False)
            top_n_weights.append(row.iloc[:n].sum())

        return pd.Series(top_n_weights, index=self._weights.index)

    def generate_report(self) -> dict[str, Any]:
        """
        Generate comprehensive portfolio report.

        Returns:
            Dictionary with all key metrics
        """
        concentration = self.concentration_metrics()
        diversification = self.diversification_metrics()
        turnover = self.turnover_analysis()

        return {
            "performance": {
                "total_return": self.total_return(),
                "annual_return": self.annual_return(),
                "volatility": self.volatility(),
                "sharpe_ratio": self.sharpe_ratio(),
                "sortino_ratio": self.sortino_ratio(),
                "max_drawdown": self.max_drawdown(),
            },
            "concentration": {
                "herfindahl_index": concentration.herfindahl_index,
                "effective_n": concentration.effective_n,
                "top_1_weight": concentration.top_1_weight,
                "top_5_weight": concentration.top_5_weight,
                "top_10_weight": concentration.top_10_weight,
            },
            "diversification": {
                "diversification_ratio": diversification.diversification_ratio,
                "avg_correlation": diversification.avg_correlation,
                "max_correlation": diversification.max_correlation,
                "min_correlation": diversification.min_correlation,
            },
            "turnover": {
                "total": turnover["total_turnover"],
                "average": turnover["avg_turnover"],
                "maximum": turnover["max_turnover"],
            },
        }

    def to_dataframe(self) -> pd.DataFrame:
        """Convert report to DataFrame."""
        report = self.generate_report()

        records = []
        for category, metrics in report.items():
            for metric, value in metrics.items():
                records.append(
                    {
                        "category": category,
                        "metric": metric,
                        "value": value,
                    }
                )

        return pd.DataFrame(records)

    @classmethod
    def from_backtest_result(
        cls,
        result: Any,  # PortfolioBacktestResult
        sector_map: dict[str, str] | None = None,
        risk_free_rate: float = 0.0,
    ) -> PortfolioMetrics:
        """
        Create PortfolioMetrics from backtest result.

        Args:
            result: PortfolioBacktestResult from engine
            sector_map: Optional symbol -> sector mapping
            risk_free_rate: Annual risk-free rate

        Returns:
            Configured PortfolioMetrics
        """
        return cls(
            returns=result.returns,
            weights_history=result.weights_history,
            asset_returns=result.asset_returns,
            sector_map=sector_map,
            risk_free_rate=risk_free_rate,
        )
