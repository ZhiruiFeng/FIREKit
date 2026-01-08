"""
Portfolio Metrics Contract
==========================

API contract for portfolio-level analytics.
These stubs define the public interface; implementations follow.

Contract Tests: tests/contract/test_metrics_contract.py
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from vectorforge.analysis.metrics import PerformanceMetrics


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

    diversification_ratio: float  # σ_weighted / σ_portfolio
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

    weights: dict[str, float]  # Sector → weight
    date: pd.Timestamp

    def top_sector(self) -> tuple[str, float]:
        """Get largest sector exposure."""
        ...

    def to_series(self) -> pd.Series:
        """Convert to pandas Series."""
        ...


@dataclass
class ReturnAttribution:
    """Per-asset return attribution."""

    symbol: str
    weight: float  # Average weight
    asset_return: float  # Asset's return
    contribution: float  # Weight × return
    contribution_pct: float  # Contribution as % of total return


class PortfolioMetrics(ABC):
    """
    Extended metrics for multi-asset portfolios.

    Inherits from PerformanceMetrics and adds portfolio-specific analytics.

    Example:
        >>> metrics = PortfolioMetrics(
        ...     returns=portfolio_returns,
        ...     weights=weight_history,
        ...     asset_returns=per_asset_returns,
        ... )
        >>> print(metrics.diversification_ratio())
        >>> print(metrics.sector_exposure())
    """

    # Inherited from PerformanceMetrics
    @abstractmethod
    def sharpe_ratio(self) -> float:
        """Annualized Sharpe ratio."""
        ...

    @abstractmethod
    def sortino_ratio(self) -> float:
        """Sortino ratio using downside deviation."""
        ...

    @abstractmethod
    def max_drawdown(self) -> float:
        """Maximum peak-to-trough decline."""
        ...

    @abstractmethod
    def total_return(self) -> float:
        """Total cumulative return."""
        ...

    @abstractmethod
    def annual_return(self) -> float:
        """Annualized return."""
        ...

    @abstractmethod
    def volatility(self) -> float:
        """Annualized volatility."""
        ...

    # Portfolio-specific metrics

    @abstractmethod
    def diversification_ratio(self) -> float:
        """
        Calculate diversification ratio.

        DR = σ_weighted / σ_portfolio

        Where σ_weighted is the weighted average of individual volatilities
        and σ_portfolio is the portfolio volatility.

        DR > 1 indicates diversification benefit.

        Returns:
            Diversification ratio
        """
        ...

    @abstractmethod
    def herfindahl_index(self) -> pd.Series:
        """
        Calculate Herfindahl-Hirschman Index over time.

        HHI = Σ(weight_i²)

        HHI = 1 means single asset, HHI → 0 means highly diversified.

        Returns:
            Series of HHI values indexed by date
        """
        ...

    @abstractmethod
    def concentration_metrics(self) -> ConcentrationMetrics:
        """
        Calculate portfolio concentration metrics.

        Returns average concentration over the backtest period.

        Returns:
            ConcentrationMetrics with HHI, top-N weights, effective N
        """
        ...

    @abstractmethod
    def diversification_metrics(self) -> DiversificationMetrics:
        """
        Calculate portfolio diversification metrics.

        Returns:
            DiversificationMetrics with DR, correlations
        """
        ...

    @abstractmethod
    def sector_exposure(self) -> pd.DataFrame:
        """
        Calculate sector exposure over time.

        Returns:
            DataFrame with dates as index, sectors as columns, weights as values
        """
        ...

    @abstractmethod
    def sector_exposure_at(self, date: pd.Timestamp) -> SectorExposure:
        """Get sector exposure at a specific date."""
        ...

    @abstractmethod
    def portfolio_beta(
        self,
        benchmark_returns: pd.Series,
    ) -> float:
        """
        Calculate portfolio beta relative to benchmark.

        β = Cov(r_p, r_b) / Var(r_b)

        Args:
            benchmark_returns: Benchmark return series

        Returns:
            Portfolio beta
        """
        ...

    @abstractmethod
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
        ...

    @abstractmethod
    def correlation_matrix(
        self,
        window: int | None = None,
    ) -> pd.DataFrame:
        """
        Calculate pairwise correlation matrix.

        Args:
            window: Rolling window (None = full period)

        Returns:
            Correlation matrix (symbols × symbols)
        """
        ...

    @abstractmethod
    def rolling_correlation(
        self,
        window: int = 60,
    ) -> pd.DataFrame:
        """
        Calculate rolling average pairwise correlation.

        Args:
            window: Rolling window size

        Returns:
            Series of average correlations over time
        """
        ...

    @abstractmethod
    def contribution_to_return(self) -> pd.DataFrame:
        """
        Calculate per-asset contribution to portfolio return.

        contribution_i = weight_i × return_i

        Returns:
            DataFrame of contributions (dates × symbols)
        """
        ...

    @abstractmethod
    def return_attribution(self) -> list[ReturnAttribution]:
        """
        Get return attribution summary by asset.

        Returns:
            List of ReturnAttribution sorted by contribution
        """
        ...

    @abstractmethod
    def contribution_to_risk(self) -> pd.DataFrame:
        """
        Calculate per-asset contribution to portfolio risk.

        Uses marginal contribution to risk (MCTR).

        Returns:
            DataFrame of risk contributions (dates × symbols)
        """
        ...

    @abstractmethod
    def turnover_analysis(self) -> dict[str, Any]:
        """
        Analyze portfolio turnover.

        Returns:
            Dictionary with total, average, max turnover and dates
        """
        ...

    @abstractmethod
    def top_n_concentration(self, n: int = 5) -> pd.Series:
        """
        Calculate weight in top N holdings over time.

        Args:
            n: Number of top holdings

        Returns:
            Series of top-N concentration indexed by date
        """
        ...

    @abstractmethod
    def generate_report(self) -> dict[str, Any]:
        """
        Generate comprehensive portfolio report.

        Returns:
            Dictionary with all key metrics
        """
        ...

    @abstractmethod
    def to_dataframe(self) -> pd.DataFrame:
        """Convert report to DataFrame."""
        ...

    @classmethod
    @abstractmethod
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
            sector_map: Optional symbol → sector mapping
            risk_free_rate: Annual risk-free rate

        Returns:
            Configured PortfolioMetrics
        """
        ...
