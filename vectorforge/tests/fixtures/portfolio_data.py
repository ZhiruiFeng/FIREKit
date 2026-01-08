"""
Test fixtures for multi-asset portfolio data.

Provides sample data generators for testing portfolio functionality.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd


def generate_price_series(
    start_date: date = date(2020, 1, 1),
    n_days: int = 252,
    initial_price: float = 100.0,
    volatility: float = 0.02,
    drift: float = 0.0005,
    seed: int | None = None,
) -> pd.DataFrame:
    """
    Generate synthetic OHLCV price data.

    Args:
        start_date: Start date for the series
        n_days: Number of trading days
        initial_price: Starting price
        volatility: Daily volatility (standard deviation)
        drift: Daily drift (mean return)
        seed: Random seed for reproducibility

    Returns:
        DataFrame with OHLCV columns and DatetimeIndex
    """
    if seed is not None:
        np.random.seed(seed)

    # Generate dates (skip weekends)
    dates = []
    current = start_date
    while len(dates) < n_days:
        if current.weekday() < 5:  # Monday=0, Friday=4
            dates.append(current)
        current += timedelta(days=1)

    # Generate returns using geometric Brownian motion
    returns = np.random.normal(drift, volatility, n_days)
    prices = initial_price * np.exp(np.cumsum(returns))

    # Generate OHLCV from close prices
    close = prices
    high = close * (1 + np.abs(np.random.normal(0, volatility / 2, n_days)))
    low = close * (1 - np.abs(np.random.normal(0, volatility / 2, n_days)))
    open_price = low + (high - low) * np.random.random(n_days)

    # Ensure OHLC consistency
    high = np.maximum(high, np.maximum(open_price, close))
    low = np.minimum(low, np.minimum(open_price, close))

    # Generate volume (log-normal)
    volume = np.exp(np.random.normal(15, 1, n_days))

    df = pd.DataFrame(
        {
            "open": open_price,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=pd.DatetimeIndex(dates),
    )

    return df


def generate_multi_asset_data(
    n_symbols: int = 10,
    n_days: int = 252,
    start_date: date = date(2020, 1, 1),
    seed: int = 42,
    include_missing: bool = False,
    missing_ratio: float = 0.05,
) -> dict[str, pd.DataFrame]:
    """
    Generate synthetic multi-asset OHLCV data.

    Args:
        n_symbols: Number of symbols to generate
        n_days: Number of trading days
        start_date: Start date for the series
        seed: Random seed for reproducibility
        include_missing: Whether to include missing data
        missing_ratio: Ratio of missing data if include_missing=True

    Returns:
        Dictionary mapping symbol -> DataFrame
    """
    np.random.seed(seed)

    # Generate symbol names
    symbols = [f"SYM{i:03d}" for i in range(n_symbols)]

    data = {}
    for i, symbol in enumerate(symbols):
        # Vary parameters per symbol
        volatility = 0.01 + np.random.random() * 0.03
        drift = -0.001 + np.random.random() * 0.002
        initial_price = 50 + np.random.random() * 150

        df = generate_price_series(
            start_date=start_date,
            n_days=n_days,
            initial_price=initial_price,
            volatility=volatility,
            drift=drift,
            seed=seed + i,
        )

        if include_missing:
            # Randomly drop some rows
            drop_mask = np.random.random(len(df)) < missing_ratio
            df = df[~drop_mask]

        data[symbol] = df

    return data


def generate_sp500_like_data(
    n_symbols: int = 50,
    n_days: int = 252 * 5,  # 5 years
    start_date: date = date(2019, 1, 2),
    seed: int = 42,
) -> dict[str, pd.DataFrame]:
    """
    Generate synthetic S&P 500-like data with realistic characteristics.

    Includes:
    - Correlation between symbols (market factor)
    - Different volatility regimes
    - Sector-like groupings

    Args:
        n_symbols: Number of symbols
        n_days: Number of trading days
        start_date: Start date
        seed: Random seed

    Returns:
        Dictionary mapping symbol -> DataFrame
    """
    np.random.seed(seed)

    # Generate dates
    dates = []
    current = start_date
    while len(dates) < n_days:
        if current.weekday() < 5:
            dates.append(current)
        current += timedelta(days=1)
    dates = pd.DatetimeIndex(dates)

    # Generate market factor (affects all stocks)
    market_returns = np.random.normal(0.0003, 0.012, n_days)

    # Add some volatility clustering
    for i in range(1, n_days):
        market_returns[i] += 0.3 * np.sign(market_returns[i - 1]) * np.random.random() * 0.005

    # Generate symbol names with sector prefixes
    sectors = ["TECH", "FIN", "HEALTH", "CONS", "IND"]
    symbols = []
    for i in range(n_symbols):
        sector = sectors[i % len(sectors)]
        symbols.append(f"{sector}_{i:03d}")

    data = {}
    for i, symbol in enumerate(symbols):
        # Beta to market (0.5 to 1.5)
        beta = 0.5 + np.random.random()

        # Idiosyncratic volatility
        idio_vol = 0.01 + np.random.random() * 0.02

        # Generate returns
        idio_returns = np.random.normal(0, idio_vol, n_days)
        returns = beta * market_returns + idio_returns

        # Convert to prices
        initial_price = 50 + np.random.random() * 200
        prices = initial_price * np.exp(np.cumsum(returns))

        # Generate OHLCV
        close = prices
        high = close * (1 + np.abs(np.random.normal(0, idio_vol / 2, n_days)))
        low = close * (1 - np.abs(np.random.normal(0, idio_vol / 2, n_days)))
        open_price = low + (high - low) * np.random.random(n_days)

        high = np.maximum(high, np.maximum(open_price, close))
        low = np.minimum(low, np.minimum(open_price, close))

        volume = np.exp(np.random.normal(15, 1, n_days))

        df = pd.DataFrame(
            {
                "open": open_price,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
            },
            index=dates,
        )

        data[symbol] = df

    return data


def generate_symbol_metadata(
    symbols: list[str],
    seed: int = 42,
) -> dict[str, dict]:
    """
    Generate metadata for symbols.

    Args:
        symbols: List of symbol names
        seed: Random seed

    Returns:
        Dictionary mapping symbol -> metadata dict
    """
    np.random.seed(seed)

    sectors = ["Technology", "Financials", "Healthcare", "Consumer", "Industrial"]
    exchanges = ["NYSE", "NASDAQ"]

    metadata = {}
    for symbol in symbols:
        # Parse sector from symbol if present
        if "_" in symbol:
            sector_prefix = symbol.split("_")[0]
            sector_map = {
                "TECH": "Technology",
                "FIN": "Financials",
                "HEALTH": "Healthcare",
                "CONS": "Consumer",
                "IND": "Industrial",
            }
            sector = sector_map.get(sector_prefix, np.random.choice(sectors))
        else:
            sector = np.random.choice(sectors)

        metadata[symbol] = {
            "sector": sector,
            "industry": f"{sector} Industry {np.random.randint(1, 5)}",
            "market_cap": np.exp(np.random.normal(23, 2)) * 1000,  # Log-normal market cap
            "exchange": np.random.choice(exchanges),
        }

    return metadata


# Pre-generated fixtures for quick access

SAMPLE_SYMBOLS = ["AAPL", "GOOG", "MSFT", "AMZN", "META", "NVDA", "TSLA", "JPM", "JNJ", "V"]


def get_sample_portfolio_data(
    n_days: int = 252,
    seed: int = 42,
) -> dict[str, pd.DataFrame]:
    """
    Get sample portfolio data with realistic ticker names.

    Args:
        n_days: Number of trading days
        seed: Random seed

    Returns:
        Dictionary mapping real-looking symbols to DataFrames
    """
    np.random.seed(seed)

    # Realistic initial prices and volatilities
    symbol_params = {
        "AAPL": (175.0, 0.018),
        "GOOG": (140.0, 0.020),
        "MSFT": (370.0, 0.016),
        "AMZN": (150.0, 0.022),
        "META": (350.0, 0.025),
        "NVDA": (480.0, 0.030),
        "TSLA": (240.0, 0.035),
        "JPM": (170.0, 0.015),
        "JNJ": (160.0, 0.012),
        "V": (270.0, 0.014),
    }

    data = {}
    for i, (symbol, (price, vol)) in enumerate(symbol_params.items()):
        data[symbol] = generate_price_series(
            start_date=date(2023, 1, 3),
            n_days=n_days,
            initial_price=price,
            volatility=vol,
            drift=0.0003,
            seed=seed + i,
        )

    return data


def get_sample_metadata() -> dict:
    """Get metadata for sample symbols."""
    return {
        "AAPL": {
            "sector": "Technology",
            "industry": "Consumer Electronics",
            "market_cap": 3e12,
            "exchange": "NASDAQ",
        },
        "GOOG": {
            "sector": "Technology",
            "industry": "Internet Services",
            "market_cap": 2e12,
            "exchange": "NASDAQ",
        },
        "MSFT": {
            "sector": "Technology",
            "industry": "Software",
            "market_cap": 2.8e12,
            "exchange": "NASDAQ",
        },
        "AMZN": {
            "sector": "Consumer",
            "industry": "E-commerce",
            "market_cap": 1.6e12,
            "exchange": "NASDAQ",
        },
        "META": {
            "sector": "Technology",
            "industry": "Social Media",
            "market_cap": 1.2e12,
            "exchange": "NASDAQ",
        },
        "NVDA": {
            "sector": "Technology",
            "industry": "Semiconductors",
            "market_cap": 1.3e12,
            "exchange": "NASDAQ",
        },
        "TSLA": {
            "sector": "Consumer",
            "industry": "Electric Vehicles",
            "market_cap": 0.8e12,
            "exchange": "NASDAQ",
        },
        "JPM": {
            "sector": "Financials",
            "industry": "Banking",
            "market_cap": 0.5e12,
            "exchange": "NYSE",
        },
        "JNJ": {
            "sector": "Healthcare",
            "industry": "Pharmaceuticals",
            "market_cap": 0.4e12,
            "exchange": "NYSE",
        },
        "V": {
            "sector": "Financials",
            "industry": "Payments",
            "market_cap": 0.5e12,
            "exchange": "NYSE",
        },
    }
