# DataStream: Unified Market Data Pipeline

## Product Overview

**DataStream** is the central data infrastructure of the FIREKit ecosystem. It provides a unified interface to ingest, normalize, and serve market data from multiple sources including stocks, cryptocurrency, and alternative data providers.

### Key Value Propositions

- **Unified API**: Single interface for stocks, crypto, and alternative data
- **Cost Optimized**: Intelligent routing to minimize API costs
- **Point-in-Time**: Proper handling of survivorship and lookahead bias
- **High Performance**: Parquet-based storage (10x smaller than CSV, 3x faster reads)
- **Real-Time Ready**: WebSocket streaming for live data needs

## Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                          DataStream                                  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │
│  │   Alpaca    │  │  Polygon    │  │  CoinGecko  │  │   Binance   │ │
│  │  Connector  │  │  Connector  │  │  Connector  │  │  Connector  │ │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘ │
│         │                │                │                │        │
│         └────────────────┴────────────────┴────────────────┘        │
│                                   │                                  │
│                          ┌────────▼────────┐                        │
│                          │   Normalizer    │                        │
│                          │  (Schema Align) │                        │
│                          └────────┬────────┘                        │
│                                   │                                  │
│         ┌─────────────────────────┼─────────────────────────┐       │
│         │                         │                         │       │
│  ┌──────▼──────┐          ┌───────▼──────┐          ┌──────▼──────┐│
│  │   Storage   │          │    Cache     │          │  Streaming  ││
│  │  (Parquet)  │          │   (Redis)    │          │ (WebSocket) ││
│  └──────┬──────┘          └───────┬──────┘          └──────┬──────┘│
│         │                         │                         │       │
│         └─────────────────────────┴─────────────────────────┘       │
│                                   │                                  │
│                          ┌────────▼────────┐                        │
│                          │  Query Engine   │                        │
│                          │    (DuckDB)     │                        │
│                          └────────┬────────┘                        │
│                                   │                                  │
│                          ┌────────▼────────┐                        │
│                          │   Unified API   │                        │
│                          └─────────────────┘                        │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Data Sources

| Source | Asset Type | Data Types | Cost | Latency |
|--------|------------|------------|------|---------|
| Alpaca | US Stocks | OHLCV, Trades, Quotes | Free-$9/mo | 1.5ms |
| Polygon | US Stocks | OHLCV, Reference, News | $29-199/mo | 2ms |
| CoinGecko | Crypto | OHLCV, Market Cap, Volume | Free-$129/mo | ~30s |
| Binance | Crypto | OHLCV, Order Book, Trades | Free | <10ms |
| Alpha Vantage | Global Stocks | OHLCV, Fundamentals | Free-$50/mo | ~1s |
| SEC EDGAR | US Stocks | Filings, Financials | Free | Daily |
| Yahoo Finance | Global | OHLCV (unofficial) | Free | Variable |

## Technical Specification

### Core Data Models

```python
from dataclasses import dataclass
from datetime import datetime
from typing import Optional
import pandas as pd

@dataclass
class OHLCV:
    """Standard OHLCV bar format."""
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    vwap: Optional[float] = None
    trade_count: Optional[int] = None

    def to_dict(self) -> dict:
        return {
            'symbol': self.symbol,
            'timestamp': self.timestamp,
            'open': self.open,
            'high': self.high,
            'low': self.low,
            'close': self.close,
            'volume': self.volume,
            'vwap': self.vwap,
            'trade_count': self.trade_count
        }

@dataclass
class Quote:
    """Best bid/ask quote."""
    symbol: str
    timestamp: datetime
    bid_price: float
    bid_size: int
    ask_price: float
    ask_size: int

@dataclass
class Trade:
    """Individual trade execution."""
    symbol: str
    timestamp: datetime
    price: float
    size: int
    exchange: str
    conditions: list

@dataclass
class Fundamental:
    """Company fundamental data."""
    symbol: str
    period_end: datetime
    revenue: float
    net_income: float
    eps: float
    book_value: float
    market_cap: float
    pe_ratio: float
    dividend_yield: float
```

### Connector Implementation

```python
from abc import ABC, abstractmethod
from typing import List, Optional
import asyncio
import aiohttp

class BaseConnector(ABC):
    """Abstract base class for data connectors."""

    def __init__(self, api_key: str, rate_limit: int = 100):
        self.api_key = api_key
        self.rate_limit = rate_limit
        self.semaphore = asyncio.Semaphore(rate_limit)

    @abstractmethod
    async def get_bars(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        timeframe: str = '1D'
    ) -> List[OHLCV]:
        pass

    @abstractmethod
    async def get_quotes(self, symbol: str) -> Quote:
        pass

    @abstractmethod
    async def stream_bars(self, symbols: List[str]):
        pass


class AlpacaConnector(BaseConnector):
    """Alpaca Markets data connector."""

    BASE_URL = "https://data.alpaca.markets/v2"

    async def get_bars(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        timeframe: str = '1D'
    ) -> List[OHLCV]:
        async with self.semaphore:
            async with aiohttp.ClientSession() as session:
                url = f"{self.BASE_URL}/stocks/{symbol}/bars"
                params = {
                    'start': start.isoformat(),
                    'end': end.isoformat(),
                    'timeframe': timeframe,
                    'limit': 10000
                }
                headers = {
                    'APCA-API-KEY-ID': self.api_key,
                    'APCA-API-SECRET-KEY': self.api_secret
                }

                async with session.get(url, params=params, headers=headers) as resp:
                    data = await resp.json()
                    return [
                        OHLCV(
                            symbol=symbol,
                            timestamp=bar['t'],
                            open=bar['o'],
                            high=bar['h'],
                            low=bar['l'],
                            close=bar['c'],
                            volume=bar['v'],
                            vwap=bar.get('vw'),
                            trade_count=bar.get('n')
                        )
                        for bar in data['bars']
                    ]


class BinanceConnector(BaseConnector):
    """Binance cryptocurrency data connector."""

    BASE_URL = "https://api.binance.com/api/v3"

    TIMEFRAME_MAP = {
        '1m': '1m', '5m': '5m', '15m': '15m', '1h': '1h',
        '4h': '4h', '1D': '1d', '1W': '1w', '1M': '1M'
    }

    async def get_bars(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        timeframe: str = '1D'
    ) -> List[OHLCV]:
        async with self.semaphore:
            async with aiohttp.ClientSession() as session:
                url = f"{self.BASE_URL}/klines"
                params = {
                    'symbol': symbol.replace('/', ''),
                    'interval': self.TIMEFRAME_MAP[timeframe],
                    'startTime': int(start.timestamp() * 1000),
                    'endTime': int(end.timestamp() * 1000),
                    'limit': 1000
                }

                async with session.get(url, params=params) as resp:
                    data = await resp.json()
                    return [
                        OHLCV(
                            symbol=symbol,
                            timestamp=datetime.fromtimestamp(bar[0] / 1000),
                            open=float(bar[1]),
                            high=float(bar[2]),
                            low=float(bar[3]),
                            close=float(bar[4]),
                            volume=float(bar[5]),
                            trade_count=int(bar[8])
                        )
                        for bar in data
                    ]
```

### Unified Data API

```python
class DataStream:
    """Unified interface for all data sources."""

    def __init__(self, config: dict):
        self.connectors = self._init_connectors(config)
        self.storage = ParquetStorage(config['storage_path'])
        self.cache = RedisCache(config['redis_url'])

    def _init_connectors(self, config: dict) -> dict:
        """Initialize configured data connectors."""
        connectors = {}

        if 'alpaca' in config:
            connectors['alpaca'] = AlpacaConnector(
                api_key=config['alpaca']['api_key'],
                api_secret=config['alpaca']['api_secret']
            )

        if 'binance' in config:
            connectors['binance'] = BinanceConnector(
                api_key=config['binance'].get('api_key')
            )

        # Add more connectors...
        return connectors

    async def get_bars(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        timeframe: str = '1D',
        source: str = 'auto'
    ) -> pd.DataFrame:
        """
        Get OHLCV bars for a symbol.

        Args:
            symbol: Ticker symbol (e.g., 'AAPL' or 'BTC/USDT')
            start: Start datetime
            end: End datetime
            timeframe: Bar timeframe ('1m', '5m', '1h', '1D', etc.)
            source: Data source ('auto', 'alpaca', 'binance', etc.)

        Returns:
            DataFrame with OHLCV data
        """
        # Check cache first
        cache_key = f"bars:{symbol}:{start}:{end}:{timeframe}"
        cached = await self.cache.get(cache_key)
        if cached is not None:
            return cached

        # Check local storage
        stored = await self.storage.get_bars(symbol, start, end, timeframe)
        if stored is not None and len(stored) > 0:
            # Check if we need to update with recent data
            last_bar = stored.index[-1]
            if last_bar >= end:
                return stored
            start = last_bar + timedelta(days=1)

        # Fetch from source
        connector = self._select_connector(symbol, source)
        bars = await connector.get_bars(symbol, start, end, timeframe)

        # Convert to DataFrame
        df = pd.DataFrame([bar.to_dict() for bar in bars])
        df.set_index('timestamp', inplace=True)

        # Store and cache
        await self.storage.save_bars(symbol, df, timeframe)
        await self.cache.set(cache_key, df, ttl=300)  # 5 min cache

        return df

    def _select_connector(self, symbol: str, source: str) -> BaseConnector:
        """Select appropriate connector based on symbol and preference."""
        if source != 'auto':
            return self.connectors[source]

        # Auto-detect based on symbol format
        if '/' in symbol or symbol.endswith('USDT') or symbol.endswith('BTC'):
            # Crypto symbol
            if 'binance' in self.connectors:
                return self.connectors['binance']
            return self.connectors['coingecko']
        else:
            # Stock symbol
            if 'alpaca' in self.connectors:
                return self.connectors['alpaca']
            return self.connectors['polygon']

    async def get_universe(
        self,
        date: datetime,
        index: str = 'SP500'
    ) -> List[str]:
        """Get point-in-time universe constituents."""
        return await self.storage.get_universe(date, index)

    async def stream(
        self,
        symbols: List[str],
        data_type: str = 'bars'
    ):
        """Stream real-time data for symbols."""
        connector = self._select_connector(symbols[0], 'auto')

        if data_type == 'bars':
            async for bar in connector.stream_bars(symbols):
                yield bar
        elif data_type == 'quotes':
            async for quote in connector.stream_quotes(symbols):
                yield quote
        elif data_type == 'trades':
            async for trade in connector.stream_trades(symbols):
                yield trade
```

### Storage Layer

```python
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path

class ParquetStorage:
    """High-performance Parquet-based data storage."""

    SCHEMA = pa.schema([
        ('timestamp', pa.timestamp('ns')),
        ('open', pa.float64()),
        ('high', pa.float64()),
        ('low', pa.float64()),
        ('close', pa.float64()),
        ('volume', pa.float64()),
        ('vwap', pa.float64()),
        ('trade_count', pa.int64())
    ])

    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    def _get_path(self, symbol: str, timeframe: str) -> Path:
        """Get storage path for symbol/timeframe."""
        # Partition by symbol first character for better disk access
        partition = symbol[0].upper()
        return self.base_path / partition / f"{symbol}_{timeframe}.parquet"

    async def save_bars(
        self,
        symbol: str,
        df: pd.DataFrame,
        timeframe: str
    ):
        """Save bars to Parquet with upsert behavior."""
        path = self._get_path(symbol, timeframe)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Load existing data if present
        if path.exists():
            existing = pq.read_table(path).to_pandas()
            df = pd.concat([existing, df]).drop_duplicates()
            df.sort_index(inplace=True)

        # Write with compression
        table = pa.Table.from_pandas(df, schema=self.SCHEMA)
        pq.write_table(
            table,
            path,
            compression='zstd',
            compression_level=3
        )

    async def get_bars(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        timeframe: str
    ) -> Optional[pd.DataFrame]:
        """Read bars from Parquet storage."""
        path = self._get_path(symbol, timeframe)

        if not path.exists():
            return None

        # Use predicate pushdown for efficient reading
        filters = [
            ('timestamp', '>=', start),
            ('timestamp', '<=', end)
        ]

        table = pq.read_table(path, filters=filters)
        return table.to_pandas()

    def get_storage_stats(self) -> dict:
        """Get storage statistics."""
        total_size = sum(f.stat().st_size for f in self.base_path.rglob('*.parquet'))
        file_count = len(list(self.base_path.rglob('*.parquet')))

        return {
            'total_size_mb': total_size / (1024 * 1024),
            'file_count': file_count,
            'avg_file_size_mb': total_size / file_count / (1024 * 1024) if file_count > 0 else 0
        }
```

### Point-in-Time Universe

```python
class PointInTimeUniverse:
    """Manages historical index composition to avoid survivorship bias."""

    def __init__(self, storage: ParquetStorage):
        self.storage = storage
        self._cache = {}

    async def load_index_history(self, index: str):
        """Load historical index composition from SEC filings."""
        # For S&P 500, we track additions/deletions
        changes = await self._fetch_index_changes(index)

        # Build point-in-time lookup
        composition = {}
        current = set()

        for date, added, removed in sorted(changes):
            current = current.union(added) - removed
            composition[date] = current.copy()

        self._cache[index] = composition

    async def get_universe(self, date: datetime, index: str = 'SP500') -> List[str]:
        """Get index constituents as of specific date."""
        if index not in self._cache:
            await self.load_index_history(index)

        # Find most recent composition before date
        composition = self._cache[index]
        valid_dates = [d for d in composition.keys() if d <= date]

        if not valid_dates:
            return []

        latest = max(valid_dates)
        return list(composition[latest])

    async def get_delisted(
        self,
        start: datetime,
        end: datetime
    ) -> List[dict]:
        """Get stocks that were delisted in period."""
        # Include bankrupt, merged, acquired companies
        return await self._fetch_delistings(start, end)
```

## Data Quality

### Validation Pipeline

```python
class DataValidator:
    """Validates incoming market data."""

    def validate_ohlcv(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Validate OHLCV data and return cleaned data with issues."""
        issues = []

        # Check for required columns
        required = ['open', 'high', 'low', 'close', 'volume']
        missing = set(required) - set(df.columns)
        if missing:
            issues.append(f"Missing columns: {missing}")
            return df, issues

        # OHLC relationship checks
        invalid_high = df['high'] < df[['open', 'close']].max(axis=1)
        if invalid_high.any():
            issues.append(f"{invalid_high.sum()} bars with high < max(open, close)")
            df.loc[invalid_high, 'high'] = df.loc[invalid_high, ['open', 'close']].max(axis=1)

        invalid_low = df['low'] > df[['open', 'close']].min(axis=1)
        if invalid_low.any():
            issues.append(f"{invalid_low.sum()} bars with low > min(open, close)")
            df.loc[invalid_low, 'low'] = df.loc[invalid_low, ['open', 'close']].min(axis=1)

        # Negative volume check
        negative_vol = df['volume'] < 0
        if negative_vol.any():
            issues.append(f"{negative_vol.sum()} bars with negative volume")
            df.loc[negative_vol, 'volume'] = 0

        # Extreme price moves (likely errors)
        returns = df['close'].pct_change()
        extreme = returns.abs() > 0.5  # 50% move in one bar
        if extreme.any():
            issues.append(f"{extreme.sum()} bars with >50% price move (verify)")

        # Missing data gaps
        if isinstance(df.index, pd.DatetimeIndex):
            gaps = df.index.to_series().diff()
            expected_gap = gaps.mode()[0]
            large_gaps = gaps > expected_gap * 5
            if large_gaps.any():
                issues.append(f"{large_gaps.sum()} potential data gaps detected")

        return df, issues

    def detect_splits(self, df: pd.DataFrame) -> List[dict]:
        """Detect potential stock splits in price series."""
        splits = []
        close = df['close']
        volume = df['volume']

        # Large price drop with volume spike
        price_change = close.pct_change()
        vol_change = volume.pct_change()

        for i, (pc, vc) in enumerate(zip(price_change, vol_change)):
            if pd.isna(pc) or pd.isna(vc):
                continue

            # Check for common split ratios
            ratio = close.iloc[i-1] / close.iloc[i] if close.iloc[i] != 0 else 0

            if ratio > 1.8 and vc > 1.0:  # Price halved, volume doubled
                likely_ratio = round(ratio)
                splits.append({
                    'date': df.index[i],
                    'ratio': f"{likely_ratio}:1",
                    'price_before': close.iloc[i-1],
                    'price_after': close.iloc[i]
                })

        return splits
```

## Configuration

```yaml
# datastream.yaml
sources:
  alpaca:
    api_key: ${ALPACA_API_KEY}
    api_secret: ${ALPACA_API_SECRET}
    data_feed: iex  # iex (free) or sip (paid)
    rate_limit: 200

  polygon:
    api_key: ${POLYGON_API_KEY}
    rate_limit: 100

  binance:
    api_key: ${BINANCE_API_KEY}  # Optional for public endpoints
    api_secret: ${BINANCE_API_SECRET}
    testnet: false
    rate_limit: 1200

  coingecko:
    api_key: ${COINGECKO_API_KEY}
    pro: false
    rate_limit: 30

storage:
  type: parquet
  path: ./data/market
  compression: zstd
  partition_by: [symbol, year]

cache:
  type: redis
  url: redis://localhost:6379
  ttl_seconds: 300
  max_memory: 1gb

streaming:
  enabled: true
  buffer_size: 1000
  reconnect_delay: 5

validation:
  enabled: true
  auto_fix: true
  max_price_change: 0.5
  log_issues: true

universe:
  indexes:
    - SP500
    - NASDAQ100
    - RUSSELL2000
  update_frequency: daily
```

## Cost Optimization

### Smart Source Routing

```python
class CostOptimizer:
    """Optimizes data fetching costs across sources."""

    # Monthly cost per request (approximate)
    COSTS = {
        'alpaca_free': 0,
        'alpaca_paid': 0.00003,
        'polygon_basic': 0.0001,
        'polygon_advanced': 0.00005,
        'coingecko_free': 0,
        'coingecko_pro': 0.00001,
        'binance': 0
    }

    def select_cheapest(
        self,
        symbol: str,
        data_type: str,
        freshness_required: str
    ) -> str:
        """Select cheapest source that meets requirements."""
        candidates = []

        for source, cost in self.COSTS.items():
            if self._meets_requirements(source, symbol, data_type, freshness_required):
                candidates.append((source, cost))

        # Sort by cost, return cheapest
        candidates.sort(key=lambda x: x[1])
        return candidates[0][0] if candidates else None

    def estimate_monthly_cost(self, usage: dict) -> float:
        """Estimate monthly costs based on usage patterns."""
        total = 0
        for source, requests in usage.items():
            total += requests * self.COSTS.get(source, 0)
        return total
```

### Budget Recommendations

| Monthly Budget | Stocks | Crypto | Alternative |
|----------------|--------|--------|-------------|
| $0 | Alpha Vantage Free + yfinance | CoinGecko Demo + Binance | SEC EDGAR |
| $10-50 | Alpaca ($9) | CoinGecko + Binance | Finnhub |
| $100-300 | Polygon Advanced ($199) | CoinGecko Pro | Benzinga |

## Roadmap

### v1.0 (Core)
- [x] Alpaca connector
- [x] Binance connector
- [x] Parquet storage
- [x] Basic caching
- [x] Data validation

### v1.1 (Expansion)
- [ ] Polygon connector
- [ ] CoinGecko connector
- [ ] Alpha Vantage connector
- [ ] SEC EDGAR connector
- [ ] Point-in-time universe

### v1.2 (Streaming)
- [ ] WebSocket streaming
- [ ] Real-time data normalization
- [ ] Streaming to VectorForge

### v2.0 (Enterprise)
- [ ] Multi-region deployment
- [ ] Data quality dashboard
- [ ] Cost tracking/alerting
- [ ] Custom connector SDK
