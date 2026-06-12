# DataStream

Unified market data pipeline for the FIREKit ecosystem (offline MVP): a single
canonical OHLCV schema, pluggable sources, a Parquet-backed local store with
caching, a data-quality engine with repair + scoring, and point-in-time
universe membership to avoid survivorship bias.

## Install

```bash
cd datastream && pip install -e ".[dev]"
```

## Usage

```python
from datastream import ParquetStore, PointInTimeUniverse, QualityEngine, SyntheticSource

# 1. Ingest a seeded synthetic universe (or FileSource for CSV/Parquet files)
source = SyntheticSource(n_symbols=10, start="2022-01-03", end="2023-12-29", seed=42)
bars = source.fetch()  # canonical long frame: timestamp, symbol, OHLCV (+vwap)

# 2. Store locally (Parquet, partitioned by symbol, upsert + LRU read cache)
store = ParquetStore("./market_data")
store.write(bars)
aapl = store.read("SYM001", start="2023-01-01")

# 3. Quality: detect/repair duplicates, bad prices, OHLC violations; flag outliers & gaps
clean, report = QualityEngine().run(store.read_many())
print(f"quality score {report.score:.1f}%, {report.n_issues} issues, {report.n_repaired} repaired")

# 4. Point-in-time universe (no survivorship bias)
universe = PointInTimeUniverse()
universe.add("SYM001", "2022-01-03")
universe.add("SYM002", "2022-01-03", end="2023-06-01")  # delisted
universe.members_as_of("2023-07-01")  # -> ["SYM001"]
survivorship_safe = universe.filter_frame(clean)
```

## Demo

```bash
cd datastream && python3 -m datastream.demo   # writes ../hub/data/datastream.json
```

## Tests

```bash
cd datastream && python3 -m pytest tests -q
```

Out of scope for this MVP (per design doc): live API connectors, WebSocket
streaming, Redis caching, corporate-action adjustments (handled by VectorForge).
