"""DataStream demo pipeline.

Run with ``cd datastream && python3 -m datastream.demo``. Writes
``hub/data/datastream.json`` following ``hub/SCHEMA.md``: ingests a
50-symbol, 3-year synthetic universe (with injected data issues) into a
Parquet store, runs the quality engine, and applies a point-in-time
universe to show survivorship-safe coverage.
"""

from __future__ import annotations

import json
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from datastream import __version__
from datastream.quality import QualityEngine
from datastream.sources import SyntheticSource
from datastream.store import ParquetStore
from datastream.universe import PointInTimeUniverse

HUB_DATA_DIR = Path(__file__).resolve().parents[2] / "hub" / "data"

SEED = 20260612
N_SYMBOLS = 50
START, END = "2021-01-04", "2023-12-29"


def _round(value: float, digits: int = 4) -> float:
    out = round(float(value), digits)
    return 0.0 if out == 0 else out  # avoid -0.0 in JSON


def build_universe(source: SyntheticSource) -> PointInTimeUniverse:
    """Staggered membership: some symbols join late, some are delisted early."""
    universe = PointInTimeUniverse()
    symbols = source.list_symbols()
    for i, sym in enumerate(symbols):
        if i % 9 == 4:  # joins ~9 months in
            universe.add(sym, "2021-10-01")
        elif i % 9 == 7:  # delisted ~7 months before the end
            universe.add(sym, START, "2023-06-01")
        else:
            universe.add(sym, START)
    return universe


def run_pipeline() -> dict:
    source = SyntheticSource(
        n_symbols=N_SYMBOLS, start=START, end=END, seed=SEED, issue_rate=0.012
    )
    raw = source.fetch()

    universe = build_universe(source)
    member_rows = universe.filter_frame(raw)

    with tempfile.TemporaryDirectory(prefix="datastream_demo_") as tmp:
        store = ParquetStore(tmp)
        store.write(member_rows)
        stored = store.read_many()
        store_stats = store.stats()

    engine = QualityEngine(outlier_z=6.0, max_gap_bdays=3, repair=True)
    clean, report = engine.run(stored)

    dates = pd.bdate_range(START, END)
    member_counts = universe.member_counts(dates)
    weekly = member_counts.resample("W-FRI").last().dropna()

    issue_counts = report.issue_counts
    repaired_counts: dict[str, int] = {}
    examples: dict[str, str] = {}
    for sym_report in report.per_symbol.values():
        for issue in sym_report.issues:
            key = issue.issue_type.value
            if issue.repaired:
                repaired_counts[key] = repaired_counts.get(key, 0) + 1
            examples.setdefault(key, f"{issue.symbol} @ {issue.timestamp.date()}")

    issues_rows = [
        [
            kind,
            int(found),
            int(repaired_counts.get(kind, 0)),
            "repaired" if repaired_counts.get(kind, 0) else "flagged",
            examples.get(kind, ""),
        ]
        for kind, found in sorted(issue_counts.items(), key=lambda kv: -kv[1])
    ]

    worst = report.worst_symbols(15)
    scores = sorted(report.per_symbol.values(), key=lambda r: r.score)

    return {
        "schema_version": 1,
        "product": "datastream",
        "title": "DataStream",
        "tagline": "Unified market data pipeline with quality scoring and point-in-time universes",
        "version": __version__,
        "generated_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "status": "ok",
        "summary_metrics": [
            {"label": "Symbols ingested", "value": str(len(report.per_symbol))},
            {"label": "Rows ingested", "value": f"{report.n_rows:,}"},
            {"label": "Rows after cleaning", "value": f"{len(clean):,}"},
            {"label": "Quality score", "value": f"{report.score:.2f}", "unit": "%"},
            {"label": "Issues found", "value": str(report.n_issues)},
            {"label": "Issues repaired", "value": str(report.n_repaired)},
            {"label": "Store size", "value": f"{store_stats['total_mb']:.2f}", "unit": "MB"},
            {"label": "Date range", "value": f"{START} → {END}"},
        ],
        "charts": [
            {
                "id": "pit_coverage",
                "title": "Point-in-time universe membership (survivorship-safe)",
                "type": "line",
                "x_label": "Date",
                "y_label": "Active members",
                "x": [d.strftime("%Y-%m-%d") for d in weekly.index],
                "series": [
                    {"name": "Members", "data": [int(v) for v in weekly.to_numpy()]}
                ],
            },
            {
                "id": "worst_quality",
                "title": "Lowest per-symbol quality scores (15 worst of 50)",
                "type": "bar",
                "x_label": "Symbol",
                "y_label": "Quality score (%)",
                "x": [r.symbol for r in worst],
                "series": [
                    {"name": "Quality score", "data": [_round(r.score, 2) for r in worst]}
                ],
            },
        ],
        "tables": [
            {
                "id": "issues_by_type",
                "title": "Data quality issues by type",
                "columns": ["Issue type", "Found", "Repaired", "Action", "Example"],
                "rows": issues_rows,
            },
            {
                "id": "symbol_quality",
                "title": "Per-symbol quality (10 worst)",
                "columns": ["Symbol", "Rows", "Issues", "Repaired", "Score (%)"],
                "rows": [
                    [r.symbol, int(r.n_rows), len(r.issues), int(r.n_repaired), _round(r.score, 2)]
                    for r in scores[:10]
                ],
            },
        ],
        "notes": (
            f"Synthetic 50-symbol GBM universe ({START} to {END}, seed {SEED}) with a 1.2% "
            "injected issue rate. Pipeline: SyntheticSource -> point-in-time membership filter "
            "-> ParquetStore (zstd, partitioned by symbol) -> QualityEngine. Duplicates, NaNs, "
            "non-positive prices and OHLC violations are repaired; return outliers and calendar "
            "gaps are flagged for review."
        ),
    }


def main() -> Path:
    payload = run_pipeline()
    HUB_DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = HUB_DATA_DIR / "datastream.json"
    with out_path.open("w") as fh:
        json.dump(payload, fh, indent=2)
        fh.write("\n")
    print(f"wrote {out_path}")
    for metric in payload["summary_metrics"]:
        unit = metric.get("unit", "")
        print(f"  {metric['label']}: {metric['value']}{unit}")
    return out_path


if __name__ == "__main__":
    main()
