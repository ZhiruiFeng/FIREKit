# FIREKit Hub — Results JSON Contract

Every product ships a demo pipeline that writes its results to
`hub/data/<product>.json`. The hub website renders all files found there.
Demos must be deterministic (fixed seeds), self-contained (synthetic or
bundled data), and fast (< 60 s).

Run a single product demo:

```bash
cd <product>/ && python3 -m <package>.demo
```

Run everything and rebuild the hub data bundle:

```bash
python3 run_all.py
```

## JSON schema (version 1)

```jsonc
{
  "schema_version": 1,
  "product": "datastream",            // lowercase package name, matches filename
  "title": "DataStream",              // display name
  "tagline": "Unified data pipeline", // one-line description
  "version": "0.1.0",                 // package version
  "generated_at": "2026-06-12T00:00:00Z",
  "status": "ok",                     // "ok" | "error"
  "summary_metrics": [                // headline cards (4-8 items)
    {"label": "Symbols ingested", "value": "50"},
    {"label": "Quality score", "value": "99.2", "unit": "%"}
  ],
  "charts": [
    {
      "id": "equity",                 // unique within the file
      "title": "Equity curve",
      "type": "line",                 // "line" | "bar" | "scatter"
      "x_label": "Date",
      "y_label": "Equity ($)",
      "x": ["2020-01-01", "..."],     // strings or numbers; shared by all series
      "series": [
        {"name": "Strategy", "data": [100000.0, 100123.4]},
        {"name": "Benchmark", "data": [100000.0, 100050.0]}
      ]
    }
  ],
  "tables": [
    {
      "id": "top_factors",
      "title": "Top factors by IC",
      "columns": ["Factor", "IC", "Rank IC"],
      "rows": [["alpha_006", 0.042, 0.051]]
    }
  ],
  "notes": "Free-form text shown under the product section (plain text, short)."
}
```

Rules:

- Keep each chart to <= 500 x-points (downsample longer series) and <= 5 series.
- Numbers must be JSON-serializable floats/ints (convert NumPy types; no NaN —
  use `null`).
- `charts` and `tables` may be empty lists, `summary_metrics` must not be.
- Write with `json.dump(..., indent=2)` so diffs stay readable.
