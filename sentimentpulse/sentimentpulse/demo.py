"""SentimentPulse demo: synthetic news -> scoring -> indexes -> event study.

Writes hub results JSON to ``<repo>/hub/data/sentimentpulse.json`` per
hub/SCHEMA.md. Run with: ``cd sentimentpulse && python3 -m sentimentpulse.demo``
"""

from __future__ import annotations

import json
import math
import time
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from sentimentpulse import __version__
from sentimentpulse.aggregate import (
    daily_scores,
    detect_shocks,
    scores_frame,
    sentiment_breadth,
    sentiment_index,
)
from sentimentpulse.evaluation import event_study, scorer_accuracy
from sentimentpulse.lexicon import lexicon_summary
from sentimentpulse.news import deduplicate
from sentimentpulse.providers import LexiconProvider, MockLLMProvider
from sentimentpulse.synthetic import generate_news, generate_prices, generate_universe

HUB_DATA_DIR = Path(__file__).resolve().parents[2] / "hub" / "data"

N_SYMBOLS = 20
N_ITEMS = 2000
N_DAYS = 504  # ~2 years of business days
START = "2024-01-01"
SEED = 21
HORIZONS = (1, 3, 5)


def _clean(value: float) -> float | None:
    f = float(value)
    return None if (math.isnan(f) or math.isinf(f)) else round(f, 6)


def _downsample_idx(n: int, max_points: int = 500) -> list[int]:
    if n <= max_points:
        return list(range(n))
    step = math.ceil(n / max_points)
    idx = list(range(0, n, step))
    if idx[-1] != n - 1:
        idx.append(n - 1)
    return idx


def run_demo() -> dict:
    """Run the full pipeline and return the hub results payload."""
    t0 = time.time()

    # 1. Synthetic universe, news (with ground truth) and sentiment-driven prices.
    symbols, alias_map = generate_universe(N_SYMBOLS)
    items, truths = generate_news(
        symbols, alias_map, n_items=N_ITEMS, start=START, n_days=N_DAYS, seed=SEED
    )
    truth_by_id = {item.id: t for item, t in zip(items, truths, strict=True)}
    items = deduplicate(items)
    truths = [truth_by_id[item.id] for item in items]
    dates = pd.bdate_range(START, periods=N_DAYS)
    close = generate_prices(symbols, dates, items, truths, seed=SEED + 1)

    # 2. Score with the real lexicon provider and the deterministic mock LLM.
    lexicon = LexiconProvider()
    mock = MockLLMProvider()
    texts = [item.text for item in items]
    lex_results = lexicon.score_batch(texts)
    mock_results = mock.score_batch(texts)

    lex_acc = scorer_accuracy([r.score for r in lex_results], truths)
    mock_acc = scorer_accuracy([r.score for r in mock_results], truths)

    # 3. Aggregate: daily index, breadth, shocks (lexicon scores).
    frame = scores_frame(items, lex_results)
    daily = daily_scores(frame, dates=dates)
    index = sentiment_index(daily, halflife=5.0)
    breadth = sentiment_breadth(index)
    shocks = detect_shocks(daily, window=20, z_threshold=2.0)

    # 4. Event study: forward returns after positive vs negative shocks.
    study = event_study(shocks, close, horizons=HORIZONS)
    spread_3d = study.at["positive", "fwd_3d"] - study.at["negative", "fwd_3d"]

    # 5. Index-vs-price chart for the most-covered symbol.
    focus = frame["symbol"].value_counts().idxmax()
    focus_index = index[focus]
    focus_price = close[focus] / close[focus].iloc[0]
    keep = _downsample_idx(len(dates))
    chart_dates = [dates[i].strftime("%Y-%m-%d") for i in keep]

    elapsed = time.time() - t0
    lex_words = lexicon_summary()

    payload = {
        "schema_version": 1,
        "product": "sentimentpulse",
        "title": "SentimentPulse",
        "tagline": "Sentiment analysis for financial text",
        "version": __version__,
        "generated_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "status": "ok",
        "summary_metrics": [
            {"label": "News items scored", "value": str(len(items))},
            {"label": "Symbols", "value": str(N_SYMBOLS)},
            {"label": "Lexicon accuracy", "value": f"{100 * lex_acc['accuracy']:.1f}", "unit": "%"},
            {"label": "Mock-LLM accuracy", "value": f"{100 * mock_acc['accuracy']:.1f}", "unit": "%"},
            {"label": "Sentiment shocks", "value": str(len(shocks))},
            {"label": "Shock spread (+3d, pos-neg)", "value": f"{100 * spread_3d:.2f}", "unit": "%"},
            {"label": "Lexicon size", "value": str(sum(lex_words[k] for k in ("positive", "negative", "uncertainty", "litigious")))},
            {"label": "Demo runtime", "value": f"{elapsed:.1f}", "unit": "s"},
        ],
        "charts": [
            {
                "id": "index_vs_price",
                "title": f"Sentiment index vs price — {focus}",
                "type": "line",
                "x_label": "Date",
                "y_label": "Index / normalized price",
                "x": chart_dates,
                "series": [
                    {
                        "name": "Sentiment index (EWM, halflife 5d)",
                        "data": [_clean(focus_index.iloc[i]) for i in keep],
                    },
                    {
                        "name": "Price (normalized)",
                        "data": [_clean(focus_price.iloc[i]) for i in keep],
                    },
                ],
            },
            {
                "id": "shock_event_study",
                "title": "Avg forward return after sentiment shocks",
                "type": "bar",
                "x_label": "Horizon",
                "y_label": "Mean forward return",
                "x": [f"+{h}d" for h in HORIZONS],
                "series": [
                    {
                        "name": "Positive shocks",
                        "data": [_clean(study.at["positive", f"fwd_{h}d"]) for h in HORIZONS],
                    },
                    {
                        "name": "Negative shocks",
                        "data": [_clean(study.at["negative", f"fwd_{h}d"]) for h in HORIZONS],
                    },
                ],
            },
            {
                "id": "breadth",
                "title": "Universe sentiment breadth (share of symbols positive)",
                "type": "line",
                "x_label": "Date",
                "y_label": "Breadth",
                "x": chart_dates,
                "series": [
                    {"name": "Breadth", "data": [_clean(breadth.iloc[i]) for i in keep]}
                ],
            },
        ],
        "tables": [
            {
                "id": "provider_comparison",
                "title": "Provider comparison on labeled synthetic news",
                "columns": [
                    "Provider", "Accuracy", "Pos recall", "Neg recall",
                    "Neutral recall", "Deterministic", "Notes",
                ],
                "rows": [
                    [
                        "lexicon",
                        _clean(lex_acc["accuracy"]),
                        _clean(lex_acc["per_class"]["positive"]),
                        _clean(lex_acc["per_class"]["negative"]),
                        _clean(lex_acc["per_class"]["neutral"]),
                        "yes",
                        "Bundled LM-style lexicon with negation/intensifiers",
                    ],
                    [
                        "mock_llm",
                        _clean(mock_acc["accuracy"]),
                        _clean(mock_acc["per_class"]["positive"]),
                        _clean(mock_acc["per_class"]["negative"]),
                        _clean(mock_acc["per_class"]["neutral"]),
                        "yes",
                        "Hash-based TEST DOUBLE (no real analysis; ~chance)",
                    ],
                ],
            },
            {
                "id": "event_study",
                "title": "Event study: forward returns after sentiment shocks",
                "columns": ["Direction", "N events"] + [f"Fwd {h}d" for h in HORIZONS],
                "rows": [
                    [
                        str(direction),
                        int(study.at[direction, "n_events"]),
                        *[_clean(study.at[direction, f"fwd_{h}d"]) for h in HORIZONS],
                    ]
                    for direction in ("positive", "negative")
                ],
            },
        ],
        "notes": (
            f"Fully offline: {N_ITEMS} seeded synthetic news items over {N_DAYS} business days "
            f"for {N_SYMBOLS} symbols, with known ground-truth polarity; prices are built so prior-day "
            "news polarity moves next-day returns, and the shock event study recovers that injected "
            "effect. FinBERT/FinGPT/GPT-4 from the design doc are substituted with a bundled "
            "Loughran-McDonald-style lexicon scorer plus a deterministic mock LLM provider that only "
            "exercises the provider interface."
        ),
    }
    return payload


def main() -> None:
    payload = run_demo()
    HUB_DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = HUB_DATA_DIR / "sentimentpulse.json"
    with out_path.open("w") as fh:
        json.dump(payload, fh, indent=2)
    metrics = {m["label"]: m["value"] for m in payload["summary_metrics"]}
    print(f"SentimentPulse demo complete -> {out_path}")
    print(
        f"  items={metrics['News items scored']}  lexicon_acc={metrics['Lexicon accuracy']}%  "
        f"shocks={metrics['Sentiment shocks']}"
    )


if __name__ == "__main__":
    main()
