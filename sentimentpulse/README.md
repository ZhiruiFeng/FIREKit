# SentimentPulse

Sentiment analysis for financial text — the offline MVP of the FIREKit
SentimentPulse design (`docs/products/05_sentimentpulse.md`).

## What it does

- **Bundled financial lexicon** (`sentimentpulse.lexicon`): a curated
  Loughran-McDonald-style word list written for this package (~470 words across
  positive / negative / uncertainty / litigious), plus negators, intensifiers
  and diminishers.
- **Scorer** (`sentimentpulse.scorer`): tokenizer + rule-based scorer with
  negation handling ("not good" flips polarity), intensifier/diminisher
  weighting, and a modifier window that stops at the previous sentiment word.
  Outputs per-document polarity in [-1, 1] plus uncertainty and litigious
  scores in [0, 1]. Scoring rules are exact and unit-tested on hand-built
  sentences.
- **Provider abstraction** (`sentimentpulse.providers`): `SentimentProvider`
  ABC with two implementations:
  - `LexiconProvider` — the real offline scorer.
  - `MockLLMProvider` — a **deterministic, hash-based test double** that
    simulates the shape of an LLM provider response (SHA-256 of the text; no
    real analysis, ~chance accuracy). It exists to exercise the interface and
    downstream pipeline offline; a real Claude/GPT backend would implement the
    same ABC.
- **News pipeline** (`sentimentpulse.news`): pydantic `NewsItem`
  (timestamp, symbol, headline, body), entity-to-symbol tagging via an alias
  map with word-boundary matching, and time-windowed deduplication.
- **Aggregation** (`sentimentpulse.aggregate`): per-symbol daily sentiment
  index (exponential decay weighting), sentiment momentum, shock detection
  (z-score spikes vs trailing history), and universe sentiment breadth.
- **Synthetic data** (`sentimentpulse.synthetic`): seeded news generator with
  template headlines of *known ground-truth polarity* (including adversarial
  "hard" templates), and synthetic prices where prior-day news polarity moves
  next-day returns — so the pipeline's ability to recover the injected effect
  is measurable.
- **Evaluation** (`sentimentpulse.evaluation`): three-way scorer accuracy with
  confusion matrix, and an event study of forward returns after positive vs
  negative sentiment shocks.

## Substitutions vs the design doc

No transformers/torch, no network, no API keys in this environment, so:

| Design doc | This MVP |
|---|---|
| FinBERT classification | `LexiconScorer` (bundled financial lexicon, negation-aware) |
| FinGPT / GPT-4 / Claude | `MockLLMProvider` (deterministic hash-based mock) behind the `SentimentProvider` ABC |
| News APIs (Benzinga/Polygon) | seeded synthetic news generator with ground truth |
| Model router | provider abstraction (routing trivially pluggable) |

## Quick start

```python
from sentimentpulse import (
    LexiconProvider, daily_scores, detect_shocks, event_study,
    generate_news, generate_prices, generate_universe, scorer_accuracy,
    scores_frame, sentiment_index,
)
import pandas as pd

symbols, alias_map = generate_universe(20)
items, truths = generate_news(symbols, alias_map, n_items=2000)
provider = LexiconProvider()
results = provider.score_batch([item.text for item in items])
print(scorer_accuracy([r.score for r in results], truths)["accuracy"])

dates = pd.bdate_range("2024-01-01", periods=504)
close = generate_prices(symbols, dates, items, truths)
daily = daily_scores(scores_frame(items, results), dates=dates)
index = sentiment_index(daily, halflife=5.0)
shocks = detect_shocks(daily)
print(event_study(shocks, close))
```

## Demo

```bash
cd sentimentpulse && python3 -m sentimentpulse.demo
```

Generates ~2000 synthetic news items over 2 years for 20 symbols, scores them
with both providers, and writes `hub/data/sentimentpulse.json`
(per `hub/SCHEMA.md`): accuracy metrics, sentiment-index-vs-price chart, shock
event-study bar chart, and a lexicon-vs-mock provider comparison table.
Deterministic, offline, < 5 s.

## Tests

```bash
cd sentimentpulse && python3 -m pytest tests -q
```
