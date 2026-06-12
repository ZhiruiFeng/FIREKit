"""SentimentPulse: financial text sentiment analysis.

Offline MVP of the FIREKit SentimentPulse design: bundled financial lexicon,
negation-aware scorer, pluggable provider abstraction (lexicon + deterministic
mock-LLM test double), news pipeline, daily sentiment indexes, shock detection,
and a measurable synthetic evaluation harness.
"""

from sentimentpulse.aggregate import (
    daily_scores,
    detect_shocks,
    scores_frame,
    sentiment_breadth,
    sentiment_index,
    sentiment_momentum,
)
from sentimentpulse.evaluation import event_study, forward_returns, scorer_accuracy
from sentimentpulse.lexicon import (
    LITIGIOUS,
    NEGATIVE,
    POSITIVE,
    UNCERTAINTY,
    lexicon_summary,
)
from sentimentpulse.news import AliasMap, NewsItem, deduplicate, tag_items, tag_symbols
from sentimentpulse.providers import LexiconProvider, MockLLMProvider, SentimentProvider
from sentimentpulse.scorer import LexiconScorer, SentimentResult, classify, tokenize
from sentimentpulse.synthetic import (
    generate_news,
    generate_prices,
    generate_universe,
)

__version__ = "0.1.0"

__all__ = [
    "LITIGIOUS",
    "NEGATIVE",
    "POSITIVE",
    "UNCERTAINTY",
    "AliasMap",
    "LexiconProvider",
    "LexiconScorer",
    "MockLLMProvider",
    "NewsItem",
    "SentimentProvider",
    "SentimentResult",
    "__version__",
    "classify",
    "daily_scores",
    "deduplicate",
    "detect_shocks",
    "event_study",
    "forward_returns",
    "generate_news",
    "generate_prices",
    "generate_universe",
    "lexicon_summary",
    "scorer_accuracy",
    "scores_frame",
    "sentiment_breadth",
    "sentiment_index",
    "sentiment_momentum",
    "tag_items",
    "tag_symbols",
    "tokenize",
]
