"""Tokenizer and lexicon-based sentiment scorer.

Scoring rules (deterministic, hand-testable):

- Each positive/negative lexicon hit contributes a base weight of 1.0.
- A *negator* within the preceding ``negation_window`` tokens flips the hit's
  polarity ("not good" counts as negative). Multiple negators toggle parity.
- An *intensifier* in the window multiplies the weight by 1.5; a *diminisher*
  multiplies it by 0.5 (both can stack with negation: "not very good").
- ``score = (pos_weight - neg_weight) / (pos_weight + neg_weight)`` in
  [-1, 1]; 0.0 when no sentiment words are found.
- ``uncertainty = min(1, 0.25 * uncertainty_hits)``; ``litigious`` likewise.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from sentimentpulse.lexicon import (
    DIMINISHERS,
    INTENSIFIERS,
    LITIGIOUS,
    NEGATIVE,
    NEGATORS,
    POSITIVE,
    UNCERTAINTY,
)

_TOKEN_RE = re.compile(r"[a-z]+(?:'[a-z]+)?")

INTENSIFIER_FACTOR = 1.5
DIMINISHER_FACTOR = 0.5
CATEGORY_HIT_STEP = 0.25  # each uncertainty/litigious hit adds this, capped at 1


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens (letters plus internal apostrophes)."""
    return _TOKEN_RE.findall(text.lower())


@dataclass(frozen=True)
class SentimentResult:
    """Per-document sentiment output."""

    score: float  # polarity in [-1, 1]
    uncertainty: float  # [0, 1]
    litigious: float  # [0, 1]
    positive_hits: tuple[str, ...] = ()
    negative_hits: tuple[str, ...] = ()
    n_tokens: int = 0
    provider: str = "lexicon"
    meta: dict = field(default_factory=dict)

    @property
    def label(self) -> str:
        """Three-way label using the default +/-0.15 thresholds."""
        return classify(self.score)


def classify(score: float, pos_threshold: float = 0.15, neg_threshold: float = -0.15) -> str:
    """Map a polarity score to 'positive' / 'negative' / 'neutral'."""
    if score > pos_threshold:
        return "positive"
    if score < neg_threshold:
        return "negative"
    return "neutral"


class LexiconScorer:
    """Rule-based scorer over the bundled financial lexicon."""

    def __init__(self, negation_window: int = 3) -> None:
        if negation_window < 0:
            raise ValueError("negation_window must be >= 0")
        self.negation_window = negation_window

    def _hit_weight(self, tokens: list[str], i: int) -> float:
        """Signed weight of the sentiment word at position i (sign = flip).

        Scans backwards through the preceding window, stopping at the nearest
        earlier sentiment word so a modifier only attaches to the sentiment
        word that follows it ("very good but weak" intensifies only "good").
        """
        weight = 1.0
        flipped = False
        for j in range(i - 1, max(0, i - self.negation_window) - 1, -1):
            tok = tokens[j]
            if tok in POSITIVE or tok in NEGATIVE:
                break
            if tok in NEGATORS:
                flipped = not flipped
            elif tok in INTENSIFIERS:
                weight *= INTENSIFIER_FACTOR
            elif tok in DIMINISHERS:
                weight *= DIMINISHER_FACTOR
        return -weight if flipped else weight

    def score_text(self, text: str) -> SentimentResult:
        """Score a document; returns a SentimentResult with score in [-1, 1]."""
        tokens = tokenize(text)
        pos_weight = 0.0
        neg_weight = 0.0
        pos_hits: list[str] = []
        neg_hits: list[str] = []
        unc_hits = 0
        lit_hits = 0

        for i, tok in enumerate(tokens):
            if tok in UNCERTAINTY:
                unc_hits += 1
            if tok in LITIGIOUS:
                lit_hits += 1
            is_pos = tok in POSITIVE
            is_neg = tok in NEGATIVE
            if not (is_pos or is_neg):
                continue
            signed = self._hit_weight(tokens, i)
            # a flip converts the hit to the opposite polarity bucket
            if (is_pos and signed > 0) or (is_neg and signed < 0):
                pos_weight += abs(signed)
                pos_hits.append(tok)
            else:
                neg_weight += abs(signed)
                neg_hits.append(tok)

        total = pos_weight + neg_weight
        score = (pos_weight - neg_weight) / total if total > 0 else 0.0
        return SentimentResult(
            score=max(-1.0, min(1.0, score)),
            uncertainty=min(1.0, CATEGORY_HIT_STEP * unc_hits),
            litigious=min(1.0, CATEGORY_HIT_STEP * lit_hits),
            positive_hits=tuple(pos_hits),
            negative_hits=tuple(neg_hits),
            n_tokens=len(tokens),
            provider="lexicon",
        )
