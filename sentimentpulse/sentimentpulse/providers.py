"""Provider abstraction: pluggable sentiment backends.

The design doc routes between FinBERT / FinGPT / GPT-4 / Claude. This offline
MVP ships:

- :class:`LexiconProvider` — the real scorer (bundled financial lexicon).
- :class:`MockLLMProvider` — a **deterministic, hash-based mock** that stands
  in for a remote LLM provider. It performs *no real analysis*: scores are
  derived from a SHA-256 digest of the text so the provider interface,
  batching, and downstream aggregation can be exercised offline and
  reproducibly. Expect ~chance accuracy; never use it for real signals.

A real LLM backend (e.g. Claude) would implement the same
:class:`SentimentProvider` interface.
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod

from sentimentpulse.scorer import LexiconScorer, SentimentResult, tokenize


class SentimentProvider(ABC):
    """Abstract sentiment backend: text in, SentimentResult out."""

    name: str = "provider"

    @abstractmethod
    def score_text(self, text: str) -> SentimentResult:
        """Score a single document."""

    def score_batch(self, texts: list[str]) -> list[SentimentResult]:
        """Score many documents, preserving order."""
        return [self.score_text(t) for t in texts]


class LexiconProvider(SentimentProvider):
    """Production offline provider backed by the bundled lexicon scorer."""

    name = "lexicon"

    def __init__(self, negation_window: int = 3) -> None:
        self._scorer = LexiconScorer(negation_window=negation_window)

    def score_text(self, text: str) -> SentimentResult:
        return self._scorer.score_text(text)


class MockLLMProvider(SentimentProvider):
    """Deterministic hash-based MOCK of an LLM sentiment provider.

    THIS IS A TEST DOUBLE. It simulates the latency-free, deterministic shape
    of an LLM provider response so the SentimentProvider interface can be
    integration-tested offline. The "sentiment" is a pure function of
    SHA-256(model_name + text) and carries no semantic information.
    """

    def __init__(self, model_name: str = "mock-llm-1") -> None:
        self.model_name = model_name
        self.name = f"mock_llm({model_name})"

    def _digest(self, text: str) -> int:
        payload = f"{self.model_name}::{text}".encode()
        return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")

    def score_text(self, text: str) -> SentimentResult:
        h = self._digest(text)
        # score uniform on [-1, 1] in 0.001 steps; uncertainty on [0, 1]
        score = ((h % 2001) - 1000) / 1000.0
        uncertainty = ((h >> 16) % 1001) / 1000.0
        litigious = ((h >> 32) % 1001) / 1000.0
        return SentimentResult(
            score=score,
            uncertainty=uncertainty,
            litigious=litigious,
            n_tokens=len(tokenize(text)),
            provider=self.name,
            meta={"mock": True, "model_name": self.model_name},
        )
