"""Tests for the provider abstraction."""

from __future__ import annotations

import pytest

from sentimentpulse.providers import LexiconProvider, MockLLMProvider, SentimentProvider
from sentimentpulse.scorer import LexiconScorer


def test_abc_cannot_be_instantiated():
    with pytest.raises(TypeError):
        SentimentProvider()  # type: ignore[abstract]


def test_lexicon_provider_matches_scorer():
    text = "record profit beats expectations despite lawsuit risk"
    provider = LexiconProvider()
    direct = LexiconScorer().score_text(text)
    via_provider = provider.score_text(text)
    assert via_provider.score == direct.score
    assert via_provider.uncertainty == direct.uncertainty
    assert via_provider.provider == "lexicon"


def test_providers_are_sentiment_providers():
    assert isinstance(LexiconProvider(), SentimentProvider)
    assert isinstance(MockLLMProvider(), SentimentProvider)


def test_mock_is_deterministic():
    a = MockLLMProvider().score_text("Acme beats estimates")
    b = MockLLMProvider().score_text("Acme beats estimates")
    assert a.score == b.score
    assert a.uncertainty == b.uncertainty


def test_mock_varies_with_text_and_model_name():
    p = MockLLMProvider()
    assert p.score_text("text one").score != p.score_text("text two").score
    other = MockLLMProvider(model_name="mock-llm-2")
    assert p.score_text("same text").score != other.score_text("same text").score


def test_mock_outputs_in_bounds():
    p = MockLLMProvider()
    for i in range(200):
        res = p.score_text(f"headline number {i}")
        assert -1.0 <= res.score <= 1.0
        assert 0.0 <= res.uncertainty <= 1.0
        assert 0.0 <= res.litigious <= 1.0


def test_mock_is_flagged_as_mock():
    res = MockLLMProvider().score_text("anything")
    assert res.meta["mock"] is True
    assert res.provider.startswith("mock_llm")


def test_score_batch_preserves_order():
    texts = ["strong profit", "heavy loss", "board meeting"]
    for provider in (LexiconProvider(), MockLLMProvider()):
        batch = provider.score_batch(texts)
        single = [provider.score_text(t) for t in texts]
        assert [r.score for r in batch] == [r.score for r in single]


def test_custom_provider_can_plug_in():
    class ConstantProvider(SentimentProvider):
        name = "constant"

        def score_text(self, text: str):
            from sentimentpulse.scorer import SentimentResult

            return SentimentResult(score=0.5, uncertainty=0.0, litigious=0.0, provider=self.name)

    results = ConstantProvider().score_batch(["a", "b"])
    assert [r.score for r in results] == [0.5, 0.5]
