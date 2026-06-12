"""Exact-value tests for the tokenizer and lexicon scorer."""

from __future__ import annotations

import pytest

from sentimentpulse.scorer import LexiconScorer, classify, tokenize


@pytest.fixture()
def scorer() -> LexiconScorer:
    return LexiconScorer()


def test_tokenize_lowercases_and_strips_punctuation():
    assert tokenize("Profit SOARS, beats-estimates!") == ["profit", "soars", "beats", "estimates"]


def test_tokenize_keeps_internal_apostrophes():
    assert tokenize("Doesn't grow") == ["doesn't", "grow"]


def test_single_positive_word_scores_one(scorer):
    res = scorer.score_text("profit")
    assert res.score == 1.0
    assert res.label == "positive"
    assert res.positive_hits == ("profit",)


def test_single_negative_word_scores_minus_one(scorer):
    res = scorer.score_text("loss")
    assert res.score == -1.0
    assert res.label == "negative"
    assert res.negative_hits == ("loss",)


def test_negation_flips_positive(scorer):
    assert scorer.score_text("not good").score == -1.0


def test_negation_flips_negative(scorer):
    assert scorer.score_text("not bad").score == 1.0


def test_double_negation_restores_polarity(scorer):
    assert scorer.score_text("never not profitable").score == 1.0


def test_negator_outside_window_does_not_flip(scorer):
    # default window = 3 preceding tokens; negator is 4 tokens back
    assert scorer.score_text("not the very same old good").score == 1.0
    # within window it flips
    assert scorer.score_text("not the same good").score == -1.0


def test_intensifier_weighting_exact(scorer):
    # "very good but weak": pos = 1.5, neg = 1.0 -> (0.5)/(2.5) = 0.2
    assert scorer.score_text("very good but weak").score == pytest.approx(0.2)


def test_diminisher_weighting_exact(scorer):
    # "slightly weak results but good": neg = 0.5, pos = 1.0 -> 0.5/1.5
    assert scorer.score_text("slightly weak but good").score == pytest.approx(1 / 3)


def test_negated_intensified_word(scorer):
    # "not very good": flipped and intensified -> neg bucket 1.5;
    # "strong" (outside the modifier window) -> pos 1.0; score = -0.5/2.5
    assert scorer.score_text("not very good but otherwise strong").score == pytest.approx(-0.2)


def test_neutral_text_scores_zero(scorer):
    res = scorer.score_text("the company held a meeting on tuesday")
    assert res.score == 0.0
    assert res.label == "neutral"


def test_empty_text(scorer):
    res = scorer.score_text("")
    assert res.score == 0.0
    assert res.n_tokens == 0


def test_uncertainty_exact(scorer):
    # two uncertainty hits ("may", "perhaps") -> 0.25 * 2 = 0.5
    res = scorer.score_text("results may improve perhaps")
    assert res.uncertainty == pytest.approx(0.5)


def test_uncertainty_caps_at_one(scorer):
    res = scorer.score_text("may might could possibly perhaps maybe unclear")
    assert res.uncertainty == 1.0


def test_litigious_exact(scorer):
    res = scorer.score_text("lawsuit settlement reached")
    assert res.litigious == pytest.approx(0.5)


def test_balanced_text_scores_zero(scorer):
    assert scorer.score_text("strong profit but heavy loss and decline").score == pytest.approx(
        (2 - 2) / 4
    )


def test_score_always_in_bounds(scorer):
    texts = [
        "very very extremely good great strong",
        "not not bad terrible loss decline crash",
        "profit loss profit loss",
    ]
    for t in texts:
        assert -1.0 <= scorer.score_text(t).score <= 1.0


def test_classify_thresholds():
    assert classify(0.2) == "positive"
    assert classify(-0.2) == "negative"
    assert classify(0.1) == "neutral"
    assert classify(0.15) == "neutral"  # boundary is exclusive
    assert classify(0.05, pos_threshold=0.0) == "positive"


def test_invalid_negation_window():
    with pytest.raises(ValueError):
        LexiconScorer(negation_window=-1)


def test_realistic_headlines(scorer):
    pos = scorer.score_text("Acme reports record quarterly profit and beats expectations")
    neg = scorer.score_text("Acme shares plunge on disappointing quarterly loss")
    assert pos.label == "positive"
    assert neg.label == "negative"
