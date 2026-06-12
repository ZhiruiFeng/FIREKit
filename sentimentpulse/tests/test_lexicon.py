"""Tests for the bundled lexicon."""

from __future__ import annotations

from sentimentpulse.lexicon import (
    DIMINISHERS,
    INTENSIFIERS,
    LITIGIOUS,
    NEGATIVE,
    NEGATORS,
    POSITIVE,
    UNCERTAINTY,
    lexicon_summary,
)


def test_positive_negative_disjoint():
    assert not POSITIVE & NEGATIVE


def test_modifier_roles_unambiguous():
    """A token must map to exactly one modifier role for the scorer."""
    assert not NEGATORS & INTENSIFIERS
    assert not NEGATORS & DIMINISHERS
    assert not INTENSIFIERS & DIMINISHERS


def test_minimum_sizes():
    counts = lexicon_summary()
    assert counts["positive"] >= 100
    assert counts["negative"] >= 120
    assert counts["uncertainty"] >= 60
    assert counts["litigious"] >= 50
    core = counts["positive"] + counts["negative"] + counts["uncertainty"] + counts["litigious"]
    assert core >= 300


def test_all_entries_lowercase_single_tokens():
    for words in (POSITIVE, NEGATIVE, UNCERTAINTY, LITIGIOUS, NEGATORS, INTENSIFIERS, DIMINISHERS):
        for w in words:
            assert w == w.lower()
            assert " " not in w


def test_key_financial_terms_present():
    assert {"profit", "growth", "beat", "upgrade"} <= POSITIVE
    assert {"loss", "decline", "miss", "downgrade", "bankruptcy"} <= NEGATIVE
    assert {"may", "uncertain", "approximately"} <= UNCERTAINTY
    assert {"lawsuit", "litigation", "investigation"} <= LITIGIOUS
