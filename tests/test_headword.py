"""Tests for Pipeline B headword (Stage 3): infer_headword, QC headword_in_example."""

from __future__ import annotations

from eng_words.word_family.batch_qc import cards_lemma_not_in_example
from eng_words.word_family.headword import infer_headword


def test_infer_headword_look_up_when_in_examples() -> None:
    """'look up' accepted as headword only when it appears in examples."""
    card = {
        "lemma": "look",
        "headword": "look up",
        "examples": ["I look up the word in a dictionary."],
        "definition_en": "search for",
        "definition_ru": "искать",
    }
    assert infer_headword(card) == "look up"


def test_infer_headword_made_up_rejected() -> None:
    """Made-up headword not in examples is rejected."""
    card = {
        "lemma": "look",
        "headword": "look sideways",
        "examples": ["I look up the word."],
        "definition_en": "search",
        "definition_ru": "искать",
    }
    assert infer_headword(card) is None


def test_infer_headword_empty_or_missing() -> None:
    assert infer_headword({"lemma": "x", "examples": ["x"]}) is None
    assert infer_headword({"lemma": "x", "headword": "", "examples": ["x"]}) is None
    assert infer_headword({"lemma": "x", "headword": "   ", "examples": ["x"]}) is None


def test_infer_headword_must_be_in_every_example() -> None:
    card = {
        "lemma": "look",
        "headword": "look up",
        "examples": ["I look up the word.", "I look at the sky."],
        "definition_en": "search",
        "definition_ru": "искать",
    }
    assert infer_headword(card) is None


def test_qc_headword_in_example_passes() -> None:
    """When headword is present and in every example (as substring), card is not flagged."""
    cards = [
        {
            "lemma": "look",
            "headword": "look up",
            "examples": ["I look up the word.", "We look up the address."],
            "definition_en": "search for",
        }
    ]
    out = cards_lemma_not_in_example(cards)
    assert len(out) == 0


def test_qc_headword_not_in_example_flagged() -> None:
    """When headword is present but not in an example, card is flagged."""
    cards = [
        {
            "lemma": "look",
            "headword": "look up",
            "examples": ["I look at the sky."],
            "definition_en": "search for",
        }
    ]
    out = cards_lemma_not_in_example(cards)
    assert len(out) == 1
    assert out[0]["lemma"] == "look"
