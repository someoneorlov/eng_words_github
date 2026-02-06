"""Tests for Pipeline B headword (Stage 3): infer_headword, resolve_headword, QC headword_in_example."""

from __future__ import annotations

from eng_words.word_family.batch_qc import (
    cards_lemma_not_in_example,
    get_cards_failing_headword_invalid_for_mode,
)
from eng_words.word_family.headword import (
    infer_headword,
    is_single_word,
    resolve_headword,
)
from eng_words.word_family.qc_types import ErrorType


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


# --- resolve_headword (Stage 3 contract) ---


def test_resolve_headword_look_up_in_examples_word_mode_returns_error() -> None:
    """Word mode: multiword headword 'look up' (even if in examples) → invalid for mode."""
    card = {
        "lemma": "look",
        "headword": "look up",
        "examples": ["I look up the word."],
    }
    hw, finding = resolve_headword(card, mode="word")
    assert hw is None
    assert finding is not None
    assert finding.error_type == ErrorType.QC_HEADWORD_INVALID_FOR_MODE


def test_resolve_headword_look_up_in_examples_phrasal_mode_accepts() -> None:
    """Phrasal mode: multiword headword in examples → accepted."""
    card = {
        "lemma": "look",
        "headword": "look up",
        "examples": ["I look up the word."],
    }
    hw, finding = resolve_headword(card, mode="phrasal")
    assert hw == "look up"
    assert finding is None


def test_resolve_headword_made_up_returns_not_in_examples() -> None:
    """Made-up headword not in examples → QC_HEADWORD_NOT_IN_EXAMPLES."""
    card = {
        "lemma": "look",
        "headword": "look sideways",
        "examples": ["I look up the word."],
    }
    hw, finding = resolve_headword(card, mode="word")
    assert hw is None
    assert finding is not None
    assert finding.error_type == ErrorType.QC_HEADWORD_NOT_IN_EXAMPLES


def test_resolve_headword_single_word_word_mode_accepts() -> None:
    """Word mode: single-word headword in examples (whole-word match) → accepted."""
    card = {
        "lemma": "run",
        "headword": "run",
        "examples": ["I run daily.", "They run fast."],
    }
    hw, finding = resolve_headword(card, mode="word")
    assert hw == "run"
    assert finding is None


def test_resolve_headword_no_headword_returns_none_none() -> None:
    card = {"lemma": "x", "examples": ["x"]}
    hw, finding = resolve_headword(card, mode="word")
    assert hw is None
    assert finding is None


def test_is_single_word() -> None:
    assert is_single_word("run") is True
    assert is_single_word("look up") is False
    assert is_single_word("well-known") is True
    assert is_single_word("") is False
    assert is_single_word("  run  ") is True


def test_get_cards_failing_headword_invalid_for_mode_word_mode() -> None:
    """Word mode: cards with multiword headword are returned for drop."""
    cards = [
        {"lemma": "run", "headword": "run", "examples": ["He runs."]},
        {"lemma": "look", "headword": "look up", "examples": ["I look up the word."]},
    ]
    failing = get_cards_failing_headword_invalid_for_mode(cards, mode="word")
    assert len(failing) == 1
    assert failing[0]["headword"] == "look up"


def test_get_cards_failing_headword_invalid_for_mode_phrasal_returns_empty() -> None:
    cards = [{"lemma": "look", "headword": "look up", "examples": ["I look up."]}]
    failing = get_cards_failing_headword_invalid_for_mode(cards, mode="phrasal")
    assert len(failing) == 0
