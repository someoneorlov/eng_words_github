"""Tests for eng_words.text_norm (Stage 2: normalize + contractions for matching)."""

from __future__ import annotations

import pytest

from eng_words.text_norm import (
    expand_contractions_for_matching,
    match_target_in_text,
    normalize_for_matching,
    word_in_text_for_matching,
)


# --- normalize_for_matching ---


def test_normalize_for_matching_unicode_nfkc() -> None:
    # NFKC: compatibility chars normalized
    s = "café"
    out = normalize_for_matching(s)
    assert "e" in out or "\u0301" not in out  # decomposed/compressed


def test_normalize_for_matching_apostrophes() -> None:
    # Typographic apostrophe → ASCII
    assert "'" in normalize_for_matching("don't")
    assert "'" in normalize_for_matching("it\u2019s")
    assert normalize_for_matching("we're") == normalize_for_matching("we're")


def test_normalize_for_matching_dashes() -> None:
    # Em dash, en dash → hyphen (deterministic)
    out = normalize_for_matching("word—word")
    assert "—" not in out
    out2 = normalize_for_matching("word–word")
    assert "–" not in out2


def test_normalize_for_matching_whitespace() -> None:
    # Multiple spaces → single, strip
    assert normalize_for_matching("  a   b  ") == "a b"
    assert normalize_for_matching("\ta\t") == "a"


def test_normalize_for_matching_empty_and_idempotent() -> None:
    assert normalize_for_matching("") == ""
    s = "  It's   fine—really.  "
    assert normalize_for_matching(normalize_for_matching(s)) == normalize_for_matching(s)


# Golden: input -> expected normalized (deterministic)
@pytest.mark.parametrize(
    "input_text,expected_contains",
    [
        ("don't", "'"),
        ("It\u2019s", "'"),
        ("word—word", "-"),
        ("  spaces  ", "spaces"),
    ],
)
def test_normalize_for_matching_golden(input_text: str, expected_contains: str) -> None:
    out = normalize_for_matching(input_text)
    assert expected_contains in out


@pytest.mark.parametrize(
    "input_text,expected",
    [
        ("  a   b  ", "a b"),
        ("word—word", "word-word"),
        ("word–word", "word-word"),
        ("don't", "don't"),
        ("\u2019x\u2018", "'x'"),
        ("a\u00a0\u00a0b", "a b"),
    ],
)
def test_normalize_for_matching_golden_exact(input_text: str, expected: str) -> None:
    """Exact golden: input -> expected string (PIPELINE_B_FIXES_PLAN 2.1)."""
    assert normalize_for_matching(input_text) == expected


def test_normalize_for_matching_determinism() -> None:
    """Same input -> same output every time (no nondeterministic logic)."""
    inputs = ["  It\u2019s   fine—really.  ", "don't", "a\u00a0b", ""]
    for s in inputs:
        out_first = normalize_for_matching(s)
        for _ in range(10):
            assert normalize_for_matching(s) == out_first


def test_normalize_for_matching_non_breaking_space() -> None:
    """Non-breaking space (U+00A0) is normalized to space and collapsed."""
    assert normalize_for_matching("a\u00a0b") == "a b"
    assert normalize_for_matching("a \u00a0 b") == "a b"


# --- expand_contractions_for_matching ---


def test_expand_contractions_dont() -> None:
    assert "do not" in expand_contractions_for_matching("don't")
    assert "do not" in expand_contractions_for_matching("Don't")


def test_expand_contractions_im() -> None:
    assert "I am" in expand_contractions_for_matching("I'm")
    assert "I am" in expand_contractions_for_matching("i'm")


def test_expand_contractions_its() -> None:
    assert "it is" in expand_contractions_for_matching("it's") or "it has" in expand_contractions_for_matching("it's")


def test_expand_contractions_unchanged_when_no_contraction() -> None:
    s = "No contraction here."
    assert expand_contractions_for_matching(s) == s


def test_expand_contractions_wont() -> None:
    assert "will not" in expand_contractions_for_matching("won't")


def test_expand_contractions_aint() -> None:
    # ain't -> am not / is not / are not (we can expand to one variant for matching)
    out = expand_contractions_for_matching("ain't")
    assert "not" in out


def test_expand_contractions_typographic_apostrophe() -> None:
    """Contractions with typographic apostrophe (U+2019): normalize first, then expand (matching pipeline)."""
    # In matching we do normalize then expand; so normalized "It's" (curly -> straight) still expands
    normalized = normalize_for_matching("It\u2019s")
    assert "'" in normalized
    expanded = expand_contractions_for_matching(normalized)
    assert "it is" in expanded or "is" in expanded
    # Direct expand on typographic apostrophe: regex uses ASCII ', so normalize first in real flow
    expanded_curly = expand_contractions_for_matching("It\u2019s")
    assert "it is" in expanded_curly or "is" in expanded_curly or expanded_curly == "It\u2019s"


# --- word_in_text_for_matching (lemma/forms + normalized + expanded) ---


def test_word_in_text_for_matching_dont_matches_do() -> None:
    # "don't" in example -> match lemma "do" (via expanded "do not")
    assert word_in_text_for_matching("do", "I don't know.") is True
    assert word_in_text_for_matching("do", "I don't like it.") is True


def test_word_in_text_for_matching_im_matches_am() -> None:
    # "I'm" -> "I am" -> match "am" (form of "be"); caller checks lemma forms
    assert word_in_text_for_matching("am", "I'm here.") is True


def test_word_in_text_for_matching_typographic_apostrophe() -> None:
    # Curly apostrophe in text
    assert word_in_text_for_matching("it", "It\u2019s time.") is True
    assert word_in_text_for_matching("is", "It\u2019s time.") is True


def test_word_in_text_for_matching_plain_word_unchanged() -> None:
    # Whole-word only: "runs" matches, "run" does not match inside "runs"/"running"
    assert word_in_text_for_matching("runs", "He runs fast.") is True
    assert word_in_text_for_matching("run", "No run here.") is True
    assert word_in_text_for_matching("running", "He is running.") is True
    assert word_in_text_for_matching("xyz", "He runs.") is False


def test_word_in_text_for_matching_whole_word_only() -> None:
    assert word_in_text_for_matching("run", "trunk") is False
    # "run" is not a whole word inside "running"
    assert word_in_text_for_matching("run", "running") is False


# --- match_target_in_text (Stage 4: word or phrase) ---


def test_match_target_in_text_word_apostrophe() -> None:
    assert match_target_in_text("do", "I don't know.") is True
    assert match_target_in_text("it", "It's time.") is True


def test_match_target_in_text_phrase_substring() -> None:
    assert match_target_in_text("look up", "I look up the word.") is True
    assert match_target_in_text("look up", "We look up the address.") is True
    assert match_target_in_text("look up", "I look at the sky.") is False


def test_match_target_in_text_hyphenated() -> None:
    assert match_target_in_text("well-known", "He is well-known.") is True
    assert match_target_in_text("well-known", "A well-known fact.") is True


def test_match_target_in_text_every_example_rule() -> None:
    """Rule: target must be in EVERY example; caller checks each example separately."""
    assert match_target_in_text("runs", "He runs.") is True
    assert match_target_in_text("run", "No run here.") is True
    assert match_target_in_text("run", "The weather is nice.") is False
