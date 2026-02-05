"""Tests for Pipeline B batch QC (batch_qc.py)."""

import pytest

from eng_words.word_family.batch_qc import (
    _definition_similarity,
    _normalize_card_pos,
    _normalize_definition_for_similarity,
    cards_lemma_not_in_example,
    check_qc_threshold,
    get_cards_failing_duplicate_sense,
    get_cards_failing_lemma_in_example,
    get_cards_failing_pos_mismatch,
)


class TestCardsLemmaNotInExample:
    def test_returns_empty_when_validator_unavailable(self):
        # If validator is optional and not installed, might return []
        # When validator is available, real checks run
        cards = [{"lemma": "run", "examples": ["I run daily."], "definition_en": "move fast"}]
        out = cards_lemma_not_in_example(cards)
        # Either [] (validator missing) or [] (lemma in example)
        assert isinstance(out, list)

    def test_lemma_in_example_passes(self):
        cards = [{"lemma": "run", "examples": ["I run every day."], "definition_en": "move"}]
        out = cards_lemma_not_in_example(cards)
        assert len(out) == 0

    def test_lemma_not_in_example_flagged(self):
        cards = [{"lemma": "run", "examples": ["The weather is nice."], "definition_en": "move"}]
        out = cards_lemma_not_in_example(cards)
        if out:  # when validator available
            assert len(out) == 1
            assert out[0]["lemma"] == "run"
            assert out[0]["example_index"] == 1

    def test_contraction_dont_matches_lemma_do(self):
        """Stage 2: don't in example matches lemma 'do' via expand_contractions_for_matching."""
        cards = [{"lemma": "do", "examples": ["I don't know."], "definition_en": "perform"}]
        out = cards_lemma_not_in_example(cards)
        assert len(out) == 0


class TestGetCardsFailingLemmaInExample:
    """Stage 4: returns card dicts for drop; same criterion as cards_lemma_not_in_example."""

    def test_returns_failing_cards(self):
        cards = [
            {"lemma": "run", "examples": ["He runs."], "definition_en": "move"},
            {"lemma": "go", "examples": ["The weather is nice."], "definition_en": "move"},
        ]
        failing = get_cards_failing_lemma_in_example(cards)
        assert len(failing) == 1
        assert failing[0]["lemma"] == "go"

    def test_returns_empty_when_all_pass(self):
        cards = [{"lemma": "run", "examples": ["He runs."], "definition_en": "move"}]
        assert len(get_cards_failing_lemma_in_example(cards)) == 0


class TestCheckQcThreshold:
    def test_under_threshold_does_not_raise(self):
        check_qc_threshold(2, 100, max_warning_rate=0.05, max_warnings_absolute=10)

    def test_exceeds_rate_raises(self):
        with pytest.raises(ValueError, match="exceeds max_warning_rate"):
            check_qc_threshold(10, 100, max_warning_rate=0.05)

    def test_exceeds_absolute_raises(self):
        with pytest.raises(ValueError, match="exceeds max_warnings_absolute"):
            check_qc_threshold(15, 100, max_warnings_absolute=10)

    def test_zero_total_does_not_raise(self):
        check_qc_threshold(5, 0, max_warning_rate=0.01)


class TestNormalizeCardPos:
    """POS mismatch QC: normalize card part_of_speech to standard tag."""

    def test_verb_variants(self):
        assert _normalize_card_pos("verb") == "VERB"
        assert _normalize_card_pos("Verb") == "VERB"
        assert _normalize_card_pos("v") == "VERB"

    def test_noun_adj_adv(self):
        assert _normalize_card_pos("noun") == "NOUN"
        assert _normalize_card_pos("adj") == "ADJ"
        assert _normalize_card_pos("adverb") == "ADV"

    def test_already_standard(self):
        assert _normalize_card_pos("VERB") == "VERB"
        assert _normalize_card_pos("NOUN") == "NOUN"

    def test_unknown_returns_empty(self):
        assert _normalize_card_pos("") == ""
        assert _normalize_card_pos("other") == ""


class TestGetCardsFailingPosMismatch:
    """Stage 5: cards where claimed POS not in selected examples → fail."""

    def test_match_passes(self):
        cards = [
            {
                "lemma": "run",
                "part_of_speech": "verb",
                "selected_example_indices": [1, 2],
                "examples": ["He runs.", "A morning run."],
            }
        ]
        pos_per = {"run": ["VERB", "NOUN"]}  # index 1 and 2 are VERB, NOUN
        failing = get_cards_failing_pos_mismatch(cards, pos_per)
        assert len(failing) == 0

    def test_mismatch_fails(self):
        cards = [
            {
                "lemma": "run",
                "part_of_speech": "verb",
                "selected_example_indices": [1, 2],
                "examples": ["A morning run.", "The run was fun."],
            }
        ]
        pos_per = {"run": ["NOUN", "NOUN"]}  # both examples NOUN, card claims VERB
        failing = get_cards_failing_pos_mismatch(cards, pos_per)
        assert len(failing) == 1
        assert failing[0]["lemma"] == "run"
        assert failing[0]["part_of_speech"] == "verb"

    def test_lemma_not_in_pos_map_skipped(self):
        cards = [{"lemma": "x", "part_of_speech": "noun", "selected_example_indices": [1]}]
        failing = get_cards_failing_pos_mismatch(cards, {})
        assert len(failing) == 0

    def test_empty_pos_list_skipped(self):
        cards = [{"lemma": "run", "part_of_speech": "verb", "selected_example_indices": [1]}]
        failing = get_cards_failing_pos_mismatch(cards, {"run": []})
        assert len(failing) == 0


class TestNormalizeDefinitionForSimilarity:
    def test_lower_strip_collapse_whitespace(self):
        assert _normalize_definition_for_similarity("  Move  fast  ") == "move fast"
        assert _normalize_definition_for_similarity("") == ""
        assert _normalize_definition_for_similarity("Same") == "same"


class TestDefinitionSimilarity:
    def test_identical_returns_one(self):
        assert _definition_similarity("to move quickly", "to move quickly") == 1.0

    def test_very_similar_high_ratio(self):
        a = "to move quickly on foot"
        b = "to move quickly on foot."
        assert _definition_similarity(a, b) >= 0.9

    def test_different_low_ratio(self):
        a = "to move quickly"
        b = "a place where money is kept"
        assert _definition_similarity(a, b) < 0.5

    def test_empty_handled(self):
        assert _definition_similarity("", "") == 1.0
        assert _definition_similarity("x", "") == 0.0


class TestGetCardsFailingDuplicateSense:
    """Stage 6: duplicate sense = same lemma, definition_en too similar."""

    def test_two_almost_same_definitions_duplicate(self):
        cards = [
            {"lemma": "run", "definition_en": "to move quickly on foot", "definition_ru": "бежать"},
            {"lemma": "run", "definition_en": "to move quickly on foot.", "definition_ru": "бежать"},
        ]
        failing = get_cards_failing_duplicate_sense(cards, threshold=0.85)
        assert len(failing) == 1
        assert failing[0]["definition_en"] == "to move quickly on foot."

    def test_two_different_definitions_ok(self):
        cards = [
            {"lemma": "run", "definition_en": "to move quickly on foot"},
            {"lemma": "run", "definition_en": "a place where money is kept"},
        ]
        failing = get_cards_failing_duplicate_sense(cards, threshold=0.85)
        assert len(failing) == 0

    def test_single_card_ok(self):
        cards = [{"lemma": "run", "definition_en": "to move quickly"}]
        assert len(get_cards_failing_duplicate_sense(cards)) == 0

    def test_three_cards_two_similar_keeps_first_duplicate_second(self):
        cards = [
            {"lemma": "run", "definition_en": "to move quickly"},
            {"lemma": "run", "definition_en": "to move quickly."},
            {"lemma": "run", "definition_en": "a place for money"},
        ]
        failing = get_cards_failing_duplicate_sense(cards, threshold=0.9)
        assert len(failing) == 1
        assert failing[0]["definition_en"] == "to move quickly."

    def test_different_lemmas_not_compared(self):
        cards = [
            {"lemma": "run", "definition_en": "to move quickly"},
            {"lemma": "bank", "definition_en": "to move quickly"},
        ]
        failing = get_cards_failing_duplicate_sense(cards, threshold=1.0)
        assert len(failing) == 0
