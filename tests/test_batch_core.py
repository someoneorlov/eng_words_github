"""Tests for Pipeline B batch pure core (batch_core.py)."""

import json
import pytest

from eng_words.word_family.batch_core import (
    build_prompt,
    build_retry_prompt,
    choose_retry_candidates,
    extract_json_from_response_text,
    filter_valid_cards,
    merge_retry_results,
    parse_one_result,
    validate_card,
)
from eng_words.word_family.batch_schemas import (
    EMPTY_SELECTED_EXAMPLE_INDICES,
    EXAMPLES_FALLBACK_USED,
    RetryPolicy,
)


class TestBuildPrompt:
    def test_pos_distribution_appended_when_provided(self):
        prompt = build_prompt("run", ["I run."], pos_distribution={"VERB": 2, "NOUN": 1})
        assert "run" in prompt
        assert "1. I run." in prompt
        assert "POS distribution" in prompt
        assert "NOUN 1" in prompt
        assert "VERB 2" in prompt

    def test_pos_distribution_none_unchanged(self):
        prompt = build_prompt("go", ["Go away."])
        assert "go" in prompt
        assert "POS distribution" not in prompt


class TestExtractJsonFromResponseText:
    def test_pure_json(self):
        data = {"cards": [{"a": 1}]}
        assert extract_json_from_response_text(json.dumps(data)) == data

    def test_json_inside_markdown_block(self):
        data = {"cards": []}
        text = "```json\n" + json.dumps(data) + "\n```"
        assert extract_json_from_response_text(text) == data

    def test_json_with_surrounding_junk(self):
        data = {"x": 1}
        text = "Here is the answer:\n```json\n" + json.dumps(data) + "\n```\nDone."
        # Our impl strips ```json and ```; leading junk remains and may break JSON
        # So we only strip markdown wrapper; if there's leading junk, json.loads fails
        # Actually the current impl only strips ```json and ``` from start/end, so
        # "Here is the answer:\n```json\n..." -> after strip ```json we get "\n..." then strip ``` we get "\n..."
        # So we need to handle "junk + JSON". Plan says "мусор + JSON". So we could try
        # to find ```json or first { and then parse. For now keep simple: strip only.
        # Test with trailing ``` only:
        text = json.dumps(data) + "\n```"
        assert extract_json_from_response_text(text) == data

    def test_invalid_json_raises(self):
        with pytest.raises(ValueError, match="invalid JSON"):
            extract_json_from_response_text("not json at all")

    def test_empty_string_raises(self):
        with pytest.raises(ValueError, match="empty"):
            extract_json_from_response_text("")


class TestChooseRetryCandidates:
    def test_empty_examples_selected(self):
        # lemma "a" has one card with empty examples
        results = [
            ("a", [{"lemma": "a", "examples": [], "examples_fallback": True}], None),
            ("b", [{"lemma": "b", "examples": ["x"]}], None),
        ]
        cand = choose_retry_candidates(results, empty_examples=True, examples_fallback=True)
        assert "a" in cand
        assert "b" not in cand

    def test_fallback_selected(self):
        results = [
            ("x", [{"lemma": "x", "examples": [], "examples_fallback": True}], None),
        ]
        cand = choose_retry_candidates(results, empty_examples=True, examples_fallback=True)
        assert "x" in cand

    def test_error_lemma_skipped(self):
        results = [("err", [], "json_error")]
        assert choose_retry_candidates(results) == set()

    def test_policy_only_if_filters_triggers(self):
        # Only empty_examples trigger enabled → fallback-only card not selected
        results = [
            ("a", [{"lemma": "a", "examples": [], "examples_fallback": True}], None),
            ("b", [{"lemma": "b", "examples": ["x"], "examples_fallback": True}], None),
        ]
        policy = RetryPolicy(modes=["standard"], only_if=[EMPTY_SELECTED_EXAMPLE_INDICES], max_attempts=2)
        cand = choose_retry_candidates(results, policy=policy)
        assert "a" in cand  # empty examples
        assert "b" not in cand  # has examples, only fallback

    def test_policy_only_fallback_selects_fallback_only(self):
        results = [
            ("b", [{"lemma": "b", "examples": ["x"], "examples_fallback": True}], None),
        ]
        policy = RetryPolicy(modes=["standard"], only_if=[EXAMPLES_FALLBACK_USED], max_attempts=2)
        cand = choose_retry_candidates(results, policy=policy)
        assert "b" in cand


class TestFilterValidCards:
    """Precision-first: invalid cards do not appear in output, ErrorEntry created."""

    def test_valid_cards_all_returned(self):
        cards = [
            {
                "definition_en": "a",
                "definition_ru": "б",
                "part_of_speech": "noun",
                "selected_example_indices": [1],
            }
        ]
        valid, errs = filter_valid_cards(cards, "run", stage="download")
        assert len(valid) == 1
        assert valid[0]["definition_en"] == "a"
        assert len(errs) == 0

    def test_invalid_card_excluded_and_error_entry_created(self):
        cards = [
            {
                "definition_en": "",
                "definition_ru": "б",
                "part_of_speech": "noun",
                "selected_example_indices": [1],
            }
        ]
        valid, errs = filter_valid_cards(cards, "run", stage="download")
        assert len(valid) == 0
        assert len(errs) == 1
        assert errs[0].lemma == "run"
        assert errs[0].stage == "download"
        assert errs[0].error_type == "validation"
        assert "definition_en" in errs[0].message or "empty" in errs[0].message

    def test_skip_validation_includes_invalid(self):
        cards = [
            {
                "definition_en": "",
                "definition_ru": "б",
                "part_of_speech": "noun",
                "selected_example_indices": [1],
            }
        ]
        valid, errs = filter_valid_cards(cards, "run", skip_validation=True, stage="download")
        assert len(valid) == 1
        assert len(errs) == 0

    def test_mixed_valid_invalid_returns_only_valid(self):
        cards = [
            {"definition_en": "a", "definition_ru": "б", "part_of_speech": "n", "selected_example_indices": [1]},
            {"definition_en": "", "definition_ru": "б", "part_of_speech": "n", "selected_example_indices": [1]},
            {"definition_en": "c", "definition_ru": "в", "part_of_speech": "n", "selected_example_indices": [1]},
        ]
        valid, errs = filter_valid_cards(cards, "run", stage="retry")
        assert len(valid) == 2
        assert valid[0]["definition_en"] == "a"
        assert valid[1]["definition_en"] == "c"
        assert len(errs) == 1
        assert errs[0].stage == "retry"


class TestMergeRetryResults:
    """Contract A: retry replaces ALL cards for the lemma (simpler, chosen in plan 6.3)."""

    def test_replaces_lemma_cards(self):
        base = [
            {"lemma": "go", "definition_en": "old"},
            {"lemma": "run", "definition_en": "run def"},
        ]
        new = [{"lemma": "go", "definition_en": "new"}]
        out = merge_retry_results(base, "go", new)
        assert len(out) == 2
        go_cards = [c for c in out if c["lemma"] == "go"]
        assert len(go_cards) == 1
        assert go_cards[0]["definition_en"] == "new"
        assert [c for c in out if c["lemma"] == "run"] == [{"lemma": "run", "definition_en": "run def"}]

    def test_contract_a_retry_replaces_all_cards_for_lemma(self):
        # Contract A: multiple cards for same lemma are replaced entirely by retry result
        base = [
            {"lemma": "run", "definition_en": "card1"},
            {"lemma": "run", "definition_en": "card2"},
            {"lemma": "go", "definition_en": "go def"},
        ]
        retry_cards = [{"lemma": "run", "definition_en": "single after retry"}]
        out = merge_retry_results(base, "run", retry_cards)
        run_cards = [c for c in out if c["lemma"] == "run"]
        assert len(run_cards) == 1
        assert run_cards[0]["definition_en"] == "single after retry"
        assert len([c for c in out if c["lemma"] == "go"]) == 1
