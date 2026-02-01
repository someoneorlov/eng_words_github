"""Tests for Pipeline B Batch API script (scripts/run_pipeline_b_batch.py)."""

import json
import sys
from pathlib import Path

import pytest

# Import script module (run from project root)
SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_pipeline_b_batch as batch


class TestBuildPrompt:
    """Tests for build_prompt."""

    def test_includes_lemma_and_numbered_examples(self):
        prompt = batch.build_prompt("test", ["First sentence.", "Second one."])
        assert "test" in prompt
        assert "1. First sentence." in prompt
        assert "2. Second one." in prompt

    def test_empty_examples_still_has_lemma(self):
        prompt = batch.build_prompt("word", [])
        assert "word" in prompt

    def test_prompt_includes_index_reminder_when_examples_present(self):
        prompt = batch.build_prompt("run", ["He runs.", "She runs."])
        assert "1-based" in prompt
        assert "1 to 2" in prompt
        assert "exactly 2 examples" in prompt

    def test_retry_prompt_concatenates_critical_reminder(self):
        prompt = batch.build_retry_prompt("run", ["He runs.", "She runs."])
        assert "run" in prompt
        assert "1. He runs." in prompt
        assert "CRITICAL" in prompt
        assert "Your previous response had invalid" in prompt
        assert "1 to 2" in prompt


class TestParseOneResult:
    """Tests for _parse_one_result."""

    def test_bad_key_returns_error(self):
        lemma, cards, err = batch._parse_one_result("other:key", {}, {})
        assert err == "bad_key"
        assert cards == []

    def test_no_candidates_returns_error(self):
        lemma, cards, err = batch._parse_one_result(
            "lemma:run", {"candidates": []}, {}
        )
        assert lemma == "run"
        assert err == "no_candidates"
        assert cards == []

    def test_valid_response_parsed_into_cards(self):
        raw = {
            "cards": [
                {
                    "meaning_id": 1,
                    "definition_en": "move fast",
                    "definition_ru": "бежать",
                    "part_of_speech": "verb",
                    "selected_example_indices": [1, 2],
                    "generated_example": "He runs every day.",
                }
            ],
            "ignored_indices": [],
            "ignore_reasons": {},
        }
        response = {
            "candidates": [
                {
                    "content": {
                        "parts": [{"text": json.dumps(raw)}],
                    }
                }
            ]
        }
        lemma_examples = {"run": ["He runs.", "She runs too."]}
        lemma, cards, err = batch._parse_one_result(
            "lemma:run", response, lemma_examples
        )
        assert err is None
        assert lemma == "run"
        assert len(cards) == 1
        card = cards[0]
        assert card["lemma"] == "run"
        assert card["definition_en"] == "move fast"
        assert card["examples"] == ["He runs.", "She runs too."]
        assert card["source"] == "pipeline_b_batch"

    def test_zero_based_indices_accepted(self):
        """LLM may return 0-based indices despite prompt asking for 1-based."""
        raw = {
            "cards": [
                {
                    "meaning_id": 1,
                    "definition_en": "first sense",
                    "definition_ru": "первый",
                    "part_of_speech": "noun",
                    "selected_example_indices": [0, 1],
                    "generated_example": "Example.",
                }
            ],
            "ignored_indices": [],
            "ignore_reasons": {},
        }
        response = {
            "candidates": [
                {
                    "content": {
                        "parts": [{"text": json.dumps(raw)}],
                    }
                }
            ]
        }
        lemma_examples = {"word": ["First sentence.", "Second sentence."]}
        lemma, cards, err = batch._parse_one_result(
            "lemma:word", response, lemma_examples
        )
        assert err is None
        assert len(cards) == 1
        assert cards[0]["examples"] == ["First sentence.", "Second sentence."]

    def test_out_of_range_indices_stay_empty_and_flag_fallback(self):
        """When indices are out of range, leave examples empty and set examples_fallback (no substitution; all go to retry)."""
        raw = {
            "cards": [
                {
                    "meaning_id": 1,
                    "definition_en": "some def",
                    "definition_ru": "перевод",
                    "part_of_speech": "noun",
                    "selected_example_indices": [2],
                    "generated_example": "Example.",
                }
            ],
            "ignored_indices": [],
            "ignore_reasons": {},
        }
        response = {
            "candidates": [
                {
                    "content": {
                        "parts": [{"text": json.dumps(raw)}],
                    }
                }
            ]
        }
        lemma_examples = {"word": ["Only one example."]}
        lemma, cards, err = batch._parse_one_result(
            "lemma:word", response, lemma_examples
        )
        assert err is None
        assert len(cards) == 1
        assert cards[0]["examples"] == []
        assert cards[0].get("examples_fallback") is True

    def test_json_in_markdown_block_stripped(self):
        raw = {
            "cards": [
                {
                    "meaning_id": 1,
                    "definition_en": "a",
                    "definition_ru": "б",
                    "part_of_speech": "noun",
                    "selected_example_indices": [1],
                    "generated_example": "Example.",
                }
            ],
            "ignored_indices": [],
            "ignore_reasons": {},
        }
        response = {
            "candidates": [
                {
                    "content": {
                        "parts": [{"text": "```json\n" + json.dumps(raw) + "\n```"}],
                    }
                }
            ]
        }
        lemma, cards, err = batch._parse_one_result(
            "lemma:x", response, {"x": ["One."]}
        )
        assert err is None
        assert len(cards) == 1
        assert cards[0]["definition_en"] == "a"

    def test_invalid_json_returns_error(self):
        response = {
            "candidates": [
                {"content": {"parts": [{"text": "not json at all"}]}}
            ]
        }
        lemma, cards, err = batch._parse_one_result("lemma:y", response, {})
        assert lemma == "y"
        assert err is not None
        assert "json_error" in err
        assert cards == []


class TestValidateCard:
    """Tests for _validate_card."""

    def test_complete_card_passes(self):
        card = {
            "definition_en": "def",
            "definition_ru": "перевод",
            "part_of_speech": "verb",
            "selected_example_indices": [1],
        }
        errs = batch._validate_card(card, "test")
        assert errs == []

    def test_missing_definition_en_fails(self):
        card = {
            "definition_ru": "перевод",
            "part_of_speech": "verb",
            "selected_example_indices": [1],
        }
        errs = batch._validate_card(card, "test")
        assert any("definition_en" in e for e in errs)

    def test_empty_definition_ru_fails(self):
        card = {
            "definition_en": "def",
            "definition_ru": "  ",
            "part_of_speech": "verb",
            "selected_example_indices": [1],
        }
        errs = batch._validate_card(card, "test")
        assert any("definition_ru" in e for e in errs)

    def test_missing_selected_example_indices_fails(self):
        card = {
            "definition_en": "def",
            "definition_ru": "перевод",
            "part_of_speech": "verb",
        }
        errs = batch._validate_card(card, "test")
        assert any("selected_example_indices" in e for e in errs)


class TestLoadLemmaGroups:
    """Tests for load_lemma_groups (file existence checks)."""

    def test_missing_tokens_raises(self, tmp_path, monkeypatch):
        monkeypatch.setattr(batch, "TOKENS_PATH", tmp_path / "tokens.parquet")
        monkeypatch.setattr(batch, "SENTENCES_PATH", tmp_path / "sentences.parquet")
        with pytest.raises(FileNotFoundError, match="Tokens not found"):
            batch.load_lemma_groups()

    def test_missing_sentences_raises(self, tmp_path, monkeypatch):
        import pandas as pd

        # Minimal valid tokens parquet so we pass the first existence check
        pd.DataFrame({"lemma": [], "sentence_id": [], "pos": [], "is_alpha": [], "is_stop": []}).to_parquet(
            tmp_path / "tokens.parquet"
        )
        monkeypatch.setattr(batch, "TOKENS_PATH", tmp_path / "tokens.parquet")
        monkeypatch.setattr(batch, "SENTENCES_PATH", tmp_path / "sentences.parquet")
        with pytest.raises(FileNotFoundError, match="Sentences not found"):
            batch.load_lemma_groups()
