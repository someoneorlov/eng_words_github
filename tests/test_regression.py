"""Tests for Pipeline B Stage 7 regression criteria."""

import json
from pathlib import Path

import pytest

from eng_words.word_family.regression import (
    REGRESSION_LEMMA_IN_EXAMPLE_MIN,
    REGRESSION_MAJOR_OR_INVALID_MAX,
    REGRESSION_POS_CONSISTENCY_MIN,
    REGRESSION_VALID_SCHEMA_MIN,
    evaluate_regression,
    load_result_and_evaluate_regression,
    RegressionResult,
)


class TestRegressionCriteria:
    def test_valid_schema_rate_all_valid(self):
        result = {
            "cards": [
                {
                    "lemma": "run",
                    "definition_en": "move",
                    "definition_ru": "бежать",
                    "part_of_speech": "verb",
                    "examples": ["He runs."],
                    "selected_example_indices": [1],
                },
            ],
        }
        r = evaluate_regression(result)
        assert r.valid_schema_rate == 1.0
        assert r.total_cards == 1
        assert r.major_or_invalid_rate == 0.0

    def test_valid_schema_rate_one_invalid(self):
        result = {
            "cards": [
                {
                    "lemma": "run",
                    "definition_en": "move",
                    "definition_ru": "бежать",
                    "part_of_speech": "verb",
                    "examples": ["He runs."],
                    "selected_example_indices": [1],
                },
                {"lemma": "go"},  # missing required fields
            ],
        }
        r = evaluate_regression(result)
        assert r.valid_schema_rate == 0.5
        assert r.passed is False
        assert any("valid_schema" in c for c in r.checklist)

    def test_empty_cards_pass_schema(self):
        r = evaluate_regression({"cards": []})
        assert r.total_cards == 0
        assert r.valid_schema_rate == 1.0
        assert r.passed is True

    def test_lemma_headword_in_example_computed_when_available(self):
        result = {
            "cards": [
                {
                    "lemma": "run",
                    "definition_en": "move",
                    "definition_ru": "бежать",
                    "part_of_speech": "verb",
                    "examples": ["He runs."],
                    "selected_example_indices": [1],
                },
            ],
        }
        r = evaluate_regression(result)
        # If batch_qc available, rate is computed; otherwise checklist has "skipped"
        assert r.lemma_headword_in_example_rate is not None or any("skipped" in c for c in r.checklist)

    def test_pos_consistency_na_without_lemma_pos(self):
        result = {
            "cards": [
                {
                    "lemma": "run",
                    "definition_en": "move",
                    "definition_ru": "бежать",
                    "part_of_speech": "verb",
                    "examples": ["He runs."],
                    "selected_example_indices": [1],
                },
            ],
        }
        r = evaluate_regression(result, lemma_pos_per_example=None)
        assert r.pos_consistency_rate == 1.0

    def test_pos_consistency_computed_when_provided(self):
        result = {
            "cards": [
                {
                    "lemma": "run",
                    "definition_en": "move",
                    "definition_ru": "бежать",
                    "part_of_speech": "verb",
                    "examples": ["He runs."],
                    "selected_example_indices": [1],
                },
            ],
        }
        lemma_pos = {"run": ["VERB"]}
        r = evaluate_regression(result, lemma_pos_per_example=lemma_pos)
        assert r.pos_consistency_rate is not None
        assert 0 <= r.pos_consistency_rate <= 1.0

    def test_to_dict(self):
        result = {"cards": [{"lemma": "x", "definition_en": "a", "definition_ru": "б", "part_of_speech": "noun", "examples": ["x."], "selected_example_indices": [1]}]}
        r = evaluate_regression(result)
        d = r.to_dict()
        assert "valid_schema_rate" in d
        assert "passed" in d
        assert "checklist" in d
        assert "failing_cards_count" in d


class TestLoadResultAndEvaluateRegression:
    def test_missing_file_fail(self, tmp_path: Path):
        r = load_result_and_evaluate_regression(tmp_path / "missing.json")
        assert r.passed is False
        assert r.total_cards == 0
        assert any("not found" in c for c in r.checklist)

    def test_load_and_evaluate(self, tmp_path: Path):
        path = tmp_path / "cards.json"
        path.write_text(
            json.dumps({
                "cards": [
                    {
                        "lemma": "run",
                        "definition_en": "move",
                        "definition_ru": "бежать",
                        "part_of_speech": "verb",
                        "examples": ["He runs."],
                        "selected_example_indices": [1],
                    },
                ],
            }),
            encoding="utf-8",
        )
        r = load_result_and_evaluate_regression(path)
        assert r.total_cards == 1
        assert r.valid_schema_rate == 1.0
