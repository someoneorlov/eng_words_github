"""Tests for Pipeline B QC gate (Stage 7)."""

import json
from pathlib import Path

import pytest

from eng_words.word_family.qc_gate import (
    DEFAULT_QC_GATE_THRESHOLDS,
    QCGateThresholds,
    evaluate_gate,
    load_result_and_evaluate_gate,
)


class TestQCGateThresholds:
    def test_default_all_zero(self):
        t = DEFAULT_QC_GATE_THRESHOLDS
        assert t.max_lemma_not_in_example_rate == 0.0
        assert t.max_pos_mismatch_rate == 0.0
        assert t.max_duplicate_sense_rate == 0.0
        assert t.max_validation_rate == 0.0
        assert t.max_headword_invalid_for_mode_rate == 0.0

    def test_to_dict(self):
        t = QCGateThresholds(max_lemma_not_in_example_rate=0.01)
        d = t.to_dict()
        assert d["max_lemma_not_in_example_rate"] == 0.01
        assert d["max_pos_mismatch_rate"] == 0.0


class TestEvaluateGate:
    def test_pass_when_no_validation_errors(self):
        result = {"cards": [{"lemma": "run"}], "stats": {}, "validation_errors": []}
        passed, summary, msg = evaluate_gate(result)
        assert passed is True
        assert "PASS" in msg
        assert summary["total_processed"] == 1
        assert summary["validation_errors_count"] == 0

    def test_fail_when_lemma_not_in_example_above_zero(self):
        result = {
            "cards": [{"lemma": "a"}],
            "stats": {},
            "validation_errors": [
                {"lemma": "x", "error_type": "lemma_not_in_example", "message": "drop"},
            ],
        }
        passed, summary, msg = evaluate_gate(result)
        assert passed is False
        assert "FAIL" in msg
        assert "lemma_not_in_example" in msg
        assert summary["rates"]["lemma_not_in_example"] == 0.5  # 1/(1+1)

    def test_pass_when_rate_within_threshold(self):
        result = {
            "cards": [{"lemma": "a"}, {"lemma": "b"}, {"lemma": "c"}, {"lemma": "d"}, {"lemma": "e"}],
            "stats": {},
            "validation_errors": [{"lemma": "x", "error_type": "pos_mismatch", "message": "drop"}],
        }
        # 1 error / 6 total = 16.67%; allow 20%
        thresholds = QCGateThresholds(max_pos_mismatch_rate=0.20)
        passed, summary, msg = evaluate_gate(result, thresholds)
        assert passed is True
        assert summary["rates"]["pos_mismatch"] == pytest.approx(1 / 6)

    def test_fail_when_rate_exceeds_threshold(self):
        result = {
            "cards": [{"lemma": "a"}],
            "stats": {},
            "validation_errors": [{"lemma": "x", "error_type": "duplicate_sense", "message": "drop"}],
        }
        thresholds = QCGateThresholds(max_duplicate_sense_rate=0.1)
        passed, _, msg = evaluate_gate(result, thresholds)
        assert passed is False
        assert "duplicate_sense" in msg
        assert "10.00%" in msg  # threshold shown in message

    def test_empty_result_pass(self):
        result = {"cards": [], "stats": {}, "validation_errors": []}
        passed, summary, msg = evaluate_gate(result)
        assert passed is True
        assert summary["total_processed"] == 0

    def test_fail_when_headword_invalid_for_mode_above_zero(self):
        result = {
            "cards": [{"lemma": "go", "headword": "go"}],
            "stats": {},
            "validation_errors": [
                {"lemma": "take off", "error_type": "headword_invalid_for_mode", "message": "drop"},
            ],
        }
        passed, summary, msg = evaluate_gate(result)
        assert passed is False
        assert "headword_invalid_for_mode" in msg
        assert summary["rates"]["headword_invalid_for_mode"] == 0.5  # 1/(1+1)


class TestLoadResultAndEvaluateGate:
    def test_missing_file_fail(self, tmp_path: Path):
        passed, summary, msg = load_result_and_evaluate_gate(tmp_path / "missing.json")
        assert passed is False
        assert "not found" in msg or "File not found" in msg

    def test_load_and_evaluate(self, tmp_path: Path):
        path = tmp_path / "cards.json"
        path.write_text(
            json.dumps({
                "cards": [{"lemma": "run"}],
                "stats": {},
                "validation_errors": [],
            }),
            encoding="utf-8",
        )
        passed, summary, msg = load_result_and_evaluate_gate(path)
        assert passed is True
        assert summary["cards_generated"] == 1
