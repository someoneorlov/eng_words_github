"""Tests for eng_words.word_family.qc_types (Stage 1.1 TDD)."""

from __future__ import annotations

import pytest

from eng_words.word_family.qc_types import (
    CONTRACT_ERROR_TYPES,
    ErrorType,
    QCFinding,
    QCGateExceeded,
    QCPolicy,
    Severity,
)


def test_qc_finding_serialize_deserialize_roundtrip() -> None:
    f = QCFinding(
        lemma="test",
        error_type=ErrorType.QC_LEMMA_NOT_IN_EXAMPLE,
        severity=Severity.WARNING,
        message="Lemma not in example.",
        meaning_id=1,
        context={"example_index": 0},
    )
    d = f.to_dict()
    assert d["lemma"] == "test"
    assert d["error_type"] == "lemma_not_in_example"
    assert d["severity"] == "warning"
    assert d["meaning_id"] == 1
    assert d["context"] == {"example_index": 0}
    f2 = QCFinding.from_dict(d)
    assert f2.lemma == f.lemma
    assert f2.error_type == f.error_type
    assert f2.severity == f.severity
    assert f2.message == f.message
    assert f2.meaning_id == f.meaning_id
    assert f2.context == f.context


def test_qc_finding_from_dict_minimal() -> None:
    d = {"lemma": "x", "error_type": "schema_error", "severity": "error", "message": "Bad"}
    f = QCFinding.from_dict(d)
    assert f.lemma == "x"
    assert f.error_type == ErrorType.CONTRACT_SCHEMA
    assert f.meaning_id is None
    assert f.context is None


def test_policy_strict_treats_qc_finding_as_error() -> None:
    policy = QCPolicy(strict=True)
    finding = QCFinding(
        lemma="able",
        error_type=ErrorType.QC_LEMMA_NOT_IN_EXAMPLE,
        severity=Severity.WARNING,
        message="Not in example",
    )
    assert policy.severity_for_finding(finding) == Severity.ERROR


def test_policy_strict_contract_remains_error() -> None:
    policy = QCPolicy(strict=True)
    finding = QCFinding(
        lemma="x",
        error_type=ErrorType.CONTRACT_SCHEMA,
        severity=Severity.ERROR,
        message="Missing fields",
    )
    assert policy.severity_for_finding(finding) == Severity.ERROR


def test_policy_relaxed_warning_stays_warning() -> None:
    policy = QCPolicy(strict=False)
    finding = QCFinding(
        lemma="able",
        error_type=ErrorType.QC_LEMMA_NOT_IN_EXAMPLE,
        severity=Severity.WARNING,
        message="Not in example",
    )
    assert policy.severity_for_finding(finding) == Severity.WARNING


def test_policy_relaxed_threshold_rate_exceeded_raises() -> None:
    policy = QCPolicy(
        strict=False,
        max_warning_rate=0.1,
        max_warnings_absolute=None,
    )
    policy.check_warning_thresholds(0, 100)
    policy.check_warning_thresholds(10, 100)
    with pytest.raises(QCGateExceeded) as exc_info:
        policy.check_warning_thresholds(11, 100, label="QC")
    assert "max_warning_rate" in str(exc_info.value)
    assert "11" in str(exc_info.value)


def test_policy_relaxed_threshold_absolute_exceeded_raises() -> None:
    policy = QCPolicy(
        strict=False,
        max_warning_rate=None,
        max_warnings_absolute=5,
    )
    policy.check_warning_thresholds(5, 1000)
    with pytest.raises(QCGateExceeded) as exc_info:
        policy.check_warning_thresholds(6, 1000)
    assert "max_warnings_absolute" in str(exc_info.value)


def test_contract_error_types_and_is_contract_error() -> None:
    for et in ErrorType:
        if et in CONTRACT_ERROR_TYPES:
            f = QCFinding(lemma="x", error_type=et, severity=Severity.ERROR, message="")
            assert f.is_contract_error
        else:
            f = QCFinding(lemma="x", error_type=et, severity=Severity.WARNING, message="")
            assert not f.is_contract_error
