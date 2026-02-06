"""Pipeline B QC: typed errors, findings, and strict/relaxed policy.

Contract errors: schema, parsing, index mapping — always fail.
QC errors: lemma_in_example, pos_mismatch, duplicates, bad_examples — in strict
treated as errors; in relaxed as warnings until threshold, then fail.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ErrorType(str, Enum):
    """Taxonomy: contract vs QC."""

    # Contract (invariants): always error, pipeline must not write output
    CONTRACT_JSON_PARSE = "json_parse_error"
    CONTRACT_SCHEMA = "schema_error"
    CONTRACT_INDEX_MAPPING = "out_of_range_indices"
    CONTRACT_EMPTY_EXAMPLES = "empty_examples"
    CONTRACT_BATCH_ARTIFACTS = "batch_artifacts_missing"

    # QC: in strict => error; in relaxed => warning until threshold
    QC_LEMMA_NOT_IN_EXAMPLE = "lemma_not_in_example"
    QC_POS_MISMATCH = "pos_mismatch"
    QC_POS_FORMAT = "pos_format"
    QC_DUPLICATE_SENSE = "duplicate_sense"
    QC_BAD_EXAMPLE = "bad_example"
    QC_HEADWORD_NOT_IN_EXAMPLES = "headword_not_in_examples"
    QC_HEADWORD_INVALID_FOR_MODE = "headword_invalid_for_mode"


class Severity(str, Enum):
    ERROR = "error"
    WARNING = "warning"


# Contract error types (always raise, no threshold)
CONTRACT_ERROR_TYPES = frozenset({
    ErrorType.CONTRACT_JSON_PARSE,
    ErrorType.CONTRACT_SCHEMA,
    ErrorType.CONTRACT_INDEX_MAPPING,
    ErrorType.CONTRACT_EMPTY_EXAMPLES,
    ErrorType.CONTRACT_BATCH_ARTIFACTS,
})


@dataclass
class QCFinding:
    """Single QC finding: lemma, card id, type, severity, message, optional context."""

    lemma: str
    error_type: ErrorType
    severity: Severity
    message: str
    meaning_id: int | None = None
    context: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "lemma": self.lemma,
            "error_type": self.error_type.value,
            "severity": self.severity.value,
            "message": self.message,
        }
        if self.meaning_id is not None:
            d["meaning_id"] = self.meaning_id
        if self.context:
            d["context"] = dict(self.context)
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> QCFinding:
        return cls(
            lemma=str(data["lemma"]),
            error_type=ErrorType(data["error_type"]),
            severity=Severity(data["severity"]),
            message=str(data["message"]),
            meaning_id=data.get("meaning_id"),
            context=data.get("context"),
        )

    @property
    def is_contract_error(self) -> bool:
        return self.error_type in CONTRACT_ERROR_TYPES


@dataclass
class QCPolicy:
    """Strict (default) or relaxed; in relaxed, thresholds cap warnings."""

    strict: bool = True
    max_warning_rate: float | None = None
    max_warnings_absolute: int | None = None

    def severity_for_finding(self, finding: QCFinding) -> Severity:
        """In strict mode, any QC finding is treated as ERROR."""
        if finding.is_contract_error:
            return Severity.ERROR
        if self.strict:
            return Severity.ERROR
        return finding.severity

    def check_warning_thresholds(
        self,
        warning_count: int,
        total_count: int,
        label: str = "QC",
    ) -> None:
        """Raise QCGateExceeded if warning_count exceeds policy thresholds."""
        if total_count <= 0:
            return
        rate = warning_count / total_count
        if self.max_warning_rate is not None and rate > self.max_warning_rate:
            raise QCGateExceeded(
                f"{label}: warning count {warning_count}/{total_count} = {rate:.2%} "
                f"exceeds max_warning_rate {self.max_warning_rate:.2%}. "
                "Fix data or increase threshold."
            )
        if self.max_warnings_absolute is not None and warning_count > self.max_warnings_absolute:
            raise QCGateExceeded(
                f"{label}: warning count {warning_count} exceeds "
                f"max_warnings_absolute {self.max_warnings_absolute}. "
                "Fix data or increase threshold."
            )


class QCGateExceeded(Exception):
    """Raised when QC warning thresholds are exceeded (relaxed mode)."""

    pass
