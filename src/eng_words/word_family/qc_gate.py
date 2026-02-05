"""Pipeline B — QC gate thresholds and PASS/FAIL evaluation (Stage 7).

Single place for quality thresholds. Gate reads pipeline output (cards JSON),
aggregates validation_errors by error_type, computes rates, and returns PASS/FAIL.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class QCGateThresholds:
    """Maximum allowed rates (0.0–1.0) for QC error types. Default: 0 = no errors allowed (strict)."""

    max_lemma_not_in_example_rate: float = 0.0
    max_pos_mismatch_rate: float = 0.0
    max_duplicate_sense_rate: float = 0.0
    max_validation_rate: float = 0.0
    # Any other error_type not listed: allow 0 by default (fail on unknown)
    max_other_rate: float = 0.0

    def to_dict(self) -> dict[str, float]:
        return {
            "max_lemma_not_in_example_rate": self.max_lemma_not_in_example_rate,
            "max_pos_mismatch_rate": self.max_pos_mismatch_rate,
            "max_duplicate_sense_rate": self.max_duplicate_sense_rate,
            "max_validation_rate": self.max_validation_rate,
            "max_other_rate": self.max_other_rate,
        }


# Default: strict — no QC drops allowed
DEFAULT_QC_GATE_THRESHOLDS = QCGateThresholds()


# Error types that map to threshold keys
_ERROR_TYPE_TO_THRESHOLD: dict[str, str] = {
    "lemma_not_in_example": "max_lemma_not_in_example_rate",
    "pos_mismatch": "max_pos_mismatch_rate",
    "duplicate_sense": "max_duplicate_sense_rate",
    "validation": "max_validation_rate",
}


def _get_threshold_for_error_type(error_type: str, thresholds: QCGateThresholds) -> float:
    key = _ERROR_TYPE_TO_THRESHOLD.get(error_type, "max_other_rate")
    return getattr(thresholds, key)


def evaluate_gate(
    result: dict[str, Any],
    thresholds: QCGateThresholds = DEFAULT_QC_GATE_THRESHOLDS,
) -> tuple[bool, dict[str, Any], str]:
    """Evaluate QC gate on pipeline result (cards JSON payload).

    Args:
        result: dict with "cards", "stats", "validation_errors".
        thresholds: max allowed rates per error type.

    Returns:
        (passed, summary, message). passed is False if any rate exceeds threshold.
    """
    cards = result.get("cards") or []
    validation_errors = result.get("validation_errors") or []
    stats = result.get("stats") or {}

    cards_generated = len(cards)
    total_processed = cards_generated + len(validation_errors)
    if total_processed == 0:
        return True, {"total_processed": 0, "rates": {}}, "No cards processed; gate PASS (nothing to check)."

    by_type = Counter(e.get("error_type", "other") for e in validation_errors if isinstance(e, dict))
    rates: dict[str, float] = {}
    for error_type, count in by_type.items():
        rates[error_type] = count / total_processed

    failed: list[str] = []
    for error_type, rate in rates.items():
        max_allowed = _get_threshold_for_error_type(error_type, thresholds)
        if rate > max_allowed:
            failed.append(f"{error_type}: {rate:.2%} > {max_allowed:.2%}")

    summary: dict[str, Any] = {
        "cards_generated": cards_generated,
        "validation_errors_count": len(validation_errors),
        "total_processed": total_processed,
        "counts_by_type": dict(by_type),
        "rates": rates,
        "thresholds": thresholds.to_dict(),
    }
    passed = len(failed) == 0
    if passed:
        message = f"PASS: all QC rates within thresholds (processed {total_processed} cards, {len(validation_errors)} dropped)."
    else:
        message = "FAIL: " + "; ".join(failed)
    return passed, summary, message


def load_result_and_evaluate_gate(
    path: Path,
    thresholds: QCGateThresholds = DEFAULT_QC_GATE_THRESHOLDS,
) -> tuple[bool, dict[str, Any], str]:
    """Load cards JSON from path and evaluate gate. Returns (passed, summary, message)."""
    path = Path(path)
    if not path.exists():
        return False, {}, f"File not found: {path}"
    import json
    with open(path, encoding="utf-8") as f:
        result = json.load(f)
    return evaluate_gate(result, thresholds)
