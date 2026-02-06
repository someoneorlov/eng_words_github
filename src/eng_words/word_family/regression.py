"""Pipeline B — Stage 7 regression criteria (49/49 PASS/FAIL).

Evaluates cards output against strict criteria (PIPELINE_B_FIXES_PLAN_2 §7.3):
- valid_schema_rate == 1.0
- lemma/headword_in_example_rate >= 0.98
- pos_consistency_rate >= 0.98
- major_or_invalid_rate == 0

If FAIL → returns checklist of specific cards and reasons.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from eng_words.word_family.contract import REQUIRED_CARD_FIELDS

# Thresholds for PASS (Stage 7.3)
REGRESSION_VALID_SCHEMA_MIN = 1.0
REGRESSION_LEMMA_IN_EXAMPLE_MIN = 0.98
REGRESSION_POS_CONSISTENCY_MIN = 0.98
REGRESSION_MAJOR_OR_INVALID_MAX = 0.0


@dataclass
class RegressionResult:
    """Result of regression evaluation: rates, passed flag, checklist."""

    valid_schema_rate: float
    lemma_headword_in_example_rate: float | None  # None if check skipped (missing dep)
    pos_consistency_rate: float | None  # None if lemma_pos_per_example not provided
    major_or_invalid_rate: float
    total_cards: int
    passed: bool
    checklist: list[str] = field(default_factory=list)
    failing_cards: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid_schema_rate": self.valid_schema_rate,
            "lemma_headword_in_example_rate": self.lemma_headword_in_example_rate,
            "pos_consistency_rate": self.pos_consistency_rate,
            "major_or_invalid_rate": self.major_or_invalid_rate,
            "total_cards": self.total_cards,
            "passed": self.passed,
            "checklist": self.checklist,
            "failing_cards_count": len(self.failing_cards),
        }


def _card_valid_schema(card: dict) -> bool:
    """True if card has required fields and valid types (examples list, indices ints)."""
    if not isinstance(card, dict):
        return False
    for key in REQUIRED_CARD_FIELDS:
        if key not in card:
            return False
    ex = card.get("examples")
    if not isinstance(ex, list):
        return False
    if not ex:
        return False
    idx = card.get("selected_example_indices", [])
    if not isinstance(idx, list):
        return False
    if not all(isinstance(i, int) for i in idx):
        return False
    return True


def evaluate_regression(
    result: dict[str, Any],
    *,
    lemma_pos_per_example: dict[str, list[str]] | None = None,
    manual_qc: dict[str, str] | None = None,
) -> RegressionResult:
    """Evaluate pipeline result against Stage 7 regression criteria.

    Args:
        result: pipeline output with "cards" (and optionally "validation_errors").
        lemma_pos_per_example: optional lemma -> list of POS per example (for pos_consistency).
        manual_qc: optional card_key -> "OK"|"Minor"|"Major"|"Invalid" (card_key e.g. lemma+meaning_id).

    Returns:
        RegressionResult with rates, passed, and checklist (what to fix if FAIL).
    """
    cards = result.get("cards") or []
    n = len(cards)
    checklist: list[str] = []
    failing_cards: list[dict[str, Any]] = []

    # 1) valid_schema_rate
    schema_ok = sum(1 for c in cards if _card_valid_schema(c))
    valid_schema_rate = schema_ok / n if n else 1.0
    if valid_schema_rate < REGRESSION_VALID_SCHEMA_MIN:
        bad = [c for c in cards if not _card_valid_schema(c)]
        for c in bad[:10]:
            failing_cards.append({"lemma": c.get("lemma"), "reason": "invalid_schema"})
        if len(bad) > 10:
            checklist.append(f"valid_schema: {schema_ok}/{n} — {len(bad)} cards invalid (contract/schema). Fix: contract.py, parser.")
        else:
            checklist.append(f"valid_schema: {schema_ok}/{n} — cards {[c.get('lemma') for c in bad]} invalid. Fix: contract + parser.")

    # 2) lemma/headword_in_example_rate
    lemma_headword_in_example_rate: float | None = None
    try:
        from eng_words.word_family.batch_qc import get_cards_failing_lemma_in_example

        failing_li = get_cards_failing_lemma_in_example(cards)
        ok_li = n - len(failing_li)
        lemma_headword_in_example_rate = ok_li / n if n else 1.0
        if lemma_headword_in_example_rate < REGRESSION_LEMMA_IN_EXAMPLE_MIN:
            for c in failing_li[:10]:
                failing_cards.append({"lemma": c.get("lemma"), "reason": "lemma_headword_not_in_example"})
            checklist.append(
                f"lemma/headword_in_example: {ok_li}/{n} ({lemma_headword_in_example_rate:.2%}). "
                "Fix: text_norm matching, headword resolution (Stage 2–3)."
            )
    except (ImportError, RuntimeError) as e:
        checklist.append(f"lemma/headword_in_example: check skipped ({e}). Install deps or fix batch_qc.")

    # 3) pos_consistency_rate
    pos_consistency_rate: float | None = None
    if lemma_pos_per_example:
        try:
            from eng_words.word_family.batch_qc import get_cards_failing_pos_mismatch

            failing_pos = get_cards_failing_pos_mismatch(cards, lemma_pos_per_example)
            ok_pos = n - len(failing_pos)
            pos_consistency_rate = ok_pos / n if n else 1.0
            if pos_consistency_rate < REGRESSION_POS_CONSISTENCY_MIN:
                for c in failing_pos[:10]:
                    failing_cards.append({"lemma": c.get("lemma"), "reason": "pos_mismatch"})
                checklist.append(
                    f"pos_consistency: {ok_pos}/{n} ({pos_consistency_rate:.2%}). "
                    "Fix: prompt/POS hint, retry super-strict (Stage 6.2)."
                )
        except (ImportError, RuntimeError) as e:
            checklist.append(f"pos_consistency: check skipped ({e}).")
    else:
        pos_consistency_rate = 1.0  # N/A → treat as pass

    # 4) major_or_invalid_rate (manual QC)
    major_or_invalid_rate = 0.0
    if manual_qc and n:
        major_invalid = sum(1 for v in manual_qc.values() if v in ("Major", "Invalid"))
        # manual_qc might be keyed by only part of cards
        total_qc = len(manual_qc)
        major_or_invalid_rate = major_invalid / total_qc if total_qc else 0.0
        if major_or_invalid_rate > REGRESSION_MAJOR_OR_INVALID_MAX:
            checklist.append(
                f"major_or_invalid: {major_invalid}/{total_qc} manual QC. Fix: data/prompt/QC (Stage 7 manual)."
            )

    # Pass/fail
    passed = (
        valid_schema_rate >= REGRESSION_VALID_SCHEMA_MIN
        and (lemma_headword_in_example_rate is None or lemma_headword_in_example_rate >= REGRESSION_LEMMA_IN_EXAMPLE_MIN)
        and (pos_consistency_rate is None or pos_consistency_rate >= REGRESSION_POS_CONSISTENCY_MIN)
        and major_or_invalid_rate <= REGRESSION_MAJOR_OR_INVALID_MAX
    )

    return RegressionResult(
        valid_schema_rate=valid_schema_rate,
        lemma_headword_in_example_rate=lemma_headword_in_example_rate,
        pos_consistency_rate=pos_consistency_rate,
        major_or_invalid_rate=major_or_invalid_rate,
        total_cards=n,
        passed=passed,
        checklist=checklist,
        failing_cards=failing_cards,
    )


def load_result_and_evaluate_regression(
    path: Path,
    *,
    lemma_pos_per_example_path: Path | None = None,
    manual_qc_path: Path | None = None,
) -> RegressionResult:
    """Load cards JSON and optional lemma_pos_per_example / manual_qc; run evaluate_regression."""
    path = Path(path)
    if not path.exists():
        return RegressionResult(
            valid_schema_rate=0.0,
            lemma_headword_in_example_rate=0.0,
            pos_consistency_rate=0.0,
            major_or_invalid_rate=1.0,
            total_cards=0,
            passed=False,
            checklist=[f"File not found: {path}"],
        )
    with open(path, encoding="utf-8") as f:
        result = json.load(f)

    lemma_pos: dict[str, list[str]] | None = None
    if lemma_pos_per_example_path and lemma_pos_per_example_path.exists():
        with open(lemma_pos_per_example_path, encoding="utf-8") as f:
            lemma_pos = json.load(f)

    manual_qc: dict[str, str] | None = None
    if manual_qc_path and manual_qc_path.exists():
        with open(manual_qc_path, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            manual_qc = data

    return evaluate_regression(
        result,
        lemma_pos_per_example=lemma_pos,
        manual_qc=manual_qc,
    )
