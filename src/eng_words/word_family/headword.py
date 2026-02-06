"""Pipeline B headword: optional display form (e.g. phrasal verb) and QC.

Contract (PIPELINE_B_FIXES_PLAN Stage 3):
- lemma: grouping key (unchanged).
- headword: optional string for the card (e.g. "look up"); if set, QC checks headword in examples.
- Precision-first: accept headword only if it appears in every selected example (after normalize).
- Word mode: headword must be single-word (no spaces). Phrasal/MWE mode: multiword allowed but must match.
Uses match_target_in_text (normalize + contractions) so lemma/headword QC is consistent.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

try:
    from eng_words.text_norm import match_target_in_text
except ImportError:
    match_target_in_text = None

if TYPE_CHECKING:
    from eng_words.word_family.qc_types import QCFinding


def is_single_word(headword: str) -> bool:
    """True if headword is a single token (no spaces). Word mode requires this."""
    return bool(headword and " " not in headword.strip())


def headword_in_examples(card: dict) -> bool:
    """True if card has a headword and it appears in every example (normalized + contractions for matching)."""
    h = card.get("headword")
    if not h or not isinstance(h, str):
        return False
    h = h.strip()
    if not h:
        return False
    examples = card.get("examples") or []
    if not examples:
        return False
    if match_target_in_text is None:
        return False
    for ex in examples:
        if not ex or not isinstance(ex, str):
            return False
        if not match_target_in_text(h, ex):
            return False
    return True


def infer_headword(card: dict) -> str | None:
    """Return headword only if LLM provided it and it appears in every example; else None.

    Prevents invented headwords from passing QC.
    """
    h = card.get("headword")
    if not h or not isinstance(h, str):
        return None
    h = h.strip()
    if not h:
        return None
    if not headword_in_examples(card):
        return None
    return h


def resolve_headword(
    card: dict,
    mode: str = "word",
) -> tuple[str | None, "QCFinding | None"]:
    """Resolve headword for a card: valid → (headword, None); invalid/absent → (None, finding or None).

    - No headword in card → (None, None).
    - Headword not in every example → (None, QC_HEADWORD_NOT_IN_EXAMPLES).
    - Mode "word" and headword contains space → (None, QC_HEADWORD_INVALID_FOR_MODE).
    - Phrasal/MWE mode: multiword allowed; must still be in examples (already checked above).
    In strict, caller should drop card when finding is not None.
    """
    from eng_words.word_family.qc_types import ErrorType, QCFinding, Severity

    h = card.get("headword")
    if not h or not isinstance(h, str):
        return (None, None)
    h = h.strip()
    if not h:
        return (None, None)
    if not headword_in_examples(card):
        return (
            None,
            QCFinding(
                lemma=card.get("lemma", ""),
                error_type=ErrorType.QC_HEADWORD_NOT_IN_EXAMPLES,
                severity=Severity.ERROR,
                message="Headword not found in every example; reject invented headword (precision-first).",
            ),
        )
    if mode == "word" and not is_single_word(h):
        return (
            None,
            QCFinding(
                lemma=card.get("lemma", ""),
                error_type=ErrorType.QC_HEADWORD_INVALID_FOR_MODE,
                severity=Severity.ERROR,
                message=f"Word mode requires single-word headword; got multiword '{h}'.",
                context={"headword": h, "mode": mode},
            ),
        )
    return (h, None)
