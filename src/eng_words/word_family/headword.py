"""Pipeline B headword: optional display form (e.g. phrasal verb) and QC.

Contract (PIPELINE_B_FIXES_PLAN Stage 3):
- lemma: grouping key (unchanged).
- headword: optional string for the card (e.g. "look up"); if set, QC checks headword in examples.
- Precision-first: accept headword only if it appears in every selected example (after normalize).
"""

from __future__ import annotations

try:
    from eng_words.text_norm import normalize_for_matching
except ImportError:
    normalize_for_matching = None


def headword_in_examples(card: dict) -> bool:
    """True if card has a headword and it appears in every example (normalized substring)."""
    h = card.get("headword")
    if not h or not isinstance(h, str):
        return False
    h = h.strip()
    if not h:
        return False
    examples = card.get("examples") or []
    if not examples:
        return False
    if normalize_for_matching is None:
        return False
    norm_h = normalize_for_matching(h).lower()
    for ex in examples:
        if not ex or not isinstance(ex, str):
            return False
        if norm_h not in normalize_for_matching(ex).lower():
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
