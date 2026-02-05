"""Pipeline B batch — QC checks and threshold gates.

lemma_not_in_example: cards where example does not contain lemma (or valid form).
When card has headword (Stage 3): QC checks headword in examples instead of lemma/forms.
pos_mismatch: card claims a POS but selected examples have 0 occurrences of that POS.
duplicate_sense: two cards for same lemma with very similar definition_en (cheap heuristic).
Thresholds: in strict mode fail when exceeded; in relaxed record warnings with limits.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Any

# Optional: use example_validator for word forms; text_norm for matching (normalize + contractions)
try:
    from eng_words.validation.example_validator import _get_word_forms
except ImportError:
    _get_word_forms = None
try:
    from eng_words.text_norm import word_in_text_for_matching
except ImportError:
    word_in_text_for_matching = None
try:
    from eng_words.word_family.headword import headword_in_examples
except ImportError:
    headword_in_examples = None

# Card part_of_speech -> standard POS tag (for POS mismatch QC)
_CARD_POS_TO_STANDARD: dict[str, str] = {
    "noun": "NOUN", "n": "NOUN", "n.": "NOUN",
    "verb": "VERB", "v": "VERB", "v.": "VERB",
    "adj": "ADJ", "adjective": "ADJ", "adj.": "ADJ", "a": "ADJ",
    "adv": "ADV", "adverb": "ADV", "adv.": "ADV", "r": "ADV",
}
CONTENT_POS_TAGS = frozenset({"NOUN", "VERB", "ADJ", "ADV"})


def _normalize_card_pos(part_of_speech: str) -> str:
    """Normalize card part_of_speech to standard tag (NOUN, VERB, ADJ, ADV) or empty if unknown."""
    if not part_of_speech or not isinstance(part_of_speech, str):
        return ""
    key = part_of_speech.strip().lower()
    out = _CARD_POS_TO_STANDARD.get(key)
    if out:
        return out
    # Already standard?
    if part_of_speech.upper() in CONTENT_POS_TAGS:
        return part_of_speech.upper()
    return ""


# Homograph forms to exclude (e.g. hop vs hope)
DEFAULT_HOMONYM_EXCLUDE: dict[str, set[str]] = {
    "hop": {"hoped", "hoping"},
    "hope": {"hopped", "hopping"},
}


def cards_lemma_not_in_example(
    all_cards: list[dict],
    homonym_exclude: dict[str, set[str]] | None = None,
) -> list[dict[str, Any]]:
    """Cards where at least one example does not contain the lemma (or valid form), or headword when set.
    When card has headword (Stage 3): checks headword in every example; else checks lemma/forms.
    Returns list of {lemma, definition_en, example_index, example_preview} for review.
    """
    if _get_word_forms is None or word_in_text_for_matching is None:
        return []
    exclude = homonym_exclude or DEFAULT_HOMONYM_EXCLUDE
    out: list[dict[str, Any]] = []
    for c in all_cards:
        lemma = c.get("lemma", "")
        examples = c.get("examples") or []
        if not examples:
            continue
        if c.get("headword") and headword_in_examples is not None:
            if not headword_in_examples(c):
                out.append({
                    "lemma": lemma,
                    "definition_en": (c.get("definition_en") or "")[:80],
                    "example_index": 1,
                    "example_preview": (examples[0][:100] + "...") if len(examples[0]) > 100 else examples[0],
                })
            continue
        forms = _get_word_forms(lemma) - exclude.get(lemma.lower(), set())
        for i, ex in enumerate(examples):
            if not any(word_in_text_for_matching(f, ex) for f in forms):
                out.append({
                    "lemma": lemma,
                    "definition_en": (c.get("definition_en") or "")[:80],
                    "example_index": i + 1,
                    "example_preview": (ex[:100] + "...") if len(ex) > 100 else ex,
                })
                break
    return out


def get_cards_failing_lemma_in_example(
    all_cards: list[dict],
    homonym_exclude: dict[str, set[str]] | None = None,
) -> list[dict]:
    """Cards that fail lemma/headword-in-example check (Stage 4: drop these in strict mode).
    Same criterion as cards_lemma_not_in_example but returns the card dicts for filtering.
    """
    report = cards_lemma_not_in_example(all_cards, homonym_exclude=homonym_exclude)
    if not report:
        return []
    exclude = homonym_exclude or DEFAULT_HOMONYM_EXCLUDE
    failing: list[dict] = []
    for c in all_cards:
        lemma = c.get("lemma", "")
        examples = c.get("examples") or []
        if not examples:
            continue
        if c.get("headword") and headword_in_examples is not None:
            if not headword_in_examples(c):
                failing.append(c)
            continue
        forms = _get_word_forms(lemma) - exclude.get(lemma.lower(), set())
        for ex in examples:
            if not any(word_in_text_for_matching(f, ex) for f in forms):
                failing.append(c)
                break
    return failing


def get_cards_failing_pos_mismatch(
    all_cards: list[dict],
    lemma_pos_per_example: dict[str, list[str]],
) -> list[dict]:
    """Cards that fail POS mismatch check: card claims a POS but selected examples have no such POS.
    lemma_pos_per_example: lemma -> list of POS (one per example, same order as lemma_examples).
    If lemma not in map or list empty, card is not checked (not failed). Returns card dicts for drop.
    """
    failing: list[dict] = []
    for c in all_cards:
        lemma = c.get("lemma", "")
        pos_list = lemma_pos_per_example.get(lemma)
        if not pos_list:
            continue
        card_pos = _normalize_card_pos(c.get("part_of_speech", ""))
        if not card_pos:
            continue
        indices = c.get("selected_example_indices") or []
        selected_pos: set[str] = set()
        for i in indices:
            if isinstance(i, int) and 1 <= i <= len(pos_list):
                selected_pos.add(pos_list[i - 1].upper() if isinstance(pos_list[i - 1], str) else "")
        if card_pos not in selected_pos:
            failing.append(c)
    return failing


def _normalize_definition_for_similarity(text: str) -> str:
    """Normalize definition_en for duplicate detection: lower, strip, collapse whitespace."""
    if not text or not isinstance(text, str):
        return ""
    s = text.strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def _definition_similarity(a: str, b: str) -> float:
    """Similarity of two definition strings in [0, 1]. Uses SequenceMatcher (gestalt)."""
    na, nb = _normalize_definition_for_similarity(a), _normalize_definition_for_similarity(b)
    if not na and not nb:
        return 1.0
    if not na or not nb:
        return 0.0
    return SequenceMatcher(None, na, nb).ratio()


# Default threshold for duplicate_sense: above this ratio definitions are considered duplicates
DEFAULT_DUPLICATE_SENSE_THRESHOLD = 0.85


def get_cards_failing_duplicate_sense(
    all_cards: list[dict],
    *,
    threshold: float = DEFAULT_DUPLICATE_SENSE_THRESHOLD,
) -> list[dict]:
    """Cards that are duplicate senses: same lemma, definition_en too similar to another card.
    For each lemma with >=2 cards, pairwise similarity (normalized definition_en) >= threshold
    marks the later card(s) as duplicate (we keep the first of each duplicate cluster).
    Returns list of card dicts to drop in strict mode.
    """
    from collections import defaultdict

    by_lemma: dict[str, list[dict]] = defaultdict(list)
    for c in all_cards:
        lemma = c.get("lemma", "")
        if lemma:
            by_lemma[lemma].append(c)

    failing: list[dict] = []
    for _lemma, cards in by_lemma.items():
        if len(cards) < 2:
            continue
        # Keep first of each similarity cluster; mark rest as duplicate
        kept: list[dict] = []
        for c in cards:
            def_en = (c.get("definition_en") or "").strip()
            is_dup = False
            for k in kept:
                if _definition_similarity(def_en, k.get("definition_en") or "") >= threshold:
                    is_dup = True
                    break
            if is_dup:
                failing.append(c)
            else:
                kept.append(c)
    return failing


def check_qc_threshold(
    issue_count: int,
    total_count: int,
    *,
    strict: bool = True,
    max_warning_rate: float | None = None,
    max_warnings_absolute: int | None = None,
    label: str = "QC",
) -> None:
    """Raise ValueError if issue_count exceeds thresholds. In strict, any threshold hit fails.
    total_count is used for rate (issue_count / total_count). Pass total_cards or total_examples.
    """
    if total_count <= 0:
        return
    rate = issue_count / total_count
    if max_warning_rate is not None and rate > max_warning_rate:
        raise ValueError(
            f"{label}: issue count {issue_count} / {total_count} = {rate:.2%} exceeds max_warning_rate {max_warning_rate:.2%}. "
            "Fix data or increase threshold."
        )
    if max_warnings_absolute is not None and issue_count > max_warnings_absolute:
        raise ValueError(
            f"{label}: issue count {issue_count} exceeds max_warnings_absolute {max_warnings_absolute}. "
            "Fix data or increase threshold."
        )
