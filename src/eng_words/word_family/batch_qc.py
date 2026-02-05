"""Pipeline B batch — QC checks and threshold gates.

lemma_not_in_example: cards where example does not contain lemma (or valid form).
Thresholds: in strict mode fail when exceeded; in relaxed record warnings with limits.
"""

from __future__ import annotations

from typing import Any

# Optional: use example_validator for word forms and whole-word check
try:
    from eng_words.validation.example_validator import _get_word_forms, _word_in_text
except ImportError:
    _get_word_forms = _word_in_text = None

# Homograph forms to exclude (e.g. hop vs hope)
DEFAULT_HOMONYM_EXCLUDE: dict[str, set[str]] = {
    "hop": {"hoped", "hoping"},
    "hope": {"hopped", "hopping"},
}


def cards_lemma_not_in_example(
    all_cards: list[dict],
    homonym_exclude: dict[str, set[str]] | None = None,
) -> list[dict[str, Any]]:
    """Cards where at least one example does not contain the lemma (or a valid word form).
    Returns list of {lemma, definition_en, example_index, example_preview} for review.
    If example_validator is not available, returns [].
    """
    if _get_word_forms is None or _word_in_text is None:
        return []
    exclude = homonym_exclude or DEFAULT_HOMONYM_EXCLUDE
    out: list[dict[str, Any]] = []
    for c in all_cards:
        lemma = c.get("lemma", "")
        examples = c.get("examples") or []
        if not examples:
            continue
        forms = _get_word_forms(lemma) - exclude.get(lemma.lower(), set())
        for i, ex in enumerate(examples):
            if not any(_word_in_text(f, ex) for f in forms):
                out.append({
                    "lemma": lemma,
                    "definition_en": (c.get("definition_en") or "")[:80],
                    "example_index": i + 1,
                    "example_preview": (ex[:100] + "...") if len(ex) > 100 else ex,
                })
                break
    return out


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
