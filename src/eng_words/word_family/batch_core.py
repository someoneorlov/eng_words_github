"""Pipeline B batch — pure core (no I/O, no network).

Functions here are deterministic and testable without mocks.
"""

from __future__ import annotations

import json

from eng_words.word_family import CLUSTER_PROMPT_TEMPLATE
from eng_words.word_family.batch_schemas import (
    ErrorEntry,
    EXAMPLES_FALLBACK_USED,
    EMPTY_SELECTED_EXAMPLE_INDICES,
    RetryPolicy,
)

REQUIRED_CARD_KEYS = {"definition_en", "definition_ru", "part_of_speech"}


def build_prompt(
    lemma: str,
    examples: list[str],
    pos_distribution: dict[str, int] | None = None,
) -> str:
    """Build Pipeline B cluster prompt for one lemma.
    If pos_distribution is provided (e.g. {"NOUN": 5, "VERB": 2}), appends a POS hint block.
    """
    numbered = "\n".join(f"{i+1}. {ex}" for i, ex in enumerate(examples))
    prompt = CLUSTER_PROMPT_TEMPLATE.format(lemma=lemma, numbered_examples=numbered)
    if pos_distribution:
        pos_hint = ", ".join(f"{pos} {count}" for pos, count in sorted(pos_distribution.items()))
        prompt += f"\n\nPOS distribution in the examples above: {pos_hint}."
    n = len(examples)
    if n > 0:
        prompt += (
            f"\n\nIMPORTANT: selected_example_indices must be 1-based (1 = first example, 2 = second, ...). "
            f"Use only integers from 1 to {n} — you have exactly {n} examples above."
        )
    return prompt


def build_retry_prompt(lemma: str, examples: list[str]) -> str:
    """Same as build_prompt plus a concatenated line asking to fix indices (for retry)."""
    prompt = build_prompt(lemma, examples)
    n = len(examples)
    if n > 0:
        prompt += (
            f"\n\nCRITICAL: Your previous response had invalid selected_example_indices. "
            f"Use only 1-based indices from 1 to {n} (you have exactly {n} examples above)."
        )
    return prompt


def extract_json_from_response_text(text: str) -> dict:
    """Extract JSON from response text (may be inside ```json block or have surrounding junk)."""
    if not text or not isinstance(text, str):
        raise ValueError("empty or non-string text")
    t = text.strip()
    if t.startswith("```json"):
        t = t[7:]
    if t.startswith("```"):
        t = t[3:]
    if t.endswith("```"):
        t = t[:-3]
    t = t.strip()
    try:
        return json.loads(t)
    except json.JSONDecodeError as e:
        raise ValueError(f"invalid JSON: {e}") from e


def parse_one_result(
    key: str,
    response: dict,
    lemma_examples: dict[str, list[str]],
) -> tuple[str, list[dict], str | None]:
    """Parse one batch result. Returns (lemma, cards, error). error is None on success."""
    if not key.startswith("lemma:"):
        return "", [], "bad_key"
    lemma = key[6:]

    if "candidates" not in response or not response["candidates"]:
        return lemma, [], "no_candidates"
    parts = response["candidates"][0].get("content", {}).get("parts", [])
    if not parts:
        return lemma, [], "empty_content"
    text = parts[0].get("text", "").strip()
    if not text:
        return lemma, [], "empty_content"

    try:
        data = extract_json_from_response_text(text)
    except ValueError as e:
        return lemma, [], f"json_error: {e}"

    cards = data.get("cards", [])
    examples = lemma_examples.get(lemma, [])

    for card in cards:
        if not isinstance(card, dict):
            continue
        card["lemma"] = lemma
        card["source"] = "pipeline_b_batch"
        raw = [i for i in card.get("selected_example_indices", []) if isinstance(i, int)]
        if not raw:
            idxs = []
        elif 0 in raw and max(raw) < len(examples):
            idxs = [i for i in raw if 0 <= i < len(examples)]
        else:
            idxs = [i - 1 for i in raw if 0 < i <= len(examples)]
        card["examples"] = [examples[j] for j in idxs]
        if not card["examples"] and examples:
            card["examples_fallback"] = True

    return lemma, cards, None


def validate_card(card: dict, lemma: str) -> list[str]:
    """Return list of validation error messages for a card (no exceptions)."""
    errs = []
    for k in REQUIRED_CARD_KEYS:
        if k not in card or not (card[k] and str(card[k]).strip()):
            errs.append(f"missing or empty '{k}'")
    if "selected_example_indices" not in card:
        errs.append("missing selected_example_indices")
    return errs


def filter_valid_cards(
    cards: list[dict],
    lemma: str,
    *,
    skip_validation: bool = False,
    stage: str = "download",
) -> tuple[list[dict], list[ErrorEntry]]:
    """Precision-first: return (valid_cards, error_entries). Invalid cards become ErrorEntry, not in output."""
    valid: list[dict] = []
    error_entries: list[ErrorEntry] = []
    for c in cards:
        if skip_validation:
            valid.append(c)
            continue
        ve = validate_card(c, lemma)
        if ve:
            error_entries.append(
                ErrorEntry(
                    lemma=lemma,
                    stage=stage,
                    error_type="validation",
                    message="; ".join(ve),
                )
            )
            continue
        valid.append(c)
    return valid, error_entries


def choose_retry_candidates(
    parsed_results: list[tuple[str, list[dict], str | None]],
    *,
    policy: RetryPolicy | None = None,
    empty_examples: bool = True,
    examples_fallback: bool = True,
) -> set[str]:
    """Return set of lemmas that should be retried (empty examples or fallback).
    If policy is given, only_if triggers are used; otherwise empty_examples/examples_fallback booleans.
    """
    use_empty = empty_examples
    use_fallback = examples_fallback
    if policy is not None:
        use_empty = EMPTY_SELECTED_EXAMPLE_INDICES in policy.only_if
        use_fallback = EXAMPLES_FALLBACK_USED in policy.only_if
    lemmas = set()
    for lemma, cards, err in parsed_results:
        if err:
            continue
        for c in cards:
            if use_empty and not c.get("examples"):
                lemmas.add(lemma)
                break
            if use_fallback and c.get("examples_fallback"):
                lemmas.add(lemma)
                break
    return lemmas


def merge_retry_results(
    base_cards: list[dict],
    retry_lemma: str,
    retry_cards: list[dict],
) -> list[dict]:
    """Replace all cards for retry_lemma in base_cards with retry_cards. Returns new list."""
    out = [c for c in base_cards if c.get("lemma") != retry_lemma]
    out.extend(retry_cards)
    return out


def compute_stats(
    all_cards: list[dict],
    errors: list[dict],
    lemmas_with_zero_cards: list[str],
) -> dict[str, int | list]:
    """Compute summary stats for download result."""
    return {
        "cards_generated": len(all_cards),
        "errors_count": len(errors),
        "lemmas_with_zero_cards": len(lemmas_with_zero_cards),
    }
