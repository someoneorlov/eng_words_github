"""Pipeline B contract invariants: hard errors that must fail before writing output.

Hard errors (per PIPELINE_B_FIXES_PLAN):
1. invalid JSON / parse failure — handled in parse_one_result; callers pass already-parsed cards.
2. missing required card fields (lemma, definition_en, definition_ru, part_of_speech, examples, selected_example_indices)
3. selected_example_indices not int or out-of-range (1-based, must be in [1, len(lemma_examples[lemma])])
4. examples empty when required (strict)
5. batch_dir artifact missing: lemma_examples file absent when running parse-results

Actionable: every raise includes "How to fix" hint.
"""

from __future__ import annotations

from pathlib import Path

from eng_words.word_family.qc_types import ErrorType

REQUIRED_CARD_FIELDS = ("lemma", "definition_en", "definition_ru", "part_of_speech", "examples", "selected_example_indices")


class ContractInvariantError(ValueError):
    """Raised when a contract invariant is violated. Always includes actionable hint."""

    def __init__(self, error_type: ErrorType, message: str, hint: str) -> None:
        self.error_type = error_type
        self.hint = hint
        super().__init__(f"{message} How to fix: {hint}")


def assert_contract_invariants(
    cards: list[dict],
    lemma_examples: dict[str, list[str]],
    *,
    strict: bool = True,
    require_examples_non_empty: bool = True,
    lemma_examples_path: Path | None = None,
) -> None:
    """Raise ContractInvariantError if any invariant is violated. No-op if cards is empty.

    Args:
        cards: list of card dicts (already parsed, e.g. from parse_one_result).
        lemma_examples: lemma -> list of example strings (1-based indices refer to these).
        strict: if True, empty examples trigger error.
        require_examples_non_empty: if True (default), each card must have at least one example.
        lemma_examples_path: if set, require this path to exist (for parse-results flow).
    """
    if lemma_examples_path is not None:
        if not Path(lemma_examples_path).exists():
            raise ContractInvariantError(
                ErrorType.CONTRACT_BATCH_ARTIFACTS,
                f"lemma_examples file not found: {lemma_examples_path}.",
                "Run render_requests first to create lemma_examples.json, or fix batch_dir.",
            )

    for i, card in enumerate(cards):
        if not isinstance(card, dict):
            raise ContractInvariantError(
                ErrorType.CONTRACT_SCHEMA,
                f"Card at index {i} is not a dict.",
                "Parser must return list of dicts; check LLM response format.",
            )

        for key in REQUIRED_CARD_FIELDS:
            if key not in card:
                raise ContractInvariantError(
                    ErrorType.CONTRACT_SCHEMA,
                    f"Card (lemma={card.get('lemma', '?')}) missing required field '{key}'.",
                    "Ensure prompt asks for all fields; check parse_one_result output.",
                )

        lemma = card.get("lemma", "")
        examples_list = lemma_examples.get(lemma, [])
        indices = card.get("selected_example_indices", [])

        if not all(isinstance(j, int) for j in indices):
            raise ContractInvariantError(
                ErrorType.CONTRACT_INDEX_MAPPING,
                f"Card lemma={lemma}: selected_example_indices must be list of int, got {indices}.",
                "Use 1-based indices (1 = first example). Normalize in parser if LLM returns 0-based.",
            )

        n = len(examples_list)
        for j in indices:
            if j < 1 or j > n:
                raise ContractInvariantError(
                    ErrorType.CONTRACT_INDEX_MAPPING,
                    f"Card lemma={lemma}: index {j} out of range (1..{n}).",
                    "selected_example_indices must be 1-based and within lemma example count.",
                )

        ex = card.get("examples") or []
        if not isinstance(ex, list):
            raise ContractInvariantError(
                ErrorType.CONTRACT_SCHEMA,
                f"Card lemma={lemma}: 'examples' must be a list, got {type(ex).__name__}.",
                "Parser must set examples from lemma_examples using selected_example_indices.",
            )

        if strict or require_examples_non_empty:
            if not ex:
                raise ContractInvariantError(
                    ErrorType.CONTRACT_EMPTY_EXAMPLES,
                    f"Card lemma={lemma}: examples are empty (strict/require_examples_non_empty).",
                    "Retry with valid indices or drop card; do not write cards with empty examples in strict mode.",
                )
