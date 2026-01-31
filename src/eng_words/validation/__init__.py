"""Validation module for card quality checks."""

from eng_words.validation.example_validator import (
    ValidationResult,
    fix_invalid_cards,
    validate_card_examples,
)
from eng_words.validation.synset_validator import validate_examples_for_synset_group

__all__ = [
    "ValidationResult",
    "validate_card_examples",
    "fix_invalid_cards",
    "validate_examples_for_synset_group",
]
