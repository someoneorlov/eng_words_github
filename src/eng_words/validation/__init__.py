"""Validation module for card quality checks."""

from eng_words.validation.example_validator import (
    ValidationResult,
    _get_word_forms,
    _word_in_text,
)

__all__ = [
    "ValidationResult",
    "_get_word_forms",
    "_word_in_text",
]
