"""Tests for eng_words.word_family.contract (Stage 1.2 TDD)."""

from __future__ import annotations

from pathlib import Path

import pytest

from eng_words.word_family.contract import (
    ContractInvariantError,
    assert_contract_invariants,
)
from eng_words.word_family.qc_types import ErrorType


def test_contract_empty_cards_passes() -> None:
    assert_contract_invariants([], {})


def test_contract_lemma_examples_path_missing_raises(tmp_path: Path) -> None:
    missing = tmp_path / "nonexistent_lemma_examples.json"
    assert not missing.exists()
    with pytest.raises(ContractInvariantError) as exc_info:
        assert_contract_invariants([], {}, lemma_examples_path=missing)
    assert exc_info.value.error_type == ErrorType.CONTRACT_BATCH_ARTIFACTS
    assert "How to fix" in str(exc_info.value)


def test_contract_lemma_examples_path_exists_passes(tmp_path: Path) -> None:
    path = tmp_path / "lemma_examples.json"
    path.write_text("{}")
    assert_contract_invariants([], {}, lemma_examples_path=path)


def test_contract_card_not_dict_raises() -> None:
    with pytest.raises(ContractInvariantError) as exc_info:
        assert_contract_invariants([["not", "a", "dict"]], {})
    assert exc_info.value.error_type == ErrorType.CONTRACT_SCHEMA


def test_contract_missing_required_field_raises() -> None:
    card = {
        "lemma": "test",
        "definition_en": "x",
        "definition_ru": "y",
        "part_of_speech": "noun",
        "examples": ["a"],
        "selected_example_indices": [1],
    }
    lemma_examples = {"test": ["a"]}
    assert_contract_invariants([card], lemma_examples)
    bad = {k: v for k, v in card.items() if k != "definition_en"}
    with pytest.raises(ContractInvariantError) as exc_info:
        assert_contract_invariants([bad], lemma_examples)
    assert exc_info.value.error_type == ErrorType.CONTRACT_SCHEMA
    assert "definition_en" in str(exc_info.value)


def test_contract_indices_not_int_raises() -> None:
    card = {
        "lemma": "test",
        "definition_en": "x",
        "definition_ru": "y",
        "part_of_speech": "noun",
        "examples": ["a"],
        "selected_example_indices": ["1"],
    }
    lemma_examples = {"test": ["a"]}
    with pytest.raises(ContractInvariantError) as exc_info:
        assert_contract_invariants([card], lemma_examples)
    assert exc_info.value.error_type == ErrorType.CONTRACT_INDEX_MAPPING


def test_contract_index_out_of_range_raises() -> None:
    card = {
        "lemma": "test",
        "definition_en": "x",
        "definition_ru": "y",
        "part_of_speech": "noun",
        "examples": ["a"],
        "selected_example_indices": [2],
    }
    lemma_examples = {"test": ["only one"]}
    with pytest.raises(ContractInvariantError) as exc_info:
        assert_contract_invariants([card], lemma_examples)
    assert exc_info.value.error_type == ErrorType.CONTRACT_INDEX_MAPPING
    assert "out of range" in str(exc_info.value).lower()


def test_contract_examples_not_list_raises() -> None:
    card = {
        "lemma": "test",
        "definition_en": "x",
        "definition_ru": "y",
        "part_of_speech": "noun",
        "examples": "not a list",
        "selected_example_indices": [1],
    }
    lemma_examples = {"test": ["a"]}
    with pytest.raises(ContractInvariantError) as exc_info:
        assert_contract_invariants([card], lemma_examples)
    assert exc_info.value.error_type == ErrorType.CONTRACT_SCHEMA
    assert "must be a list" in str(exc_info.value)


def test_contract_empty_examples_strict_raises() -> None:
    card = {
        "lemma": "test",
        "definition_en": "x",
        "definition_ru": "y",
        "part_of_speech": "noun",
        "examples": [],
        "selected_example_indices": [],
    }
    lemma_examples = {"test": ["a"]}
    with pytest.raises(ContractInvariantError) as exc_info:
        assert_contract_invariants([card], lemma_examples, strict=True)
    assert exc_info.value.error_type == ErrorType.CONTRACT_EMPTY_EXAMPLES


def test_contract_valid_card_passes() -> None:
    card = {
        "lemma": "test",
        "definition_en": "x",
        "definition_ru": "y",
        "part_of_speech": "noun",
        "examples": ["sentence with test"],
        "selected_example_indices": [1],
    }
    lemma_examples = {"test": ["sentence with test"]}
    assert_contract_invariants([card], lemma_examples)
