"""Tests for Pipeline B batch schemas and paths (Stage 2)."""

import json
from pathlib import Path

import pytest

from eng_words.word_family.batch_schemas import (
    BatchInfo,
    BatchPaths,
    CardRecord,
    ErrorEntry,
    read_batch_info,
    read_cards_output,
    write_batch_info,
    write_cards_output,
)


class TestBatchPaths:
    def test_from_dir_builds_all_paths(self, tmp_path):
        paths = BatchPaths.from_dir(tmp_path)
        assert paths.requests == tmp_path / "requests.jsonl"
        assert paths.results == tmp_path / "results.jsonl"
        assert paths.lemma_examples == tmp_path / "lemma_examples.json"
        assert paths.batch_info == tmp_path / "batch_info.json"
        assert paths.download_log == tmp_path / "download_log.json"
        assert paths.retry_cache == tmp_path / "retry_cache.jsonl"


class TestBatchInfo:
    def test_serialize_deserialize_roundtrip(self):
        info = BatchInfo(
            schema_version="1",
            batch_name="batches/xyz",
            model="gemini-3-flash-preview",
            lemmas_count=10,
            created_at="2026-02-01 12:00:00",
            uploaded_file="files/abc",
        )
        data = info.to_dict()
        assert data["schema_version"] == "1"
        assert data["batch_name"] == "batches/xyz"
        restored = BatchInfo.from_dict(data)
        assert restored.schema_version == info.schema_version
        assert restored.batch_name == info.batch_name
        assert restored.lemmas_count == info.lemmas_count

    def test_schema_version_required_on_write(self):
        info = BatchInfo(
            schema_version="1",
            batch_name="b",
            model="m",
            lemmas_count=0,
            created_at="",
            uploaded_file=None,
        )
        d = info.to_dict()
        assert "schema_version" in d
        assert d["schema_version"] == "1"

    def test_from_dict_requires_schema_version_after_migration(self):
        # Old file without schema_version: migrated to "0" then normalized to "1" when building BatchInfo
        old = {"batch_name": "b", "model": "m", "lemmas_count": 5, "created_at": "x"}
        info = BatchInfo.from_dict(old)
        assert info.schema_version == "0"


class TestReadWriteBatchInfo:
    def test_write_then_read_has_schema_version(self, tmp_path):
        path = tmp_path / "batch_info.json"
        info = BatchInfo(
            schema_version="1",
            batch_name="batches/x",
            model="gemini-3-flash",
            lemmas_count=3,
            created_at="2026-02-01 12:00:00",
            uploaded_file="files/y",
        )
        write_batch_info(path, info)
        raw = json.loads(path.read_text())
        assert raw["schema_version"] == "1"
        read_info = read_batch_info(path)
        assert read_info.schema_version == "1"
        assert read_info.batch_name == "batches/x"

    def test_read_old_file_without_schema_version_migrates(self, tmp_path):
        path = tmp_path / "batch_info.json"
        path.write_text(
            json.dumps({
                "batch_name": "batches/old",
                "model": "gemini-2",
                "lemmas_count": 100,
                "created_at": "2025-01-01 00:00:00",
                "uploaded_file": "files/old",
            }, indent=2)
        )
        info = read_batch_info(path)
        assert info.schema_version == "0"
        assert info.batch_name == "batches/old"
        assert info.lemmas_count == 100


class TestErrorEntryAndCardRecord:
    def test_error_entry_fields(self):
        e = ErrorEntry(lemma="run", stage="parse", error_type="json_error", message="Invalid JSON")
        assert e.lemma == "run"
        assert e.error_type == "json_error"

    def test_card_record_minimal(self):
        c = CardRecord(
            lemma="go",
            pos="verb",
            definitions={"en": "move", "ru": "идти"},
            examples=["I go home."],
            indices=[1],
            source="pipeline_b_batch",
        )
        assert c.lemma == "go"
        assert c.warnings is None or c.warnings == []


class TestReadWriteCardsOutput:
    def test_read_old_cards_without_schema_version_gets_zero(self, tmp_path):
        path = tmp_path / "cards.json"
        path.write_text(
            json.dumps({
                "pipeline": "B",
                "source": "batch_api",
                "timestamp": "2026-02-01T12:00:00",
                "cards": [{"lemma": "go", "definition_en": "move"}],
                "errors": [],
            }, indent=2)
        )
        data = read_cards_output(path)
        assert data["schema_version"] == "0"
        assert len(data["cards"]) == 1
        assert data["cards"][0]["lemma"] == "go"

    def test_write_cards_adds_schema_version_one(self, tmp_path):
        path = tmp_path / "cards.json"
        data = {
            "pipeline": "B",
            "source": "batch_api",
            "timestamp": "2026-02-01T12:00:00",
            "cards": [],
            "errors": [],
        }
        write_cards_output(path, data)
        raw = json.loads(path.read_text())
        assert raw["schema_version"] == "1"
        read_back = read_cards_output(path)
        assert read_back["schema_version"] == "1"
