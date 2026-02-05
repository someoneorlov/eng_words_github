"""Pipeline B batch artifact schemas and paths (Stage 2).

Contracts for batch_info.json, cards_B_batch.json, download_log.json.
Backward compatible read (old files without schema_version → "0"), write always schema_version="1".
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


# --- Paths ---

@dataclass(frozen=True)
class BatchPaths:
    """Paths for a batch run under a given batch_dir."""

    requests: Path
    results: Path
    lemma_examples: Path
    batch_info: Path
    download_log: Path
    retry_cache: Path

    @classmethod
    def from_dir(cls, batch_dir: Path) -> BatchPaths:
        batch_dir = Path(batch_dir)
        return cls(
            requests=batch_dir / "requests.jsonl",
            results=batch_dir / "results.jsonl",
            lemma_examples=batch_dir / "lemma_examples.json",
            batch_info=batch_dir / "batch_info.json",
            download_log=batch_dir / "download_log.json",
            retry_cache=batch_dir / "retry_cache.jsonl",
        )


# --- Retry triggers (contract: use these names in only_if and in logs) ---

JSON_PARSE_ERROR = "json_parse_error"
SCHEMA_ERROR = "schema_error"  # missing_required_fields, validation
EMPTY_SELECTED_EXAMPLE_INDICES = "empty_selected_example_indices"
OUT_OF_RANGE_INDICES = "out_of_range_indices"
EXAMPLES_FALLBACK_USED = "examples_fallback_used"

DEFAULT_RETRY_TRIGGERS = (EMPTY_SELECTED_EXAMPLE_INDICES, EXAMPLES_FALLBACK_USED)


@dataclass
class RetryPolicy:
    """When to retry and how. only_if: trigger names; modes: standard and/or thinking."""

    modes: list[str]  # e.g. ["standard", "thinking"]
    only_if: list[str]  # trigger names from RETRY_TRIGGERS
    max_attempts: int = 2

    @classmethod
    def default(cls) -> RetryPolicy:
        return cls(modes=["standard"], only_if=list(DEFAULT_RETRY_TRIGGERS), max_attempts=2)


# --- Config (for future use) ---

@dataclass
class BatchConfig:
    """Configuration for a batch run (paths + limits + flags)."""

    tokens_path: Path
    sentences_path: Path
    batch_dir: Path
    output_cards_path: Path
    limit: int | None = None
    max_examples: int = 50
    model: str = "gemini-3-flash-preview"
    retry_policy: str | None = None
    strict: bool = True
    max_warning_rate: float | None = None
    max_warnings_absolute: int | None = None


# --- BatchInfo (batch_info.json) ---

CURRENT_BATCH_INFO_SCHEMA = "1"


@dataclass
class BatchInfo:
    """Schema for batch_info.json."""

    schema_version: str
    batch_name: str | None
    model: str
    lemmas_count: int
    created_at: str
    uploaded_file: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "batch_name": self.batch_name,
            "model": self.model,
            "lemmas_count": self.lemmas_count,
            "created_at": self.created_at,
            "uploaded_file": self.uploaded_file,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BatchInfo:
        # Backward compat: missing schema_version → "0"
        schema_version = data.get("schema_version", "0")
        return cls(
            schema_version=schema_version,
            batch_name=data.get("batch_name"),
            model=data.get("model", ""),
            lemmas_count=data.get("lemmas_count", 0),
            created_at=data.get("created_at", ""),
            uploaded_file=data.get("uploaded_file"),
        )


def read_batch_info(path: Path) -> BatchInfo:
    """Load batch_info.json. Backward compatible: missing schema_version treated as \"0\"."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Batch info not found: {path}")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return BatchInfo.from_dict(data)


def write_batch_info(path: Path, info: BatchInfo) -> None:
    """Write batch_info.json. Always writes schema_version=\"1\" (current)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = info.to_dict()
    data["schema_version"] = CURRENT_BATCH_INFO_SCHEMA
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# --- cards_B_batch.json (backward compatible read/write) ---

CURRENT_CARDS_SCHEMA = "1"


def read_cards_output(path: Path) -> dict[str, Any]:
    """Load cards_B_batch.json. Backward compat: missing schema_version → \"0\"."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Cards file not found: {path}")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if "schema_version" not in data:
        data["schema_version"] = "0"
    return data


def write_cards_output(path: Path, data: dict[str, Any]) -> None:
    """Write cards_B_batch.json. Always writes schema_version=\"1\"."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = dict(data)
    data["schema_version"] = CURRENT_CARDS_SCHEMA
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# --- ErrorEntry (download_log / errors) ---

@dataclass
class ErrorEntry:
    """Single error record (e.g. parse error per lemma)."""

    lemma: str
    stage: str
    error_type: str
    message: str


# --- CardRecord (one card in cards_B_batch.json) ---

@dataclass
class CardRecord:
    """One card in output (minimal contract for schema)."""

    lemma: str
    pos: str
    definitions: dict[str, str]  # e.g. {"en": "...", "ru": "..."}
    examples: list[str]
    indices: list[int]
    source: str
    warnings: list[str] | None = None


__all__ = [
    "BatchConfig",
    "BatchInfo",
    "BatchPaths",
    "CardRecord",
    "DEFAULT_RETRY_TRIGGERS",
    "EMPTY_SELECTED_EXAMPLE_INDICES",
    "ErrorEntry",
    "EXAMPLES_FALLBACK_USED",
    "JSON_PARSE_ERROR",
    "OUT_OF_RANGE_INDICES",
    "RetryPolicy",
    "SCHEMA_ERROR",
    "read_batch_info",
    "read_cards_output",
    "write_batch_info",
    "write_cards_output",
]
