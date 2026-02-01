"""Data directory paths and file path builders.

All data paths should be imported from here to avoid magic strings.
"""

from pathlib import Path

# Base directories (relative to project root)
DATA_DIR = Path("data")
DATA_RAW_DIR = DATA_DIR / "raw"
DATA_PROCESSED_DIR = DATA_DIR / "processed"
DATA_WSD_GOLD_DIR = DATA_DIR / "wsd_gold"

# Output directories
ANKI_EXPORTS_DIR = Path("anki_exports")
LOGS_DIR = Path("logs")

# Cache directories
LLM_CACHE_DIR = DATA_DIR / "llm_cache"
LLM_RESPONSE_CACHE_DIR = DATA_DIR / "cache" / "llm_responses"


# === File path builders ===


def get_book_path(book_name: str) -> Path:
    """Get path to raw EPUB file."""
    return DATA_RAW_DIR / f"{book_name}.epub"


def get_tokens_path(book_name: str) -> Path:
    """Get path to tokenized book file."""
    return DATA_PROCESSED_DIR / f"{book_name}_tokens.parquet"


def get_lemma_stats_path(book_name: str) -> Path:
    """Get path to lemma statistics file."""
    return DATA_PROCESSED_DIR / f"{book_name}_lemma_stats.parquet"


def get_lemma_stats_full_path(book_name: str) -> Path:
    """Get path to full lemma statistics file."""
    return DATA_PROCESSED_DIR / f"{book_name}_lemma_stats_full.parquet"


def get_phrasal_verbs_path(book_name: str) -> Path:
    """Get path to phrasal verbs file."""
    return DATA_PROCESSED_DIR / f"{book_name}_phrasal_verbs.parquet"


def get_anki_export_path(book_name: str) -> Path:
    """Get path to Anki CSV export file."""
    return ANKI_EXPORTS_DIR / f"{book_name}_anki.csv"


# === WSD Gold dataset paths ===


def get_gold_dev_path() -> Path:
    """Get path to gold dev dataset."""
    return DATA_WSD_GOLD_DIR / "gold_dev.jsonl"


def get_gold_test_locked_path() -> Path:
    """Get path to locked gold test dataset."""
    return DATA_WSD_GOLD_DIR / "gold_test_locked.jsonl"


def get_gold_checksum_path() -> Path:
    """Get path to gold test checksum."""
    return DATA_WSD_GOLD_DIR / "gold_test_locked.sha256"


def get_gold_labels_dir() -> Path:
    """Get directory for gold labels."""
    return DATA_WSD_GOLD_DIR / "labels"


def get_gold_examples_path() -> Path:
    """Get path to gold examples file."""
    return DATA_WSD_GOLD_DIR / "examples_all.jsonl"


def get_gold_labels_final_path() -> Path:
    """Get path to final gold labels."""
    return DATA_WSD_GOLD_DIR / "gold_labels_final.jsonl"


# === Logging ===


def get_log_path(name: str) -> Path:
    """Get path to log file."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    return LOGS_DIR / f"{name}.log"
