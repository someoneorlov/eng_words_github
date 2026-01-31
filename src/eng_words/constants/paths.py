"""Data directory paths and file path builders.

All data paths should be imported from here to avoid magic strings.
"""

from pathlib import Path

# Base directories (relative to project root)
DATA_DIR = Path("data")
DATA_RAW_DIR = DATA_DIR / "raw"
DATA_PROCESSED_DIR = DATA_DIR / "processed"
DATA_SYNSET_CARDS_DIR = DATA_DIR / "synset_cards"
DATA_SYNSET_AGGREGATION_DIR = DATA_DIR / "synset_aggregation_full"
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


def get_sense_tokens_path(book_name: str) -> Path:
    """Get path to WSD-annotated tokens file."""
    return DATA_PROCESSED_DIR / f"{book_name}_wsd_sense_tokens.parquet"


def get_supersense_stats_path(book_name: str) -> Path:
    """Get path to supersense statistics file."""
    return DATA_PROCESSED_DIR / f"{book_name}_supersense_stats.parquet"


def get_phrasal_verbs_path(book_name: str) -> Path:
    """Get path to phrasal verbs file."""
    return DATA_PROCESSED_DIR / f"{book_name}_phrasal_verbs.parquet"


def get_anki_export_path(book_name: str) -> Path:
    """Get path to Anki CSV export file."""
    return ANKI_EXPORTS_DIR / f"{book_name}_anki.csv"


# === Synset aggregation paths ===


def get_aggregated_cards_path() -> Path:
    """Get path to aggregated cards parquet."""
    return DATA_SYNSET_AGGREGATION_DIR / "aggregated_cards.parquet"


def get_synset_stats_path() -> Path:
    """Get path to synset stats parquet."""
    return DATA_SYNSET_AGGREGATION_DIR / "synset_stats.parquet"


def get_aggregation_cache_dir() -> Path:
    """Get path to LLM cache for aggregation."""
    return DATA_SYNSET_AGGREGATION_DIR / "llm_cache"


# === Card generation paths ===


def get_card_generation_output_dir() -> Path:
    """Get output directory for card generation."""
    return DATA_SYNSET_CARDS_DIR


def get_card_generation_cache_dir() -> Path:
    """Get LLM cache directory for card generation."""
    return DATA_SYNSET_CARDS_DIR / "llm_cache"


def get_smart_cards_partial_path() -> Path:
    """Get path to partial (checkpoint) smart cards."""
    return DATA_SYNSET_CARDS_DIR / "synset_smart_cards_partial.json"


def get_smart_cards_final_path() -> Path:
    """Get path to final smart cards."""
    return DATA_SYNSET_CARDS_DIR / "synset_smart_cards_final.json"


def get_anki_csv_path() -> Path:
    """Get path to final Anki CSV."""
    return DATA_SYNSET_CARDS_DIR / "synset_anki.csv"


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


def get_generation_log_path() -> Path:
    """Get path to generation log file."""
    return DATA_SYNSET_CARDS_DIR / "full_generation.log"
