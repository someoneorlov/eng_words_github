#!/usr/bin/env python3
"""Collect full gold dataset from multiple books.

This script:
1. Processes books through WSD pipeline (if not already done)
2. Extracts WSD examples from each book
3. Applies stratified sampling
4. Splits into dev (25%) and test_locked (75%)

Usage:
    uv run python scripts/collect_full_gold_dataset.py --help
    uv run python scripts/collect_full_gold_dataset.py --list-books
    uv run python scripts/collect_full_gold_dataset.py --dry-run
    uv run python scripts/collect_full_gold_dataset.py --target-n 3000
"""

import hashlib
import json
import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import typer

from eng_words.wsd_gold import (
    GoldExample,
    calculate_difficulty_features,
    classify_difficulty,
    extract_examples_from_tokens,
    get_sampling_stats,
    split_by_source,
    stratified_sample,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

app = typer.Typer(help="Collect full gold dataset from multiple books")

# Book configurations
# Note: Pipeline outputs both regular and _wsd_ prefixed files
# We prefer _wsd_ versions when available, fall back to regular
BOOKS = {
    "american_tragedy": {
        "raw_path": "data/raw/theodore-dreiser_an-american-tragedy.epub",
        "bucket": "classic_fiction",
        "tokens_patterns": [
            "american_tragedy_wsd_tokens.parquet",
            "american_tragedy_tokens.parquet",
        ],
        "sense_tokens_patterns": [
            "american_tragedy_wsd_sense_tokens.parquet",
            "american_tragedy_sense_tokens.parquet",
        ],
    },
    "game_of_thrones": {
        "raw_path": "data/raw/Game_of_Thrones_1.epub",
        "bucket": "modern_fiction",
        "tokens_patterns": [
            "game_of_thrones_wsd_tokens.parquet",
            "game_of_thrones_tokens.parquet",
        ],
        "sense_tokens_patterns": [
            "game_of_thrones_wsd_sense_tokens.parquet",
            "game_of_thrones_sense_tokens.parquet",
        ],
    },
    "on_the_edge": {
        "raw_path": "data/raw/On_the_Edge.epub",
        "bucket": "modern_nonfiction",
        "tokens_patterns": [
            "on_the_edge_wsd_tokens.parquet",
            "on_the_edge_tokens.parquet",
        ],
        "sense_tokens_patterns": [
            "on_the_edge_wsd_sense_tokens.parquet",
            "on_the_edge_sense_tokens.parquet",
        ],
    },
    "lever_of_riches": {
        "raw_path": "data/raw/2008_Joel_Mokyr_The_Lever_of_Riches_Technological_Creativity_and_Economic_Progress_Oxford_University_Press.epub",
        "bucket": "academic_nonfiction",
        "tokens_patterns": [
            "lever_of_riches_wsd_tokens.parquet",
            "lever_of_riches_tokens.parquet",
        ],
        "sense_tokens_patterns": [
            "lever_of_riches_wsd_sense_tokens.parquet",
            "lever_of_riches_sense_tokens.parquet",
        ],
    },
}


@dataclass
class BookStatus:
    """Status of a book's WSD processing."""

    name: str
    raw_exists: bool
    wsd_processed: bool
    tokens_path: Path | None
    sense_tokens_path: Path | None
    example_count: int = 0


def check_book_status(book_name: str, processed_dir: Path) -> BookStatus:
    """Check if a book has been processed through WSD."""
    config = BOOKS[book_name]

    raw_exists = Path(config["raw_path"]).exists()

    # Find first existing tokens file
    tokens_path = None
    for pattern in config["tokens_patterns"]:
        candidate = processed_dir / pattern
        if candidate.exists():
            tokens_path = candidate
            break

    # Find first existing sense_tokens file
    sense_tokens_path = None
    for pattern in config["sense_tokens_patterns"]:
        candidate = processed_dir / pattern
        if candidate.exists():
            sense_tokens_path = candidate
            break

    wsd_processed = tokens_path is not None and sense_tokens_path is not None

    return BookStatus(
        name=book_name,
        raw_exists=raw_exists,
        wsd_processed=wsd_processed,
        tokens_path=tokens_path,
        sense_tokens_path=sense_tokens_path,
    )


def run_wsd_pipeline(book_name: str, config: dict, output_dir: Path) -> bool:
    """Run WSD pipeline for a book."""
    raw_path = Path(config["raw_path"])
    if not raw_path.exists():
        logger.error(f"Raw file not found: {raw_path}")
        return False

    cmd = [
        "uv",
        "run",
        "python",
        "-m",
        "eng_words.pipeline",
        "--book-path",
        str(raw_path),
        "--book-name",
        book_name,
        "--output-dir",
        str(output_dir),
        "--enable-wsd",
    ]

    logger.info(f"Running WSD pipeline for {book_name}...")
    logger.info(f"Command: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        logger.info(f"WSD pipeline completed for {book_name}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"WSD pipeline failed for {book_name}: {e.stderr}")
        return False


def load_and_extract_examples(
    book_name: str,
    tokens_path: Path,
    sense_tokens_path: Path,
    bucket: str,
) -> list[GoldExample]:
    """Load processed data and extract WSD examples."""
    logger.info(f"Loading data for {book_name}...")

    tokens_df = pd.read_parquet(tokens_path)
    sense_tokens_df = pd.read_parquet(sense_tokens_path)

    # Reconstruct sentences_df from tokens
    # Simple join with spaces (whitespace column may have special chars)
    # Column must be named 'sentence' for extract_examples_from_tokens
    sentences_df = (
        tokens_df.groupby("sentence_id")
        .agg(
            sentence=("surface", lambda x: " ".join(str(s) for s in x)),
            book_id=("book", "first"),
        )
        .reset_index()
    )

    # Build source metadata
    source_metadata = {
        "source_id": book_name,
        "source_bucket": bucket,
        "book_title": book_name.replace("_", " ").title(),
    }

    # Extract examples
    # sense_tokens_df contains the synset_id column needed for WSD examples
    examples = extract_examples_from_tokens(
        tokens_df=sense_tokens_df,  # Use sense_tokens (has synset_id)
        sentences_df=sentences_df,
        source_metadata=source_metadata,
    )

    logger.info(f"Extracted {len(examples)} examples from {book_name}")
    return examples


def calculate_file_checksum(path: Path) -> str:
    """Calculate SHA256 checksum of a file."""
    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


@app.command("list-books")
def list_books(
    processed_dir: Path = typer.Option(
        Path("data/processed"),
        help="Directory with processed data",
    ),
) -> None:
    """List available books and their processing status."""
    print("\n" + "=" * 70)
    print("📚 BOOK STATUS")
    print("=" * 70 + "\n")

    for book_name in BOOKS:
        status = check_book_status(book_name, processed_dir)
        config = BOOKS[book_name]

        raw_icon = "✅" if status.raw_exists else "❌"
        wsd_icon = "✅" if status.wsd_processed else "❌"

        print(f"{book_name}:")
        print(f"  Bucket: {config['bucket']}")
        print(f"  Raw file: {raw_icon} {config['raw_path']}")
        print(f"  WSD processed: {wsd_icon}")
        print()


@app.command("process")
def process_books(
    processed_dir: Path = typer.Option(
        Path("data/processed"),
        help="Directory with processed data",
    ),
    force: bool = typer.Option(
        False,
        help="Force re-processing even if already done",
    ),
) -> None:
    """Process books through WSD pipeline."""
    for book_name, config in BOOKS.items():
        status = check_book_status(book_name, processed_dir)

        if not status.raw_exists:
            logger.warning(f"Skipping {book_name}: raw file not found")
            continue

        if status.wsd_processed and not force:
            logger.info(f"Skipping {book_name}: already processed")
            continue

        success = run_wsd_pipeline(book_name, config, processed_dir)
        if not success:
            raise typer.Exit(1)


@app.command("collect")
def collect_dataset(
    output_dir: Path = typer.Option(
        Path("data/wsd_gold"),
        help="Output directory",
    ),
    processed_dir: Path = typer.Option(
        Path("data/processed"),
        help="Directory with processed data",
    ),
    target_n: int = typer.Option(
        3000,
        help="Target number of examples",
    ),
    dev_ratio: float = typer.Option(
        0.25,
        help="Ratio for dev set (rest goes to test_locked)",
    ),
    dry_run: bool = typer.Option(
        False,
        help="Only show statistics, don't save",
    ),
) -> None:
    """Collect gold dataset from processed books."""
    # Check which books are available
    available_books = []
    for book_name in BOOKS:
        status = check_book_status(book_name, processed_dir)
        if status.wsd_processed:
            available_books.append(book_name)
        else:
            logger.warning(f"Skipping {book_name}: not WSD processed")

    if not available_books:
        logger.error("No books available! Run 'process' command first.")
        raise typer.Exit(1)

    logger.info(f"Available books: {available_books}")

    # Calculate per-book quota
    per_book = target_n // len(available_books)
    remainder = target_n % len(available_books)

    logger.info(f"Target: {target_n} examples ({per_book} per book + {remainder})")

    # Collect examples from each book
    all_examples: list[GoldExample] = []

    for i, book_name in enumerate(available_books):
        status = check_book_status(book_name, processed_dir)
        config = BOOKS[book_name]

        examples = load_and_extract_examples(
            book_name=book_name,
            tokens_path=status.tokens_path,
            sense_tokens_path=status.sense_tokens_path,
            bucket=config["bucket"],
        )

        # Calculate difficulty features
        for ex in examples:
            features = calculate_difficulty_features(ex)
            ex.metadata.difficulty = classify_difficulty(
                sense_count=features.wn_sense_count,
                margin=features.baseline_margin,
            )

        # Target for this book (add remainder to last book)
        book_target = per_book + (remainder if i == len(available_books) - 1 else 0)

        # Stratified sample
        if len(examples) > book_target:
            sampled = stratified_sample(
                examples=examples,
                n=book_target,
            )
            logger.info(f"Sampled {len(sampled)} from {len(examples)} ({book_name})")
        else:
            sampled = examples
            logger.warning(
                f"Only {len(examples)} examples available for {book_name} (target: {book_target})"
            )

        all_examples.extend(sampled)

    logger.info(f"Total examples: {len(all_examples)}")

    # Get statistics
    stats = get_sampling_stats(all_examples)

    print("\n" + "=" * 70)
    print("📊 DATASET STATISTICS")
    print("=" * 70)
    print(f"\nTotal examples: {stats['total']}")
    print("\nBy bucket:")
    for bucket, count in stats.get("by_bucket", {}).items():
        pct = 100 * count / stats["total"]
        print(f"  {bucket}: {count} ({pct:.0f}%)")
    print("\nBy difficulty:")
    for diff, count in stats.get("by_difficulty", {}).items():
        pct = 100 * count / stats["total"]
        print(f"  {diff}: {count} ({pct:.0f}%)")
    print("\nBy POS:")
    for pos, count in stats.get("by_pos", {}).items():
        pct = 100 * count / stats["total"]
        print(f"  {pos}: {count} ({pct:.0f}%)")

    if dry_run:
        print("\n[DRY RUN] Не сохраняем файлы")
        return

    # Split into dev and test_locked
    dev_examples, test_examples = split_by_source(
        examples=all_examples,
        dev_ratio=dev_ratio,
    )

    logger.info(f"Split: dev={len(dev_examples)}, test_locked={len(test_examples)}")

    # Save examples
    output_dir.mkdir(parents=True, exist_ok=True)

    # All examples (before split)
    all_path = output_dir / "examples_all.jsonl"
    with open(all_path, "w") as f:
        for ex in all_examples:
            f.write(json.dumps(ex.to_dict()) + "\n")
    logger.info(f"Saved {len(all_examples)} examples to {all_path}")

    # Dev examples
    dev_path = output_dir / "examples_dev.jsonl"
    with open(dev_path, "w") as f:
        for ex in dev_examples:
            f.write(json.dumps(ex.to_dict()) + "\n")
    logger.info(f"Saved {len(dev_examples)} dev examples to {dev_path}")

    # Test locked examples
    test_path = output_dir / "examples_test_locked.jsonl"
    with open(test_path, "w") as f:
        for ex in test_examples:
            f.write(json.dumps(ex.to_dict()) + "\n")
    logger.info(f"Saved {len(test_examples)} test_locked examples to {test_path}")

    # Generate checksums
    checksums = {
        "examples_all.jsonl": calculate_file_checksum(all_path),
        "examples_dev.jsonl": calculate_file_checksum(dev_path),
        "examples_test_locked.jsonl": calculate_file_checksum(test_path),
    }

    checksum_path = output_dir / "checksums.json"
    with open(checksum_path, "w") as f:
        json.dump(checksums, f, indent=2)
    logger.info(f"Saved checksums to {checksum_path}")

    print("\n" + "=" * 70)
    print("✅ DATASET COLLECTION COMPLETE")
    print("=" * 70)
    print(f"\nTotal: {len(all_examples)}")
    print(f"  Dev: {len(dev_examples)} ({100*len(dev_examples)/len(all_examples):.0f}%)")
    print(f"  Test locked: {len(test_examples)} ({100*len(test_examples)/len(all_examples):.0f}%)")
    print(f"\nOutput directory: {output_dir}")
    print("\n⚠️  ВАЖНО: examples_test_locked.jsonl НЕ ОТКРЫВАТЬ до финального тестирования!")


if __name__ == "__main__":
    app()

