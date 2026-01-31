#!/usr/bin/env python3
"""Freeze the gold dataset with dev/test split and checksums.

Creates:
- data/wsd_gold/gold_dev.jsonl - Development set (for tuning)
- data/wsd_gold/gold_test_locked.jsonl - Locked test set (never peek!)
- data/wsd_gold/gold_test_locked.sha256 - Checksum file

Usage:
    uv run python scripts/freeze_gold_dataset.py
"""

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import typer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

app = typer.Typer(help="Freeze gold dataset with dev/test split")

EXAMPLES_PATH = Path("data/wsd_gold/examples_all.jsonl")
LABELS_PATH = Path("data/wsd_gold/gold_labels_final.jsonl")

OUTPUT_DIR = Path("data/wsd_gold")
DEV_PATH = OUTPUT_DIR / "gold_dev.jsonl"
TEST_LOCKED_PATH = OUTPUT_DIR / "gold_test_locked.jsonl"
CHECKSUM_PATH = OUTPUT_DIR / "gold_test_locked.sha256"
MANIFEST_PATH = OUTPUT_DIR / "gold_manifest.json"

# Split by source: 2 books for dev, 2 books for test
DEV_SOURCES = ["american_tragedy", "on_the_edge"]
TEST_SOURCES = ["game_of_thrones", "lever_of_riches"]


def load_examples() -> dict:
    """Load examples as dict by ID."""
    examples = {}
    with open(EXAMPLES_PATH) as f:
        for line in f:
            data = json.loads(line)
            examples[data["example_id"]] = data
    return examples


def load_labels() -> dict:
    """Load labels as dict by ID."""
    labels = {}
    with open(LABELS_PATH) as f:
        for line in f:
            data = json.loads(line)
            labels[data["example_id"]] = data
    return labels


def get_source_from_id(example_id: str) -> str:
    """Extract source from example_id.

    Example: 'book:american_tragedy_wsd|sent:12008|tok:1' -> 'american_tragedy'
    """
    # Extract book part
    book_part = example_id.split("|")[0].replace("book:", "")
    # Remove _wsd suffix if present
    if book_part.endswith("_wsd"):
        book_part = book_part[:-4]
    return book_part


def compute_sha256(filepath: Path) -> str:
    """Compute SHA256 hash of file."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


@app.command()
def main(
    dry_run: bool = typer.Option(False, "--dry-run", help="Show split without saving"),
) -> None:
    """Freeze gold dataset with dev/test split."""
    # Load data
    examples = load_examples()
    labels = load_labels()

    logger.info(f"Loaded {len(examples)} examples, {len(labels)} labels")

    # Split by source
    dev_items = []
    test_items = []
    source_counts = {"dev": {}, "test": {}}

    for example_id, label in labels.items():
        if example_id not in examples:
            logger.warning(f"Label without example: {example_id}")
            continue

        example = examples[example_id]
        source = get_source_from_id(example_id)

        # Combine example and label
        combined = {
            **example,
            "gold_synset_id": label["synset_id"],
            "gold_confidence": label["confidence"],
            "gold_agreement": label["agreement_ratio"],
            "gold_flags": label["flags"],
        }

        if source in DEV_SOURCES or any(s in source for s in DEV_SOURCES):
            dev_items.append(combined)
            source_counts["dev"][source] = source_counts["dev"].get(source, 0) + 1
        elif source in TEST_SOURCES or any(s in source for s in TEST_SOURCES):
            test_items.append(combined)
            source_counts["test"][source] = source_counts["test"].get(source, 0) + 1
        else:
            logger.warning(f"Unknown source: {source} for {example_id}")

    # Print split statistics
    print("\n" + "=" * 60)
    print("📊 SPLIT STATISTICS")
    print("=" * 60)
    print(f"\nDEV SET ({len(dev_items)} examples):")
    for source, count in sorted(source_counts["dev"].items()):
        print(f"  - {source}: {count}")

    print(f"\nTEST SET (LOCKED) ({len(test_items)} examples):")
    for source, count in sorted(source_counts["test"].items()):
        print(f"  - {source}: {count}")

    if dry_run:
        print("\n[DRY RUN] Files not created")
        return

    # Save dev set
    with open(DEV_PATH, "w") as f:
        for item in dev_items:
            f.write(json.dumps(item) + "\n")
    logger.info(f"Saved dev set: {DEV_PATH}")

    # Save test set (locked)
    with open(TEST_LOCKED_PATH, "w") as f:
        for item in test_items:
            f.write(json.dumps(item) + "\n")
    logger.info(f"Saved test set: {TEST_LOCKED_PATH}")

    # Compute and save checksum
    checksum = compute_sha256(TEST_LOCKED_PATH)
    with open(CHECKSUM_PATH, "w") as f:
        f.write(f"{checksum}  gold_test_locked.jsonl\n")
    logger.info(f"Saved checksum: {CHECKSUM_PATH}")

    # Create manifest
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "version": "1.0.0",
        "dev_set": {
            "path": str(DEV_PATH),
            "count": len(dev_items),
            "sources": list(source_counts["dev"].keys()),
        },
        "test_locked": {
            "path": str(TEST_LOCKED_PATH),
            "count": len(test_items),
            "sources": list(source_counts["test"].keys()),
            "sha256": checksum,
        },
        "labeling": {
            "providers": ["anthropic", "gemini", "openai_referee"],
            "models": {
                "anthropic": "claude-opus-4-5-20251101",
                "gemini": "gemini-3-flash-preview",
                "openai": "gpt-5.2",
            },
            "agreement_rate": 0.87,
            "referee_rate": 0.13,
        },
        "notes": [
            "DEV: Use for development and tuning",
            "TEST: DO NOT LOOK until final evaluation!",
            "Split is by source book to prevent data leakage",
        ],
    }

    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"Saved manifest: {MANIFEST_PATH}")

    # Final summary
    print("\n" + "=" * 60)
    print("✅ GOLD DATASET FROZEN")
    print("=" * 60)
    print(f"  Dev set:      {DEV_PATH} ({len(dev_items)} examples)")
    print(f"  Test locked:  {TEST_LOCKED_PATH} ({len(test_items)} examples)")
    print(f"  Checksum:     {CHECKSUM_PATH}")
    print(f"  Manifest:     {MANIFEST_PATH}")
    print(f"\n  SHA256: {checksum[:32]}...")
    print("\n⚠️  WARNING: Do not peek at gold_test_locked.jsonl!")


if __name__ == "__main__":
    app()
