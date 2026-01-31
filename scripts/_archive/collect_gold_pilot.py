#!/usr/bin/env python3
"""Collect pilot batch for WSD Gold Dataset.

This script collects 50-100 diverse examples for piloting
the LLM labeling process.

Usage:
    uv run python scripts/collect_gold_pilot.py --help
    uv run python scripts/collect_gold_pilot.py
    uv run python scripts/collect_gold_pilot.py --size 100 --hard-ratio 0.3
"""

import json
import logging
from pathlib import Path

import pandas as pd
import typer

from eng_words.wsd_gold.collect import (
    extract_examples_from_tokens,
)
from eng_words.wsd_gold.models import GoldExample
from eng_words.wsd_gold.sample import (
    calculate_difficulty_features,
    classify_difficulty,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

app = typer.Typer(help="Collect pilot batch for WSD Gold Dataset")

# Default sources configuration
DEFAULT_SOURCES = {
    "american_tragedy": {
        "tokens_path": "data/processed/american_tragedy_wsd_sense_tokens.parquet",
        "bucket": "classic_fiction",
        "book_name": "An American Tragedy",
        "year_bucket": "pre_1950",
        "genre_bucket": "fiction",
    },
    # Future sources (when processed):
    # "game_of_thrones": {
    #     "tokens_path": "data/processed/game_of_thrones_wsd_sense_tokens.parquet",
    #     "bucket": "modern_fiction",
    #     "book_name": "A Game of Thrones",
    #     "year_bucket": "1950_2000",
    #     "genre_bucket": "fiction",
    # },
    # "on_the_edge": {
    #     "tokens_path": "data/processed/on_the_edge_wsd_sense_tokens.parquet",
    #     "bucket": "modern_nonfiction",
    #     "book_name": "On the Edge",
    #     "year_bucket": "post_2000",
    #     "genre_bucket": "nonfiction",
    # },
    # "lever_of_riches": {
    #     "tokens_path": "data/processed/lever_of_riches_wsd_sense_tokens.parquet",
    #     "bucket": "academic_nonfiction",
    #     "book_name": "The Lever of Riches",
    #     "year_bucket": "post_2000",
    #     "genre_bucket": "nonfiction",
    # },
}


def load_and_reconstruct_sentences(tokens_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load tokens and reconstruct sentences DataFrame.

    Args:
        tokens_path: Path to tokens parquet file

    Returns:
        Tuple of (tokens_df, sentences_df)
    """
    tokens_df = pd.read_parquet(tokens_path)

    # Filter tokens with valid synset_id
    tokens_df = tokens_df[tokens_df["synset_id"].notna() & (tokens_df["synset_id"] != "")].copy()

    # Reconstruct sentences
    all_tokens = pd.read_parquet(tokens_path)
    sentences = (
        all_tokens.groupby("sentence_id")
        .apply(
            lambda g: "".join(
                row["surface"] + row["whitespace"] for _, row in g.iterrows()
            ).strip(),
            include_groups=False,
        )
        .reset_index()
    )
    sentences.columns = ["sentence_id", "sentence"]

    return tokens_df, sentences


def collect_examples_from_source(
    source_id: str,
    config: dict,
    max_examples: int = 50,
    random_state: int = 42,
) -> list[GoldExample]:
    """Collect examples from a single source.

    Args:
        source_id: Source identifier
        config: Source configuration dict
        max_examples: Maximum examples to collect from this source
        random_state: Random seed

    Returns:
        List of GoldExample instances
    """
    tokens_path = Path(config["tokens_path"])
    if not tokens_path.exists():
        logger.warning(f"Source {source_id} not found: {tokens_path}")
        return []

    logger.info(f"Loading {source_id} from {tokens_path}")

    # Load and reconstruct
    tokens_df, sentences_df = load_and_reconstruct_sentences(tokens_path)
    logger.info(f"  Loaded {len(tokens_df)} WSD tokens, {len(sentences_df)} sentences")

    # Get book name from tokens
    book_name = tokens_df["book"].iloc[0] if "book" in tokens_df.columns else source_id

    # Build source_metadata in the expected format: {book_name: {source_bucket: ...}}
    source_meta_dict = {
        book_name: {
            "source_bucket": config["bucket"],
            "year_bucket": config.get("year_bucket", "unknown"),
            "genre_bucket": config.get("genre_bucket", "unknown"),
        }
    }

    # Extract examples
    examples = extract_examples_from_tokens(
        tokens_df=tokens_df,
        sentences_df=sentences_df,
        source_metadata=source_meta_dict,
    )
    logger.info(f"  Extracted {len(examples)} raw examples")

    # Simple sampling for pilot (will use stratified_sample for full dataset)
    if len(examples) > max_examples:
        import random

        random.seed(random_state)
        examples = random.sample(examples, max_examples)

    return examples


def get_example_difficulty(example: GoldExample) -> str:
    """Calculate difficulty classification for an example."""
    features = calculate_difficulty_features(example)
    return classify_difficulty(features.wn_sense_count, features.baseline_margin)


def print_statistics(examples: list[GoldExample]) -> None:
    """Print statistics about collected examples."""
    if not examples:
        logger.warning("No examples collected!")
        return

    # By difficulty
    difficulty_counts: dict[str, int] = {}
    for ex in examples:
        diff = get_example_difficulty(ex)
        difficulty_counts[diff] = difficulty_counts.get(diff, 0) + 1

    # By POS
    pos_counts: dict[str, int] = {}
    for ex in examples:
        pos = ex.target.pos
        pos_counts[pos] = pos_counts.get(pos, 0) + 1

    # By bucket
    bucket_counts: dict[str, int] = {}
    for ex in examples:
        bucket = ex.source_bucket or "unknown"
        bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1

    print("\n" + "=" * 50)
    print("📊 PILOT COLLECTION STATISTICS")
    print("=" * 50)
    print(f"\nTotal examples: {len(examples)}")

    print("\n📌 By difficulty:")
    for diff, count in sorted(difficulty_counts.items()):
        pct = 100 * count / len(examples)
        print(f"  {diff:12s}: {count:4d} ({pct:5.1f}%)")

    print("\n📌 By POS:")
    for pos, count in sorted(pos_counts.items(), key=lambda x: -x[1]):
        pct = 100 * count / len(examples)
        print(f"  {pos:12s}: {count:4d} ({pct:5.1f}%)")

    print("\n📌 By bucket:")
    for bucket, count in sorted(bucket_counts.items()):
        pct = 100 * count / len(examples)
        print(f"  {bucket:20s}: {count:4d} ({pct:5.1f}%)")

    # Average candidates
    avg_candidates = sum(len(ex.candidates) for ex in examples) / len(examples)
    print(f"\n📌 Average candidates per example: {avg_candidates:.1f}")

    print("=" * 50 + "\n")


def save_examples(examples: list[GoldExample], output_path: Path) -> None:
    """Save examples to JSONL file.

    Args:
        examples: List of GoldExample instances
        output_path: Output file path
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        for example in examples:
            f.write(json.dumps(example.to_dict()) + "\n")

    logger.info(f"Saved {len(examples)} examples to {output_path}")


@app.command()
def main(
    output: Path = typer.Option(
        Path("data/wsd_gold/pilot_examples.jsonl"),
        "--output",
        "-o",
        help="Output JSONL file path",
    ),
    size: int = typer.Option(
        100,
        "--size",
        "-n",
        help="Total number of examples to collect",
    ),
    hard_ratio: float = typer.Option(
        0.2,
        "--hard-ratio",
        help="Ratio of hard examples (0.0-1.0)",
    ),
    seed: int = typer.Option(
        42,
        "--seed",
        help="Random seed for reproducibility",
    ),
) -> None:
    """Collect pilot batch for WSD Gold Dataset.

    Collects diverse examples across different sources and difficulty levels.
    """
    logger.info(f"Collecting pilot batch: {size} examples, {hard_ratio:.0%} hard")

    all_examples: list[GoldExample] = []

    # Collect from each available source
    sources = DEFAULT_SOURCES
    examples_per_source = size // len(sources)

    for source_id, config in sources.items():
        examples = collect_examples_from_source(
            source_id=source_id,
            config=config,
            max_examples=examples_per_source,
            random_state=seed,
        )
        all_examples.extend(examples)

    if not all_examples:
        logger.error("No examples collected! Check that sources are available.")
        raise typer.Exit(1)

    # Count hard examples
    hard_target = int(size * hard_ratio)
    hard_examples = [ex for ex in all_examples if get_example_difficulty(ex) == "hard"]
    logger.info(f"Hard examples: {len(hard_examples)} (target: {hard_target})")

    # Print statistics
    print_statistics(all_examples)

    # Save
    save_examples(all_examples, output)

    print(f"✅ Pilot collection complete: {output}")


if __name__ == "__main__":
    app()
