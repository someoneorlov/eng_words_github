#!/usr/bin/env python3
"""
Prepare sample data for Word Family experiment.

Creates:
- data/experiment/tokens_sample.parquet - tokens from 2000 random sentences
- data/experiment/sentences_sample.parquet - reconstructed sentences
- data/experiment/sample_stats.json - baseline statistics

Usage:
    uv run python scripts/prepare_pipeline_b_sample.py
    uv run python scripts/prepare_pipeline_b_sample.py --book american_tragedy_wsd --size 500
"""

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Paths
DATA_DIR = Path("data")
OUTPUT_DIR = DATA_DIR / "experiment"

# Sample parameters
SAMPLE_SIZE = 2000  # sentences
RANDOM_STATE = 42


def get_tokens_path(book_name: str) -> Path:
    """Resolve tokens path for a book (processed Stage 1 output)."""
    return DATA_DIR / "processed" / f"{book_name}_tokens.parquet"


def load_tokens(tokens_path: Path | None = None, book_name: str | None = None) -> pd.DataFrame:
    """Load the full tokens dataset."""
    if tokens_path is None:
        book_name = book_name or "american_tragedy"
        tokens_path = get_tokens_path(book_name)
    if not tokens_path.exists():
        raise FileNotFoundError(
            f"Tokens not found: {tokens_path}\n"
            "Run Stage 1 pipeline first, e.g.:\n"
            "  uv run python -m eng_words.pipeline --book-path data/raw/your.epub --book-name your_book\n"
            "Then: uv run python scripts/prepare_pipeline_b_sample.py --book your_book"
        )
    logger.info(f"Loading tokens from {tokens_path}")
    tokens = pd.read_parquet(tokens_path)
    logger.info(f"Loaded {len(tokens):,} tokens from {tokens['sentence_id'].nunique():,} sentences")
    return tokens


def sample_sentences(tokens: pd.DataFrame, n: int = SAMPLE_SIZE) -> pd.DataFrame:
    """Sample n random sentences and return their tokens."""
    all_sentence_ids = tokens["sentence_id"].drop_duplicates()
    logger.info(f"Total unique sentences: {len(all_sentence_ids):,}")

    # Random sample
    sample_sids = all_sentence_ids.sample(n=n, random_state=RANDOM_STATE)
    logger.info(f"Sampled {len(sample_sids):,} sentences")

    # Filter tokens
    tokens_sample = tokens[tokens["sentence_id"].isin(sample_sids)].copy()
    logger.info(f"Sample contains {len(tokens_sample):,} tokens")

    return tokens_sample


def filter_content_words(tokens: pd.DataFrame) -> pd.DataFrame:
    """Filter to content words only (NOUN, VERB, ADJ, ADV)."""
    content = tokens[
        (tokens["is_alpha"] == True)
        & (tokens["is_stop"] == False)
        & (tokens["pos"].isin(["NOUN", "VERB", "ADJ", "ADV"]))
    ].copy()
    logger.info(
        f"Content words: {len(content):,} tokens, {content['lemma'].nunique():,} unique lemmas"
    )
    return content


def reconstruct_sentences(tokens: pd.DataFrame) -> pd.DataFrame:
    """Reconstruct sentence text from tokens."""
    logger.info("Reconstructing sentences...")

    sentences = (
        tokens.groupby("sentence_id")
        .apply(lambda g: " ".join(g.sort_values("position")["surface"].astype(str)))
        .reset_index(name="text")
    )

    logger.info(f"Reconstructed {len(sentences):,} sentences")
    return sentences


def compute_stats(tokens_sample: pd.DataFrame, content_tokens: pd.DataFrame) -> dict:
    """Compute baseline statistics for the sample."""
    # Lemma distribution for content words
    lemma_counts = content_tokens.groupby("lemma")["sentence_id"].nunique()

    stats = {
        "sample_size": int(tokens_sample["sentence_id"].nunique()),
        "total_tokens": int(len(tokens_sample)),
        "content_tokens": int(len(content_tokens)),
        "unique_lemmas_all": int(tokens_sample["lemma"].nunique()),
        "unique_lemmas_content": int(content_tokens["lemma"].nunique()),
        "lemma_distribution": {
            "min_examples": int(lemma_counts.min()),
            "max_examples": int(lemma_counts.max()),
            "median_examples": float(lemma_counts.median()),
            "mean_examples": float(lemma_counts.mean()),
            "lemmas_gt_100": int((lemma_counts > 100).sum()),
            "lemmas_gt_50": int((lemma_counts > 50).sum()),
            "lemmas_gt_10": int((lemma_counts > 10).sum()),
        },
        "pos_distribution": content_tokens["pos"].value_counts().to_dict(),
    }

    return stats


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare sample for Word Family experiment")
    parser.add_argument(
        "--book",
        type=str,
        default="american_tragedy",
        help="Book name (processed tokens: data/processed/{book}_tokens.parquet)",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=SAMPLE_SIZE,
        help=f"Number of sentences to sample (default {SAMPLE_SIZE})",
    )
    return parser.parse_args()


def main():
    """Main function to prepare the sample."""
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load full dataset
    tokens = load_tokens(book_name=args.book)

    # 2. Sample sentences
    tokens_sample = sample_sentences(tokens, args.size)

    # 3. Filter content words
    content_tokens = filter_content_words(tokens_sample)

    # 4. Reconstruct sentences
    sentences = reconstruct_sentences(tokens_sample)

    # 5. Compute stats
    stats = compute_stats(tokens_sample, content_tokens)

    # 6. Save outputs
    tokens_output = OUTPUT_DIR / "tokens_sample.parquet"
    tokens_sample.to_parquet(tokens_output, index=False)
    logger.info(f"Saved tokens sample to {tokens_output}")

    sentences_output = OUTPUT_DIR / "sentences_sample.parquet"
    sentences.to_parquet(sentences_output, index=False)
    logger.info(f"Saved sentences to {sentences_output}")

    stats_output = OUTPUT_DIR / "sample_stats.json"
    with open(stats_output, "w") as f:
        json.dump(stats, f, indent=2)
    logger.info(f"Saved stats to {stats_output}")

    # 7. Print summary
    print("\n" + "=" * 60)
    print("SAMPLE PREPARATION COMPLETE")
    print("=" * 60)
    print(f"Sentences: {stats['sample_size']:,}")
    print(f"Total tokens: {stats['total_tokens']:,}")
    print(f"Content tokens: {stats['content_tokens']:,}")
    print(f"Unique lemmas (content): {stats['unique_lemmas_content']:,}")
    print("\nLemma distribution:")
    print(f"  - Lemmas with >100 examples: {stats['lemma_distribution']['lemmas_gt_100']}")
    print(f"  - Lemmas with >50 examples: {stats['lemma_distribution']['lemmas_gt_50']}")
    print(f"  - Lemmas with >10 examples: {stats['lemma_distribution']['lemmas_gt_10']}")
    print(f"  - Max examples per lemma: {stats['lemma_distribution']['max_examples']}")
    print("\nPOS distribution:")
    for pos, count in stats["pos_distribution"].items():
        print(f"  - {pos}: {count:,}")
    print("=" * 60)

    return tokens_sample, sentences, stats


if __name__ == "__main__":
    main()
