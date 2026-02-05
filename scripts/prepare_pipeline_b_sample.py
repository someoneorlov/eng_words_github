#!/usr/bin/env python3
"""
Prepare sample data for Word Family experiment.

Uses Stage 1 outputs: tokens.parquet and sentences.parquet (no reconstruction).
Creates:
- data/experiment/tokens_sample.parquet - tokens from N random sentences
- data/experiment/sentences_sample.parquet - sentences filtered by sample (sentence_id, text)
- data/experiment/sample_stats.json - baseline statistics

Invariant: after sample, all sentence_id from tokens_sample are in sentences_sample.

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

# Paths (OUTPUT_DIR is set from DATA_DIR in main)
DATA_DIR = Path("data")

# Sample parameters
SAMPLE_SIZE = 2000  # sentences
RANDOM_STATE = 42


def get_tokens_path(book_name: str, data_dir: Path | None = None) -> Path:
    """Resolve tokens path for a book (processed Stage 1 output)."""
    base = data_dir or DATA_DIR
    return base / "processed" / f"{book_name}_tokens.parquet"


def get_sentences_path(book_name: str, data_dir: Path | None = None) -> Path:
    """Resolve sentences path for a book (Stage 1 artifact)."""
    base = data_dir or DATA_DIR
    return base / "processed" / f"{book_name}_sentences.parquet"


def load_tokens(
    tokens_path: Path | None = None,
    book_name: str | None = None,
    data_dir: Path | None = None,
) -> pd.DataFrame:
    """Load the full tokens dataset."""
    if tokens_path is None:
        book_name = book_name or "american_tragedy"
        tokens_path = get_tokens_path(book_name, data_dir=data_dir)
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


def load_sentences(
    sentences_path: Path | None = None,
    book_name: str | None = None,
    data_dir: Path | None = None,
) -> pd.DataFrame:
    """Load sentences from Stage 1 sentences.parquet (columns: sentence_id, text)."""
    if sentences_path is None:
        book_name = book_name or "american_tragedy"
        sentences_path = get_sentences_path(book_name, data_dir=data_dir)
    if not sentences_path.exists():
        raise FileNotFoundError(
            f"Sentences not found: {sentences_path}\n"
            "Run Stage 1 pipeline first (it writes <book>_sentences.parquet).\n"
            "  uv run python -m eng_words.pipeline --book-path data/raw/your.epub --book-name your_book\n"
            "Then: uv run python scripts/prepare_pipeline_b_sample.py --book your_book"
        )
    logger.info(f"Loading sentences from {sentences_path}")
    sentences = pd.read_parquet(sentences_path)
    if "text" not in sentences.columns or "sentence_id" not in sentences.columns:
        raise ValueError(
            f"Sentences parquet must have columns sentence_id, text; got {list(sentences.columns)}"
        )
    logger.info(f"Loaded {len(sentences):,} sentences")
    return sentences


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


def filter_sentences_by_ids(sentences: pd.DataFrame, sentence_ids: pd.Series) -> pd.DataFrame:
    """Filter sentences to those in sentence_ids. Preserves columns sentence_id, text."""
    sample_sids = set(sentence_ids.unique())
    out = sentences[sentences["sentence_id"].isin(sample_sids)].copy()
    logger.info(f"Filtered to {len(out):,} sentences in sample")
    return out


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
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DATA_DIR,
        help="Base data directory (default: data)",
    )
    return parser.parse_args()


def main():
    """Main function to prepare the sample."""
    args = parse_args()
    data_dir = Path(args.data_dir)
    output_dir = data_dir / "experiment"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Stage 1 outputs
    tokens = load_tokens(book_name=args.book, data_dir=data_dir)
    sentences = load_sentences(book_name=args.book, data_dir=data_dir)

    # Invariant: tokens.sentence_id ⊆ sentences.sentence_id
    token_sids = set(tokens["sentence_id"].unique())
    sentence_sids = set(sentences["sentence_id"].unique())
    if not token_sids.issubset(sentence_sids):
        missing = token_sids - sentence_sids
        raise ValueError(
            f"Invariant violated: tokens.sentence_id must be subset of sentences.sentence_id. "
            f"Missing in sentences: {len(missing)} ids. Re-run Stage 1 to rebuild outputs."
        )

    # 2. Sample sentence_id (deterministic: RANDOM_STATE)
    tokens_sample = sample_sentences(tokens, args.size)
    sample_sids = tokens_sample["sentence_id"].drop_duplicates()
    sentences_sample = filter_sentences_by_ids(sentences, sample_sids)

    # Invariant: all sentence_id from tokens_sample in sentences_sample
    ts_sids = set(tokens_sample["sentence_id"].unique())
    ss_sids = set(sentences_sample["sentence_id"].unique())
    if not ts_sids.issubset(ss_sids):
        raise ValueError(
            "Invariant violated: after sample, tokens_sample.sentence_id must be subset of sentences_sample.sentence_id"
        )

    # 3. Filter content words
    content_tokens = filter_content_words(tokens_sample)

    # 4. Compute stats
    stats = compute_stats(tokens_sample, content_tokens)

    # 5. Save outputs
    tokens_output = output_dir / "tokens_sample.parquet"
    tokens_sample.to_parquet(tokens_output, index=False)
    logger.info(f"Saved tokens sample to {tokens_output}")

    sentences_output = output_dir / "sentences_sample.parquet"
    sentences_sample.to_parquet(sentences_output, index=False)
    logger.info(f"Saved sentences to {sentences_output}")

    stats_output = output_dir / "sample_stats.json"
    with open(stats_output, "w") as f:
        json.dump(stats, f, indent=2)
    logger.info(f"Saved stats to {stats_output}")

    # 6. Print summary
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

    return tokens_sample, sentences_sample, stats


if __name__ == "__main__":
    main()
