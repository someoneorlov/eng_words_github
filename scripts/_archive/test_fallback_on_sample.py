#!/usr/bin/env python3
"""Test fallback example generation on a sample of 200 cards.

This script:
1. Loads a sample of 200 cards from aggregated data
2. Generates cards with enable_fallback=True
3. Measures how many cards used fallback for example generation
4. Validates all cards
5. Saves results for manual review

Usage:
    uv run python scripts/test_fallback_on_sample.py [--n 200]
"""

import argparse
import json
import logging
import time
from dataclasses import asdict
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from eng_words.llm.base import get_provider
from eng_words.llm.response_cache import ResponseCache
from eng_words.llm.smart_card_generator import SmartCard, SmartCardGenerator
from eng_words.text_processing import create_sentences_dataframe, reconstruct_sentences_from_tokens
from eng_words.validation.example_validator import fix_invalid_cards, validate_card_examples

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# Paths
BOOK_NAME = "american_tragedy"
AGGREGATED_CARDS_PATH = Path("data/synset_aggregation_full/aggregated_cards.parquet")
TOKENS_PATH = Path(f"data/processed/{BOOK_NAME}_tokens.parquet")
OUTPUT_DIR = Path("data/synset_cards/test_fallback")
CACHE_DIR = OUTPUT_DIR / "llm_cache"


def card_to_serializable(card: SmartCard) -> dict:
    """Convert SmartCard to a serializable dictionary."""
    d = asdict(card)
    # Convert any numpy arrays/types to standard Python types
    for key, value in d.items():
        if isinstance(value, (list, tuple)):
            d[key] = [item.item() if hasattr(item, "item") else item for item in value]
        elif hasattr(value, "item"):
            d[key] = value.item()
    return d


def main():
    """Test fallback example generation on a sample."""
    parser = argparse.ArgumentParser(
        description="Test fallback example generation on a sample of cards."
    )
    parser.add_argument(
        "--n",
        type=int,
        default=200,
        help="Number of cards to test (default: 200).",
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("TESTING FALLBACK ON SAMPLE")
    logger.info("=" * 70)

    # Step 1: Load aggregated cards
    logger.info("\n## Step 1: Load aggregated cards")
    cards_df = pd.read_parquet(AGGREGATED_CARDS_PATH)
    logger.info(f"  Loaded {len(cards_df):,} cards")

    # Sample n cards
    if len(cards_df) > args.n:
        sample_df = cards_df.sample(n=args.n, random_state=42)
        logger.info(f"  Sampled {args.n} cards for testing")
    else:
        sample_df = cards_df
        logger.info(f"  Using all {len(cards_df)} cards (less than {args.n})")

    # Step 2: Reconstruct sentences
    logger.info("\n## Step 2: Reconstruct sentences")
    tokens_df = pd.read_parquet(TOKENS_PATH)
    sentences = reconstruct_sentences_from_tokens(tokens_df)
    sentences_df = create_sentences_dataframe(sentences)
    sentences_lookup = dict(zip(sentences_df["sentence_id"], sentences_df["sentence"]))
    logger.info(f"  {len(sentences_df):,} sentences available")

    # Step 3: Initialize SmartCardGenerator
    logger.info("\n## Step 3: Initialize SmartCardGenerator")
    provider = get_provider("gemini", "gemini-3-flash-preview")
    cache = ResponseCache(cache_dir=CACHE_DIR, enabled=True)
    generator = SmartCardGenerator(
        provider=provider, cache=cache, book_name=BOOK_NAME, max_retries=2
    )

    # Step 4: Prepare data for generation
    logger.info("\n## Step 4: Prepare data for generation")

    # Convert DataFrame to list of dicts with required fields
    items = []
    for _, row in sample_df.iterrows():
        # Convert sentence_ids to examples
        sentence_ids = row.get("sentence_ids", [])
        if isinstance(sentence_ids, (list, tuple, pd.Series)):
            examples = [
                sentences_lookup.get(sid, "") for sid in sentence_ids if sid in sentences_lookup
            ]
        else:
            examples = []

        # Prepare item dict
        item = {
            "lemma": row["lemma"],
            "pos": row["pos"],
            "supersense": row["supersense"],
            "wn_definition": row.get("definition", ""),  # Use 'definition' column
            "examples": examples[:10],  # Limit to 10 examples
            "synset_group": row.get("synset_group"),
            "primary_synset": row.get("primary_synset", ""),
        }
        items.append(item)

    logger.info(f"  Prepared {len(items)} items with examples")
    avg_examples = sum(len(item["examples"]) for item in items) / len(items) if items else 0
    logger.info(f"  Average examples per card: {avg_examples:.1f}")

    # Step 5: Generate cards with fallback
    logger.info("\n## Step 5: Generate cards with fallback")
    start_time = time.time()

    generated_cards = generator.generate_batch(
        items,
        progress=True,
        enable_fallback=True,
    )

    elapsed = time.time() - start_time

    logger.info(f"  Generated {len(generated_cards):,} cards in {elapsed:.1f}s")

    # Step 6: Analyze fallback usage
    logger.info("\n## Step 5: Analyze fallback usage")
    cards_with_examples = [c for c in generated_cards if c.selected_examples]
    cards_without_examples = [c for c in generated_cards if not c.selected_examples]
    cards_with_generated_example = [c for c in generated_cards if c.generated_example]
    # Cards where fallback was used (no examples initially, then generated via fallback)
    cards_with_fallback_generated = [
        c
        for c in generated_cards
        if c.generated_example and (not c.selected_examples or len(c.selected_examples) == 0)
    ]

    logger.info(f"  Total cards: {len(generated_cards):,}")
    logger.info(
        f"  With selected_examples: {len(cards_with_examples):,} ({len(cards_with_examples)/max(len(generated_cards),1)*100:.1f}%)"
    )
    logger.info(
        f"  Without selected_examples: {len(cards_without_examples):,} ({len(cards_without_examples)/max(len(generated_cards),1)*100:.1f}%)"
    )
    logger.info(
        f"  With generated_example: {len(cards_with_generated_example):,} ({len(cards_with_generated_example)/max(len(generated_cards),1)*100:.1f}%)"
    )
    logger.info(
        f"  With fallback-generated example: {len(cards_with_fallback_generated):,} ({len(cards_with_fallback_generated)/max(len(generated_cards),1)*100:.1f}%)"
    )

    # Step 7: Validate all cards
    logger.info("\n## Step 6: Validate all cards")
    final_cards, removed_cards = fix_invalid_cards(
        generated_cards,
        use_generated_example=True,
        remove_unfixable=True,
    )

    num_valid = sum(1 for c in final_cards if validate_card_examples(c).is_valid)
    num_invalid = len(final_cards) - num_valid

    logger.info(f"  Valid cards: {num_valid:,} ({num_valid/max(len(final_cards),1)*100:.1f}%)")
    logger.info(f"  Invalid cards: {num_invalid} ({num_invalid/max(len(final_cards),1)*100:.1f}%)")
    logger.info(f"  Removed (unfixable): {len(removed_cards):,}")

    # Step 8: Save results
    logger.info("\n## Step 7: Save results")
    output_path = OUTPUT_DIR / "test_fallback_sample.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump([card_to_serializable(c) for c in final_cards], f, ensure_ascii=False, indent=2)
    logger.info(f"  Saved to {output_path}")

    # Save fallback cards for manual review
    if cards_with_fallback_generated:
        fallback_path = OUTPUT_DIR / "fallback_cards_sample.json"
        with open(fallback_path, "w", encoding="utf-8") as f:
            json.dump(
                [card_to_serializable(c) for c in cards_with_fallback_generated[:20]],
                f,
                ensure_ascii=False,
                indent=2,
            )
        logger.info(
            f"  Saved {min(20, len(cards_with_fallback_generated))} fallback cards for manual review: {fallback_path}"
        )

    # Export to Anki CSV (skip for now - requires DataFrame conversion)
    # anki_output_path = OUTPUT_DIR / "test_fallback_sample.csv"
    # export_to_anki_csv(final_cards, anki_output_path)
    # logger.info(f"  Exported to Anki CSV: {anki_output_path}")

    # Step 9: LLM stats
    logger.info("\n## Step 8: LLM stats")
    stats = generator.stats()
    cache_stats = cache.stats()
    logger.info(f"  LLM Cost: ${stats['total_cost']:.4f}")
    logger.info(f"  Total cards processed: {stats['total_cards']}")
    logger.info(f"  Successful: {stats['successful']}")
    logger.info(f"  Failed: {stats['failed']}")
    logger.info(f"  Fallback attempts: {stats['fallback_attempts']}")
    logger.info(f"  Fallback success: {stats['fallback_success']}")
    logger.info(f"  Cache hits: {cache_stats.get('hits', 0)}")
    logger.info(f"  Cache misses: {cache_stats.get('misses', 0)}")

    logger.info("\n" + "=" * 70)
    logger.info("TESTING COMPLETE")
    logger.info("=" * 70)

    # Summary
    logger.info("\n📊 SUMMARY:")
    logger.info(f"  Total cards tested: {len(generated_cards):,}")
    logger.info(
        f"  Cards with fallback-generated examples: {len(cards_with_fallback_generated):,} ({len(cards_with_fallback_generated)/max(len(generated_cards),1)*100:.1f}%)"
    )
    logger.info(f"  Valid cards: {num_valid:,} ({num_valid/max(len(final_cards),1)*100:.1f}%)")
    logger.info(f"  Invalid cards: {num_invalid} ({num_invalid/max(len(final_cards),1)*100:.1f}%)")
    logger.info(f"  LLM Cost: ${stats['total_cost']:.4f}")


if __name__ == "__main__":
    main()
