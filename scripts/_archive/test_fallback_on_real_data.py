#!/usr/bin/env python3
"""Test Smart Fallback on real card generation data.

Compares card generation with and without fallback enabled.
"""

import json
import logging
import os
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from eng_words.llm.base import get_provider
from eng_words.llm.response_cache import ResponseCache
from eng_words.llm.smart_card_generator import SmartCardGenerator
from eng_words.text_processing import (
    create_sentences_dataframe,
    reconstruct_sentences_from_tokens,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

load_dotenv()

# Paths
AGGREGATED_CARDS_PATH = Path("data/synset_aggregation_full/aggregated_cards.parquet")
TOKENS_PATH = Path("data/processed/american_tragedy_tokens.parquet")
OUTPUT_DIR = Path("data/fallback_test")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_sentences() -> pd.DataFrame:
    """Load and reconstruct sentences from tokens."""
    logger.info(f"Loading tokens from {TOKENS_PATH}")
    tokens_df = pd.read_parquet(TOKENS_PATH)
    logger.info("  Reconstructing sentences...")
    sentences = reconstruct_sentences_from_tokens(tokens_df)
    sentences_df = create_sentences_dataframe(sentences)
    logger.info(f"  {len(sentences_df)} sentences")
    return sentences_df


def load_items(sentences_df: pd.DataFrame, limit: int | None = None) -> list[dict]:
    """Load items for card generation."""
    logger.info(f"Loading aggregated cards from {AGGREGATED_CARDS_PATH}")
    df = pd.read_parquet(AGGREGATED_CARDS_PATH)
    logger.info(f"  Total cards: {len(df)}")

    if limit:
        df = df.head(limit)
        logger.info(f"  Limited to: {len(df)}")

    items = []
    for _, row in df.iterrows():
        # Get examples from sentence_ids
        sentence_ids = row["sentence_ids"]
        examples = sentences_df[
            sentences_df["sentence_id"].isin(sentence_ids)
        ]["sentence"].tolist()[:10]

        # Convert numpy arrays to lists
        synset_group = row.get("synset_group", [])
        if hasattr(synset_group, "tolist"):
            synset_group = synset_group.tolist()

        items.append({
            "lemma": row["lemma"],
            "pos": row["pos"],
            "supersense": row["supersense"],
            "wn_definition": row["definition"],
            "examples": examples,
            "synset_group": synset_group,
            "primary_synset": row.get("primary_synset", ""),
        })

    return items


def run_test(
    items: list[dict],
    enable_fallback: bool,
    cache_suffix: str,
) -> tuple[list, dict]:
    """Run card generation test."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Running with fallback={'ENABLED' if enable_fallback else 'DISABLED'}")
    logger.info(f"{'='*60}")

    provider = get_provider("gemini", "gemini-3-flash-preview")
    cache_dir = OUTPUT_DIR / f"llm_cache_{cache_suffix}"
    cache = ResponseCache(cache_dir=cache_dir, enabled=True)

    generator = SmartCardGenerator(
        provider=provider,
        cache=cache,
        book_name="american_tragedy",
        max_retries=2,
    )

    start_time = time.time()
    cards = generator.generate_batch(
        items,
        progress=True,
        enable_fallback=enable_fallback,
    )
    elapsed = time.time() - start_time

    stats = generator.stats()
    stats["elapsed_seconds"] = elapsed
    stats["cards_per_minute"] = len(cards) / (elapsed / 60) if elapsed > 0 else 0

    # Calculate quality metrics
    cards_with_examples = sum(1 for c in cards if c.selected_examples)
    cards_without_examples = sum(1 for c in cards if not c.selected_examples)
    cards_with_fallback = sum(1 for c in cards if c.fallback_used)

    stats["cards_with_examples"] = cards_with_examples
    stats["cards_without_examples"] = cards_without_examples
    stats["cards_with_fallback"] = cards_with_fallback
    stats["no_example_rate"] = cards_without_examples / len(cards) * 100 if cards else 0

    logger.info(f"\nResults:")
    logger.info(f"  Total cards: {len(cards)}")
    logger.info(f"  With examples: {cards_with_examples} ({cards_with_examples/len(cards)*100:.1f}%)")
    logger.info(f"  Without examples: {cards_without_examples} ({cards_without_examples/len(cards)*100:.1f}%)")
    if enable_fallback:
        logger.info(f"  Fallback used: {cards_with_fallback}")
        logger.info(f"  Fallback attempts: {stats['fallback_attempts']}")
        logger.info(f"  Fallback success: {stats['fallback_success']}")
    logger.info(f"  Cost: ${stats['total_cost']:.4f}")
    logger.info(f"  Time: {elapsed:.1f}s ({stats['cards_per_minute']:.1f} cards/min)")
    logger.info(f"  Cache: {stats['cache_stats']}")

    return cards, stats


def save_results(cards: list, stats: dict, suffix: str):
    """Save results to files."""
    # Save cards as JSON
    cards_path = OUTPUT_DIR / f"cards_{suffix}.json"
    with open(cards_path, "w", encoding="utf-8") as f:
        json.dump([c.to_dict() for c in cards], f, ensure_ascii=False, indent=2)
    logger.info(f"Saved cards to {cards_path}")

    # Save stats
    stats_path = OUTPUT_DIR / f"stats_{suffix}.json"
    # Convert cache_stats to serializable format
    stats_copy = stats.copy()
    if "cache_stats" in stats_copy:
        stats_copy["cache_stats"] = dict(stats_copy["cache_stats"])
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats_copy, f, indent=2)
    logger.info(f"Saved stats to {stats_path}")


def analyze_fallback_cards(cards: list):
    """Analyze cards that used fallback."""
    fallback_cards = [c for c in cards if c.fallback_used]

    if not fallback_cards:
        logger.info("\nNo fallback cards to analyze.")
        return

    logger.info(f"\n{'='*60}")
    logger.info(f"FALLBACK ANALYSIS ({len(fallback_cards)} cards)")
    logger.info(f"{'='*60}")

    for i, card in enumerate(fallback_cards[:20]):  # Show first 20
        logger.info(f"\n{i+1}. {card.lemma} ({card.pos})")
        logger.info(f"   Original synset: {card.primary_synset}")
        logger.info(f"   Fallback synset: {card.fallback_synset}")
        logger.info(f"   Definition: {card.wn_definition[:60]}...")
        logger.info(f"   Examples selected: {len(card.selected_examples)}")
        if card.selected_examples:
            logger.info(f"   Example: {card.selected_examples[0][:80]}...")


def main():
    """Main test function."""
    # Load sentences first
    sentences_df = load_sentences()

    # Load items
    items = load_items(sentences_df, limit=500)
    logger.info(f"Loaded {len(items)} items for testing")

    # Test WITHOUT fallback
    cards_no_fb, stats_no_fb = run_test(items, enable_fallback=False, cache_suffix="no_fallback")
    save_results(cards_no_fb, stats_no_fb, "no_fallback")

    # Test WITH fallback (using same cached responses + new fallback calls)
    cards_with_fb, stats_with_fb = run_test(items, enable_fallback=True, cache_suffix="with_fallback")
    save_results(cards_with_fb, stats_with_fb, "with_fallback")

    # Compare results
    logger.info(f"\n{'='*60}")
    logger.info("COMPARISON")
    logger.info(f"{'='*60}")
    logger.info(f"{'Metric':<30} {'No Fallback':>15} {'With Fallback':>15}")
    logger.info("-" * 60)
    logger.info(f"{'Cards without examples':<30} {stats_no_fb['cards_without_examples']:>15} {stats_with_fb['cards_without_examples']:>15}")
    logger.info(f"{'No-example rate':<30} {stats_no_fb['no_example_rate']:>14.1f}% {stats_with_fb['no_example_rate']:>14.1f}%")
    logger.info(f"{'Fallback used':<30} {'-':>15} {stats_with_fb['cards_with_fallback']:>15}")
    logger.info(f"{'Total cost':<30} ${stats_no_fb['total_cost']:>14.4f} ${stats_with_fb['total_cost']:>14.4f}")
    logger.info(f"{'Additional cost':<30} {'-':>15} ${stats_with_fb['total_cost'] - stats_no_fb['total_cost']:>14.4f}")

    # Analyze fallback cards
    analyze_fallback_cards(cards_with_fb)

    # Success criteria
    improvement = stats_no_fb["no_example_rate"] - stats_with_fb["no_example_rate"]
    logger.info(f"\n{'='*60}")
    logger.info("SUCCESS CRITERIA")
    logger.info(f"{'='*60}")
    logger.info(f"  Improvement: {improvement:.1f}% points")
    logger.info(f"  Target: <5% no-example rate")
    logger.info(f"  Actual: {stats_with_fb['no_example_rate']:.1f}%")

    if stats_with_fb["no_example_rate"] < 5:
        logger.info("  ✅ SUCCESS: Target achieved!")
    else:
        logger.info("  ⚠️  Target not achieved, but improvement is positive")

    logger.info("\n✅ Test complete!")


if __name__ == "__main__":
    main()

