#!/usr/bin/env python3
"""
Run Pipeline A (current WSD-based) on experiment sample.

This script:
1. Loads the sample data (tokens_sample.parquet)
2. Finds matching lemmas in aggregated_cards.parquet (pre-computed WSD results)
3. Generates cards using SmartCardGenerator
4. Saves results to data/experiment/cards_A.json

Note: Pipeline A uses pre-computed WSD and synset aggregation from aggregated_cards.parquet.
For a fair comparison with Pipeline B, we filter to lemmas present in the sample.
"""

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from eng_words.llm.base import get_provider
from eng_words.llm.response_cache import ResponseCache
from eng_words.llm.smart_card_generator import (
    SmartCard,
    SmartCardGenerator,
    check_spoilers,
    mark_examples_by_length,
    select_examples_for_generation,
)
from eng_words.validation import validate_examples_for_synset_group

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Paths
DATA_DIR = Path("data/experiment")
AGGREGATED_CARDS_PATH = Path("data/synset_aggregation_full/aggregated_cards.parquet")
CACHE_DIR = Path("data/cache/llm_responses")
BOOK_NAME = "american_tragedy"


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run Pipeline A (WSD-based)")

    parser.add_argument(
        "--limit", type=int, default=None, help="Limit number of lemmas to process (for testing)"
    )
    parser.add_argument("--no-cache", action="store_true", help="Disable response caching")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path (default: data/experiment/cards_A.json)",
    )

    return parser.parse_args()


def normalize_synset_group(synset_group):
    """Normalize synset_group to a list."""
    if isinstance(synset_group, (np.ndarray, list)):
        return list(synset_group) if len(synset_group) > 0 else []
    elif isinstance(synset_group, str):
        try:
            parsed = json.loads(synset_group)
            if not isinstance(parsed, list):
                parsed = [parsed]
            return parsed
        except (json.JSONDecodeError, TypeError):
            return [synset_group] if synset_group else []
    elif synset_group is None:
        return []
    else:
        try:
            if pd.isna(synset_group):
                return []
            else:
                return [synset_group]
        except (ValueError, TypeError):
            return [synset_group]


def card_to_dict(card: SmartCard) -> dict:
    """Convert SmartCard to JSON-serializable dict."""
    d = asdict(card)

    def make_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_serializable(item) for item in obj]
        elif isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        return obj

    return make_serializable(d)


def main():
    args = parse_args()

    start_time = time.time()
    logger.info(f"Starting Pipeline A at {time.strftime('%Y-%m-%d %H:%M:%S')}")

    # 1. Load sample data
    logger.info("Loading sample data...")
    tokens_sample = pd.read_parquet(DATA_DIR / "tokens_sample.parquet")
    sentences_sample = pd.read_parquet(DATA_DIR / "sentences_sample.parquet")

    # Get unique lemmas from sample (content words only)
    content_tokens = tokens_sample[
        (tokens_sample["is_alpha"] == True)
        & (tokens_sample["is_stop"] == False)
        & (tokens_sample["pos"].isin(["NOUN", "VERB", "ADJ", "ADV"]))
    ]
    sample_lemmas = set(content_tokens["lemma"].unique())
    logger.info(f"Sample has {len(sample_lemmas)} unique content lemmas")

    # 2. Load aggregated cards and filter to sample lemmas
    logger.info("Loading aggregated cards...")
    if not AGGREGATED_CARDS_PATH.exists():
        raise FileNotFoundError(
            f"Aggregated cards not found: {AGGREGATED_CARDS_PATH}\n" "Run the full pipeline first."
        )

    agg_cards = pd.read_parquet(AGGREGATED_CARDS_PATH)
    logger.info(f"Loaded {len(agg_cards)} aggregated cards")

    # Filter to lemmas in sample
    agg_cards_filtered = agg_cards[agg_cards["lemma"].isin(sample_lemmas)]
    logger.info(f"Filtered to {len(agg_cards_filtered)} cards for sample lemmas")

    # Sort by frequency (most frequent first) for consistent comparison with Pipeline B
    lemma_counts = content_tokens.groupby("lemma")["sentence_id"].nunique().to_dict()
    agg_cards_filtered = agg_cards_filtered.copy()
    agg_cards_filtered["example_count"] = agg_cards_filtered["lemma"].map(lemma_counts).fillna(0)
    agg_cards_filtered = agg_cards_filtered.sort_values("example_count", ascending=False)

    # Apply limit if specified
    if args.limit:
        # Get top N lemmas by frequency
        top_lemmas = agg_cards_filtered.drop_duplicates("lemma").head(args.limit)["lemma"].tolist()
        agg_cards_filtered = agg_cards_filtered[agg_cards_filtered["lemma"].isin(top_lemmas)]
        logger.info(f"Limited to {len(agg_cards_filtered)} cards for {args.limit} lemmas")

    # 3. Create sentences lookup
    sentences_lookup = dict(zip(sentences_sample["sentence_id"], sentences_sample["text"]))

    # 4. Initialize provider and cache
    provider = get_provider("gemini", "gemini-3-flash-preview")
    cache = ResponseCache(CACHE_DIR, enabled=not args.no_cache)
    generator = SmartCardGenerator(
        provider=provider,
        cache=cache,
        book_name=BOOK_NAME,
        max_retries=2,
    )

    # 5. Generate cards
    logger.info("Generating cards...")

    generated_cards = []
    skipped_cards = 0
    errors = []

    # Statistics
    length_stats = {"too_long": 0, "appropriate_length": 0, "too_short": 0}
    spoiler_stats = {"has_spoiler": 0, "no_spoiler": 0}

    total = len(agg_cards_filtered)

    for progress, (idx, row) in enumerate(agg_cards_filtered.iterrows(), 1):
        lemma = row["lemma"]

        if progress % 10 == 0 or progress == total:
            logger.info(f"[{progress}/{total}] Processing '{lemma}'")

        try:
            # Get sentence_ids for this card
            sentence_ids = row.get("sentence_ids", [])
            if isinstance(sentence_ids, str):
                try:
                    sentence_ids = json.loads(sentence_ids)
                except:
                    sentence_ids = []
            elif not isinstance(sentence_ids, (list, np.ndarray)):
                sentence_ids = []

            # Filter to sentences in our sample
            sentence_ids = [sid for sid in sentence_ids if sid in sentences_lookup]

            if not sentence_ids:
                skipped_cards += 1
                continue

            # Get examples
            examples_with_ids = [(sid, sentences_lookup[sid]) for sid in sentence_ids]

            # Get synset info
            synset_group = normalize_synset_group(row.get("synset_group", []))
            primary_synset = row.get("primary_synset", "")

            # Validate examples for synset group
            validation = validate_examples_for_synset_group(
                lemma=lemma,
                synset_group=synset_group,
                primary_synset=primary_synset,
                examples=examples_with_ids,
                provider=provider,
                cache=cache,
            )

            if not validation.get("has_valid", False):
                skipped_cards += 1
                continue

            # Get valid examples
            valid_examples = [
                (sid, sentences_lookup[sid])
                for sid in validation.get("valid_sentence_ids", [])
                if sid in sentences_lookup
            ]

            if not valid_examples:
                skipped_cards += 1
                continue

            # Mark by length
            length_flags = mark_examples_by_length(valid_examples, max_words=50, min_words=6)

            for sid, ex in valid_examples:
                word_count = len(ex.split())
                if word_count < 6:
                    length_stats["too_short"] += 1
                elif word_count > 50:
                    length_stats["too_long"] += 1
                else:
                    length_stats["appropriate_length"] += 1

            # Check spoilers
            spoiler_flags = check_spoilers(
                examples=valid_examples,
                provider=provider,
                cache=cache,
                book_name=BOOK_NAME,
            )

            spoiler_count = sum(1 for v in spoiler_flags.values() if v)
            spoiler_stats["has_spoiler"] += spoiler_count
            spoiler_stats["no_spoiler"] += len(spoiler_flags) - spoiler_count

            # Select examples
            selection = select_examples_for_generation(
                all_examples=valid_examples,
                length_flags=length_flags,
                spoiler_flags=spoiler_flags,
                target_count=3,
            )

            selected_examples_text = [ex for _, ex in selection["selected_from_book"]]
            generate_count = selection["generate_count"]

            # Generate card
            card = generator.generate_card(
                lemma=lemma,
                pos=row.get("pos", "unknown"),
                supersense=row.get("supersense", "unknown"),
                wn_definition=row.get("definition", ""),
                examples=selected_examples_text,
                synset_group=synset_group,
                primary_synset=primary_synset,
                generate_count=generate_count,
            )

            if card:
                card_dict = card_to_dict(card)
                card_dict["source"] = "pipeline_a"
                generated_cards.append(card_dict)
            else:
                skipped_cards += 1

        except Exception as e:
            logger.error(f"Error processing '{lemma}': {e}")
            errors.append(
                {
                    "lemma": lemma,
                    "error": str(e),
                }
            )

    # 6. Save results
    output_path = Path(args.output) if args.output else DATA_DIR / "cards_A.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    elapsed = time.time() - start_time
    stats = generator.stats()

    results = {
        "pipeline": "A",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "config": {
            "limit": args.limit,
            "use_wsd": True,
            "use_synset_aggregation": True,
        },
        "stats": {
            "aggregated_cards_total": len(agg_cards),
            "aggregated_cards_in_sample": len(agg_cards_filtered),
            "cards_generated": len(generated_cards),
            "cards_skipped": skipped_cards,
            "errors": len(errors),
            "total_tokens": stats["total_tokens"],
            "total_cost": stats["total_cost"],
            "cache_hits": stats["cache_stats"].get("hits", 0),
        },
        "filtering_stats": {
            "length": length_stats,
            "spoilers": spoiler_stats,
        },
        "cards": generated_cards,
        "errors": errors,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    logger.info(f"Saved {len(generated_cards)} cards to {output_path}")

    # 7. Print summary
    unique_lemmas = len(set(c["lemma"] for c in generated_cards))

    print("\n" + "=" * 60)
    print("PIPELINE A COMPLETE")
    print("=" * 60)
    print(f"Aggregated cards in sample: {len(agg_cards_filtered)}")
    print(f"Cards generated: {len(generated_cards)}")
    print(f"Unique lemmas: {unique_lemmas}")
    print(f"Skipped: {skipped_cards}")
    print(f"Errors: {len(errors)}")
    print("\nAPI stats:")
    print(f"  - Tokens: {stats['total_tokens']:,}")
    print(f"  - Cost: ${stats['total_cost']:.4f}")
    print(f"  - Cache hits: {stats['cache_stats'].get('hits', 0)}")
    print(f"\nElapsed time: {elapsed:.1f}s")
    print(f"Output: {output_path}")
    print("=" * 60)

    return results


if __name__ == "__main__":
    main()
