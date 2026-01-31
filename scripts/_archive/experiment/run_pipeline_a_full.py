#!/usr/bin/env python3
"""
Run Pipeline A (current WSD-based) on experiment sample.

This is the FULL Pipeline A:
1. Filter sense_tokens to sample sentence_ids
2. Run Synset Aggregation (LLM for multi-synset lemmas)
3. Run Card Generation

Usage:
    uv run python scripts/experiment/run_pipeline_a_full.py
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
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from eng_words.aggregation.llm_aggregator import AggregationResult, LLMAggregator, SynsetGroup
from eng_words.aggregation.synset_aggregator import aggregate_by_synset
from eng_words.llm.base import get_provider
from eng_words.llm.response_cache import ResponseCache
from eng_words.llm.smart_card_generator import (
    SmartCard,
    SmartCardGenerator,
    check_spoilers,
    mark_examples_by_length,
    select_examples_for_generation,
)

# sentences are loaded from pre-computed parquet file
from eng_words.validation import validate_examples_for_synset_group

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Paths
DATA_DIR = Path("data/experiment")
SENSE_TOKENS_PATH = Path("data/processed/american_tragedy_wsd_sense_tokens.parquet")
CACHE_DIR = Path("data/cache/llm_responses")
BOOK_NAME = "american_tragedy"


def parse_args():
    parser = argparse.ArgumentParser(description="Run Pipeline A (full WSD-based)")
    parser.add_argument("--limit", type=int, default=None, help="Limit lemmas to process")
    parser.add_argument("--no-cache", action="store_true", help="Disable caching")
    parser.add_argument("--output", type=str, default=None, help="Output file path")
    return parser.parse_args()


def normalize_synset_group(synset_group):
    """Normalize synset_group to a list."""
    if isinstance(synset_group, (np.ndarray, list)):
        return list(synset_group) if len(synset_group) > 0 else []
    elif isinstance(synset_group, str):
        try:
            parsed = json.loads(synset_group)
            return parsed if isinstance(parsed, list) else [parsed]
        except:
            return [synset_group] if synset_group else []
    elif synset_group is None or (hasattr(pd, "isna") and pd.isna(synset_group)):
        return []
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
    logger.info("=" * 70)
    logger.info("PIPELINE A (FULL WSD-BASED) ON SAMPLE")
    logger.info("=" * 70)

    # =========================================================================
    # STEP 1: Load sample and filter sense_tokens
    # =========================================================================
    logger.info("\n## Step 1: Load sample and filter sense_tokens")

    tokens_sample = pd.read_parquet(DATA_DIR / "tokens_sample.parquet")
    sample_sids = set(tokens_sample["sentence_id"].unique())
    logger.info(f"  Sample: {len(sample_sids):,} sentences")

    sense_tokens = pd.read_parquet(SENSE_TOKENS_PATH)
    logger.info(f"  Full sense_tokens: {len(sense_tokens):,} rows")

    # Filter to sample sentences
    sense_tokens_sample = sense_tokens[sense_tokens["sentence_id"].isin(sample_sids)]
    logger.info(f"  Filtered sense_tokens: {len(sense_tokens_sample):,} rows")

    # =========================================================================
    # STEP 2: Synset Aggregation
    # =========================================================================
    logger.info("\n## Step 2: Synset Aggregation")

    synset_stats = aggregate_by_synset(sense_tokens_sample, min_freq=1, filter_dialect=True)
    logger.info(f"  Synset stats: {len(synset_stats):,} (lemma, synset) pairs")

    # Identify multi-synset lemmas
    lemma_counts = synset_stats.groupby("lemma").size()
    single_lemmas = lemma_counts[lemma_counts == 1].index.tolist()
    multi_lemmas = lemma_counts[lemma_counts > 1].index.tolist()

    logger.info(f"  Single synset lemmas: {len(single_lemmas):,} (no LLM needed)")
    logger.info(f"  Multi synset lemmas: {len(multi_lemmas):,} (need LLM)")

    # Initialize LLM
    provider = get_provider("gemini", "gemini-3-flash-preview")
    cache = ResponseCache(CACHE_DIR, enabled=not args.no_cache)

    # Run LLM aggregation for multi-synset lemmas
    aggregator = LLMAggregator(provider=provider, cache=cache)

    multi_df = synset_stats[synset_stats["lemma"].isin(multi_lemmas)]

    logger.info(f"  Running LLM aggregation on {len(multi_lemmas)} lemmas...")
    agg_results = aggregator.aggregate_batch(multi_df, progress=True)

    # Add single-synset lemmas
    single_results = []
    for lemma in single_lemmas:
        row = synset_stats[synset_stats["lemma"] == lemma].iloc[0]
        single_results.append(
            AggregationResult(
                lemma=lemma,
                groups=[
                    SynsetGroup(
                        synset_ids=[row["synset_id"]],
                        primary_synset=row["synset_id"],
                        reason="Single synset",
                    )
                ],
                original_count=1,
                aggregated_count=1,
                llm_cost=0.0,
            )
        )

    all_agg_results = agg_results + single_results

    # Apply aggregation
    cards_df = aggregator.apply_aggregation(synset_stats, all_agg_results)
    logger.info(f"  Aggregated cards: {len(cards_df):,}")

    # Apply limit
    if args.limit:
        # Sort by frequency
        lemma_freq = sense_tokens_sample.groupby("lemma")["sentence_id"].nunique().to_dict()
        cards_df = cards_df.copy()
        cards_df["_freq"] = cards_df["lemma"].map(lemma_freq).fillna(0)
        cards_df = cards_df.sort_values("_freq", ascending=False)
        top_lemmas = cards_df.drop_duplicates("lemma").head(args.limit)["lemma"].tolist()
        cards_df = cards_df[cards_df["lemma"].isin(top_lemmas)]
        cards_df = cards_df.drop(columns=["_freq"])
        logger.info(f"  Limited to {len(cards_df):,} cards for {args.limit} lemmas")

    # =========================================================================
    # STEP 3: Load sentences
    # =========================================================================
    logger.info("\n## Step 3: Load sentences")

    sentences_df = pd.read_parquet(DATA_DIR / "sentences_sample.parquet")
    sentences_lookup = dict(zip(sentences_df["sentence_id"], sentences_df["text"]))
    logger.info(f"  {len(sentences_lookup):,} sentences loaded")

    # =========================================================================
    # STEP 4: Card Generation
    # =========================================================================
    logger.info("\n## Step 4: Card Generation")

    generator = SmartCardGenerator(
        provider=provider,
        cache=cache,
        book_name=BOOK_NAME,
        max_retries=2,
    )

    generated_cards = []
    skipped = 0
    errors = []

    total = len(cards_df)

    for idx, row in tqdm(cards_df.iterrows(), total=total, desc="Generating cards"):
        lemma = row["lemma"]

        try:
            # Get sentence_ids for this card
            sentence_ids = row.get("sentence_ids", [])
            if isinstance(sentence_ids, str):
                try:
                    sentence_ids = json.loads(sentence_ids)
                except:
                    sentence_ids = []

            # Filter to sample sentences
            sentence_ids = [sid for sid in sentence_ids if sid in sentences_lookup]

            if not sentence_ids:
                skipped += 1
                continue

            # Get examples
            examples_with_ids = [(sid, sentences_lookup[sid]) for sid in sentence_ids]

            # Synset info
            synset_group = normalize_synset_group(row.get("synset_group", []))
            primary_synset = row.get("primary_synset", "")

            # Validate examples
            validation = validate_examples_for_synset_group(
                lemma=lemma,
                synset_group=synset_group,
                primary_synset=primary_synset,
                examples=examples_with_ids,
                provider=provider,
                cache=cache,
            )

            if not validation.get("has_valid", False):
                skipped += 1
                continue

            valid_examples = [
                (sid, sentences_lookup[sid])
                for sid in validation.get("valid_sentence_ids", [])
                if sid in sentences_lookup
            ]

            if not valid_examples:
                skipped += 1
                continue

            # Filter by length and spoilers
            length_flags = mark_examples_by_length(valid_examples, max_words=50, min_words=6)
            spoiler_flags = check_spoilers(valid_examples, provider, cache, BOOK_NAME)

            selection = select_examples_for_generation(
                all_examples=valid_examples,
                length_flags=length_flags,
                spoiler_flags=spoiler_flags,
                target_count=3,
            )

            selected_text = [ex for _, ex in selection["selected_from_book"]]
            generate_count = selection["generate_count"]

            # Generate card
            card = generator.generate_card(
                lemma=lemma,
                pos=row.get("pos", "unknown"),
                supersense=row.get("supersense", "unknown"),
                wn_definition=row.get("definition", ""),
                examples=selected_text,
                synset_group=synset_group,
                primary_synset=primary_synset,
                generate_count=generate_count,
            )

            if card:
                card_dict = card_to_dict(card)
                card_dict["source"] = "pipeline_a"
                generated_cards.append(card_dict)
            else:
                skipped += 1

        except Exception as e:
            logger.error(f"Error for '{lemma}': {e}")
            errors.append({"lemma": lemma, "error": str(e)})

    # =========================================================================
    # STEP 5: Save results
    # =========================================================================
    elapsed = time.time() - start_time
    stats = generator.stats()

    output_path = Path(args.output) if args.output else DATA_DIR / "cards_A.json"

    results = {
        "pipeline": "A",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "config": {
            "limit": args.limit,
            "use_wsd": True,
            "use_synset_aggregation": True,
        },
        "stats": {
            "sample_sentences": len(sample_sids),
            "sense_tokens_in_sample": len(sense_tokens_sample),
            "synset_pairs": len(synset_stats),
            "single_synset_lemmas": len(single_lemmas),
            "multi_synset_lemmas": len(multi_lemmas),
            "aggregated_cards": len(cards_df),
            "cards_generated": len(generated_cards),
            "cards_skipped": skipped,
            "errors": len(errors),
            "total_tokens": stats["total_tokens"],
            "total_cost": stats["total_cost"],
            "agg_cost": sum(r.llm_cost for r in agg_results),
            "elapsed_seconds": elapsed,
        },
        "cards": generated_cards,
        "errors": errors,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # Summary
    print("\n" + "=" * 70)
    print("PIPELINE A COMPLETE")
    print("=" * 70)
    print(f"Sample sentences: {len(sample_sids):,}")
    print(f"Synset pairs: {len(synset_stats):,}")
    print(f"Aggregated cards: {len(cards_df):,}")
    print(f"Cards generated: {len(generated_cards):,}")
    print(f"Unique lemmas: {len(set(c['lemma'] for c in generated_cards)):,}")
    print(f"Skipped: {skipped:,}")
    print(f"Errors: {len(errors)}")
    print(f"\nCost: ${stats['total_cost'] + sum(r.llm_cost for r in agg_results):.4f}")
    print(f"Time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"Output: {output_path}")
    print("=" * 70)

    return results


if __name__ == "__main__":
    main()
