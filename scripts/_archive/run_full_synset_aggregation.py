#!/usr/bin/env python3
"""
Run full LLM-based synset aggregation on all lemmas.

Usage:
    uv run python scripts/run_full_synset_aggregation.py
"""

import json
import logging
import time
from dataclasses import asdict
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

from eng_words.aggregation.llm_aggregator import LLMAggregator
from eng_words.aggregation.synset_aggregator import aggregate_by_synset
from eng_words.llm.base import get_provider
from eng_words.llm.response_cache import ResponseCache

load_dotenv()

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

# Paths
SENSE_TOKENS_PATH = Path("data/processed/american_tragedy_wsd_sense_tokens.parquet")
OUTPUT_DIR = Path("data/synset_aggregation_full")
CACHE_DIR = OUTPUT_DIR / "llm_cache"


def run_full_aggregation():
    """Run full synset aggregation on American Tragedy."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("FULL SYNSET AGGREGATION")
    logger.info("=" * 70)

    # Step 1: Load and aggregate by synset_id
    logger.info("\n## Step 1: Aggregate by synset_id")
    sense_tokens_df = pd.read_parquet(SENSE_TOKENS_PATH)
    logger.info(f"  Loaded {len(sense_tokens_df):,} tokens")

    synset_stats_df = aggregate_by_synset(sense_tokens_df, min_freq=2, filter_dialect=True)
    logger.info(f"  Aggregated to {len(synset_stats_df):,} (lemma, synset) pairs (with dialect filter)")

    # Save synset stats
    synset_stats_path = OUTPUT_DIR / "synset_stats.parquet"
    synset_stats_df.to_parquet(synset_stats_path)
    logger.info(f"  Saved to {synset_stats_path}")

    # Step 2: Identify lemmas needing LLM aggregation
    logger.info("\n## Step 2: Identify lemmas for LLM aggregation")
    lemma_counts = synset_stats_df.groupby("lemma").size()
    single_synset_lemmas = lemma_counts[lemma_counts == 1].index.tolist()
    multi_synset_lemmas = lemma_counts[lemma_counts > 1].index.tolist()

    logger.info(f"  Single synset lemmas: {len(single_synset_lemmas):,} (no LLM needed)")
    logger.info(f"  Multi synset lemmas:  {len(multi_synset_lemmas):,} (need LLM)")

    # Step 3: Run LLM aggregation
    logger.info("\n## Step 3: Run LLM aggregation")

    provider = get_provider("gemini", "gemini-3-flash-preview")
    cache = ResponseCache(cache_dir=CACHE_DIR, enabled=True)
    aggregator = LLMAggregator(provider=provider, cache=cache)

    start_time = time.time()

    # Filter to multi-synset lemmas only for LLM aggregation
    multi_synset_df = synset_stats_df[synset_stats_df["lemma"].isin(multi_synset_lemmas)]

    results = aggregator.aggregate_batch(multi_synset_df, progress=True)

    end_time = time.time()
    elapsed = end_time - start_time

    # Step 4: Apply aggregation
    logger.info("\n## Step 4: Apply aggregation results")

    # Add single-synset lemmas as-is
    from eng_words.aggregation.llm_aggregator import AggregationResult, SynsetGroup

    single_results = []
    for lemma in single_synset_lemmas:
        lemma_row = synset_stats_df[synset_stats_df["lemma"] == lemma].iloc[0]
        single_results.append(
            AggregationResult(
                lemma=lemma,
                groups=[
                    SynsetGroup(
                        synset_ids=[lemma_row["synset_id"]],
                        primary_synset=lemma_row["synset_id"],
                        reason="Single synset",
                    )
                ],
                original_count=1,
                aggregated_count=1,
                llm_cost=0.0,
            )
        )

    all_results = results + single_results

    # Apply to create final cards DataFrame
    cards_df = aggregator.apply_aggregation(synset_stats_df, all_results)
    logger.info(f"  Created {len(cards_df):,} card items")

    # Save results
    results_json_path = OUTPUT_DIR / "aggregation_results.json"
    with open(results_json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "summary": {
                    "total_lemmas": len(all_results),
                    "single_synset": len(single_synset_lemmas),
                    "multi_synset": len(multi_synset_lemmas),
                    "original_synsets": len(synset_stats_df),
                    "final_cards": len(cards_df),
                    "reduction_pct": round(
                        (1 - len(cards_df) / len(synset_stats_df)) * 100, 1
                    ),
                    "llm_cost": round(sum(r.llm_cost for r in results), 4),
                    "elapsed_seconds": round(elapsed, 1),
                },
                "multi_synset_results": [
                    {
                        "lemma": r.lemma,
                        "original": r.original_count,
                        "aggregated": r.aggregated_count,
                        "groups": [asdict(g) for g in r.groups],
                        "cost": r.llm_cost,
                    }
                    for r in results
                ],
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    logger.info(f"  Saved results to {results_json_path}")

    # Save cards DataFrame
    cards_path = OUTPUT_DIR / "aggregated_cards.parquet"
    cards_df.to_parquet(cards_path)
    logger.info(f"  Saved cards to {cards_path}")

    # Print summary
    stats = aggregator.get_stats()
    cache_stats = cache.stats()

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Total lemmas:      {len(all_results):,}")
    print(f"    - Single synset: {len(single_synset_lemmas):,}")
    print(f"    - Multi synset:  {len(multi_synset_lemmas):,}")
    print(f"  Original synsets:  {len(synset_stats_df):,}")
    print(f"  Final cards:       {len(cards_df):,}")
    print(f"  Reduction:         {(1 - len(cards_df) / len(synset_stats_df)) * 100:.1f}%")
    print(f"  LLM Cost:          ${sum(r.llm_cost for r in results):.4f}")
    print(f"  Time:              {elapsed:.1f}s ({len(multi_synset_lemmas) / elapsed:.1f} lemmas/s)")
    print(f"  Cache hits:        {cache_stats['hits']} ({cache_stats['hit_rate']})")

    return cards_df, all_results


if __name__ == "__main__":
    run_full_aggregation()

