#!/usr/bin/env python3
"""Full test of SmartCardGenerator on American Tragedy.

Runs smart card generation on the full book with:
- Gemini Flash as provider
- Response caching enabled
- Progress logging
- Cost tracking
"""

import json
import logging
import time
from pathlib import Path

import pandas as pd
from nltk.corpus import wordnet as wn

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def get_wn_definition(synset_id: str) -> str:
    """Get WordNet definition for a synset ID."""
    if not synset_id or pd.isna(synset_id):
        return ""
    try:
        synset = wn.synset(synset_id)
        return synset.definition()
    except Exception:
        return ""


def get_examples_for_lemma(
    lemma: str,
    tokens_df: pd.DataFrame,
    sentences_df: pd.DataFrame,
    max_examples: int = 10,
) -> list[str]:
    """Get example sentences for a lemma."""
    # Find sentence IDs containing this lemma
    lemma_tokens = tokens_df[tokens_df["lemma"] == lemma]
    if lemma_tokens.empty:
        return []

    sentence_ids = lemma_tokens["sentence_id"].unique()[:max_examples]

    examples = []
    for sid in sentence_ids:
        sent_row = sentences_df[sentences_df["sentence_id"] == sid]
        if not sent_row.empty:
            examples.append(sent_row.iloc[0]["sentence"])  # Fixed column name

    return examples


def main():
    from eng_words.llm.base import get_provider
    from eng_words.llm.response_cache import ResponseCache
    from eng_words.llm.smart_card_generator import SmartCardGenerator
    from eng_words.text_processing import (
        create_sentences_dataframe,
        reconstruct_sentences_from_tokens,
    )

    print("=" * 70)
    print("SMART CARD GENERATION: Full Test on American Tragedy")
    print("=" * 70)

    # Paths
    stats_path = Path("data/processed/american_tragedy_wsd_supersense_stats.parquet")
    tokens_path = Path("data/processed/american_tragedy_wsd_sense_tokens.parquet")
    output_dir = Path("data/output/american_tragedy_smart_cards")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    logger.info("Loading data...")
    stats_df = pd.read_parquet(stats_path)
    tokens_df = pd.read_parquet(tokens_path)

    # Reconstruct sentences
    logger.info("Reconstructing sentences...")
    sentences = reconstruct_sentences_from_tokens(tokens_df)
    sentences_df = create_sentences_dataframe(sentences)
    logger.info(f"  {len(sentences_df)} sentences")

    # Filter supersenses
    logger.info("Filtering supersenses...")
    df = stats_df[stats_df["supersense"] != "unknown"].copy()
    df = df[df["sense_freq"] >= 2]
    logger.info(f"  {len(df)} supersenses after filtering")

    # Sort by sense_freq (most common first)
    df = df.sort_values("sense_freq", ascending=False)

    # Initialize generator
    logger.info("Initializing SmartCardGenerator...")
    provider = get_provider("gemini")
    cache = ResponseCache(cache_dir=Path("data/cache/llm_responses"))

    generator = SmartCardGenerator(
        provider=provider,
        cache=cache,
        book_name="An American Tragedy",
    )

    # Prepare items
    logger.info("Preparing items...")
    items = []

    # Get unique synset_id for each (lemma, supersense)
    synset_map = {}
    if "synset_id" in tokens_df.columns:
        for _, row in tokens_df[["lemma", "synset_id"]].drop_duplicates().iterrows():
            if pd.notna(row["synset_id"]):
                synset_map[row["lemma"]] = row["synset_id"]

    for _, row in df.iterrows():
        lemma = row["lemma"]
        supersense = row["supersense"]
        pos = supersense.split(".")[0] if "." in supersense else "unknown"

        # Get WordNet definition
        synset_id = synset_map.get(lemma, "")
        wn_definition = get_wn_definition(synset_id)

        # Get examples
        examples = get_examples_for_lemma(lemma, tokens_df, sentences_df, max_examples=10)

        if not examples:
            continue

        items.append(
            {
                "lemma": lemma,
                "pos": pos,
                "supersense": supersense,
                "wn_definition": wn_definition,
                "examples": examples,
            }
        )

    logger.info(f"  {len(items)} items prepared")

    # Generate cards
    logger.info("=" * 70)
    logger.info("Starting card generation...")
    start_time = time.time()

    cards = []
    failed = 0

    for i, item in enumerate(items):
        try:
            card = generator.generate_card(**item)
            if card:
                cards.append(card)
            else:
                failed += 1

            # Progress every 100 cards
            if (i + 1) % 100 == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed
                eta = (len(items) - i - 1) / rate
                stats = generator.stats()
                logger.info(
                    f"[{i+1:5}/{len(items)}] "
                    f"✓{stats['successful']} ✗{stats['failed']} "
                    f"${stats['total_cost']:.3f} "
                    f"ETA: {eta/60:.1f}min"
                )

        except Exception as e:
            logger.error(f"Error generating card for '{item['lemma']}': {e}")
            failed += 1

    elapsed = time.time() - start_time

    # Final stats
    stats = generator.stats()
    cache_stats = stats["cache_stats"]

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print("\n## Generation Stats")
    print(f"   Total items: {len(items)}")
    print(f"   Successful: {stats['successful']}")
    print(f"   Failed: {stats['failed']}")
    print(f"   Success rate: {stats['successful']/len(items)*100:.1f}%")

    print("\n## Cost & Performance")
    print(f"   Total cost: ${stats['total_cost']:.4f}")
    print(f"   Cost per card: ${stats['total_cost']/max(stats['successful'],1):.6f}")
    print(f"   Total time: {elapsed/60:.1f} minutes")
    print(f"   Cards/minute: {stats['successful']/elapsed*60:.1f}")

    print("\n## Cache Stats")
    print(f"   Hits: {cache_stats['hits']}")
    print(f"   Misses: {cache_stats['misses']}")
    print(f"   Hit rate: {cache_stats['hit_rate']*100:.1f}%")
    print(f"   Cost saved: ${cache_stats['cost_saved']:.4f}")

    # Save results
    logger.info("Saving results...")

    # JSON output
    json_path = output_dir / "smart_cards.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump([card.to_dict() for card in cards], f, ensure_ascii=False, indent=2)

    # Anki CSV
    anki_rows = [card.to_anki_row() for card in cards]
    anki_df = pd.DataFrame(anki_rows)
    anki_path = output_dir / "smart_anki.csv"
    anki_df.to_csv(anki_path, index=False, sep="\t")

    # Stats summary
    summary = {
        "total_items": len(items),
        "successful": stats["successful"],
        "failed": stats["failed"],
        "total_cost": stats["total_cost"],
        "total_time_seconds": elapsed,
        "cache_hits": cache_stats["hits"],
        "cache_misses": cache_stats["misses"],
    }
    summary_path = output_dir / "generation_stats.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n## Output Files")
    print(f"   Cards JSON: {json_path}")
    print(f"   Anki CSV: {anki_path}")
    print(f"   Stats: {summary_path}")

    print("\n✅ Full test complete!")


if __name__ == "__main__":
    main()
