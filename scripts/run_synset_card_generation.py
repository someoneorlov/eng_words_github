#!/usr/bin/env python3
"""
Generate Smart Cards using synset-based aggregation.

Uses the aggregated cards from run_full_synset_aggregation.py and generates
flashcards with SmartCardGenerator.

Usage:
    uv run python scripts/run_synset_card_generation.py
"""

import json
import logging
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from eng_words.anki_export import export_to_anki_csv
from eng_words.llm.base import get_provider
from eng_words.llm.response_cache import ResponseCache
from eng_words.llm.smart_card_generator import (
    SmartCard,
    SmartCardGenerator,
    check_spoilers,
    mark_examples_by_length,
    select_examples_for_generation,
)
from eng_words.text_processing import create_sentences_dataframe, reconstruct_sentences_from_tokens
from eng_words.validation import fix_invalid_cards, validate_examples_for_synset_group
from eng_words.wsd.llm_wsd import redistribute_empty_cards

load_dotenv()

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

# Paths
BOOK_NAME = "american_tragedy"
AGGREGATED_CARDS_PATH = Path("data/synset_aggregation_full/aggregated_cards.parquet")
TOKENS_PATH = Path(f"data/processed/{BOOK_NAME}_tokens.parquet")
OUTPUT_DIR = Path("data/synset_cards")
CACHE_DIR = OUTPUT_DIR / "llm_cache"


def run_card_generation(limit: int | None = None):
    """Generate Smart Cards using synset-based aggregation."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("SYNSET-BASED CARD GENERATION")
    logger.info("=" * 70)

    # Step 1: Load aggregated cards
    logger.info("\n## Step 1: Load aggregated cards")
    cards_df = pd.read_parquet(AGGREGATED_CARDS_PATH)
    logger.info(f"  Loaded {len(cards_df):,} card items")

    # Step 2: Reconstruct sentences for examples
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

    # Step 4: Prepare items and generate cards
    logger.info("\n## Step 4: Generate cards")
    
    if limit:
        cards_df = cards_df.head(limit)
        logger.info(f"  Limited to {limit} cards")

    generated_cards = []
    skipped_cards = 0  # Counter for cards skipped due to validation
    start_time = time.time()
    
    # Statistics for Stage 2.5 filtering
    length_stats = {"too_long": 0, "appropriate_length": 0, "too_short": 0}
    spoiler_stats = {"has_spoiler": 0, "no_spoiler": 0}
    selection_stats = defaultdict(int)
    
    # Check for partial results to resume
    partial_path = OUTPUT_DIR / "synset_smart_cards_partial.json"
    final_path = OUTPUT_DIR / "synset_smart_cards_final.json"
    resume_from = 0
    if partial_path.exists():
        logger.info(f"  Found partial results, loading...")
        try:
            with open(partial_path, "r", encoding="utf-8") as f:
                partial_data = json.load(f)
                generated_cards = [
                    SmartCard(**c) for c in partial_data
                ]
                resume_from = len(generated_cards)
                logger.info(f"  Resuming from card {resume_from}/{len(cards_df)}")
        except Exception as e:
            logger.warning(f"  Failed to load partial results: {e}, starting fresh")
    elif final_path.exists():
        # If no partial but final exists, we can't resume (final has processed cards)
        # But we'll start fresh - cache will handle already generated cards
        logger.info(f"  Final file exists ({final_path}), starting fresh generation")
        logger.info(f"  Cache will help speed up already generated cards")

    from tqdm import tqdm
    import time as time_module
    from google.genai.errors import ServerError
    import numpy as np
    
    # Helper functions for serialization
    def make_serializable(obj):
        """Recursively convert numpy arrays to lists."""
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

    def card_to_serializable(card: SmartCard) -> dict:
        """Convert SmartCard to JSON-serializable dict."""
        d = asdict(card)
        return make_serializable(d)
    
    def normalize_synset_group(synset_group):
        """Normalize synset_group to a list."""
        if isinstance(synset_group, (np.ndarray, list)):
            return list(synset_group) if len(synset_group) > 0 else []
        elif isinstance(synset_group, str):
            try:
                synset_group = json.loads(synset_group)
                if not isinstance(synset_group, list):
                    synset_group = [synset_group]
                return synset_group
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
    
    # Save checkpoint every N cards
    CHECKPOINT_INTERVAL = 100
    
    # Convert to list of tuples for easier indexing
    cards_list = list(cards_df.iterrows())
    
    for i, (idx, row) in enumerate(tqdm(cards_list[resume_from:], total=len(cards_df)-resume_from, desc="Generating cards", initial=resume_from)):
        # Initialize variables for retry block
        selected_examples_text = []
        generate_count = 0
        synset_group = []
        primary_synset = ""
        
        try:
            # Get examples from sentence_ids
            sentence_ids = row.get("sentence_ids", [])
            examples_with_ids = [
                (sid, sentences_lookup.get(sid, ""))
                for sid in sentence_ids
                if sid in sentences_lookup and sentences_lookup.get(sid, "")
            ]

            if not examples_with_ids:
                # Skip items with no examples
                skipped_cards += 1
                continue

            # Prepare synset_group info
            synset_group = normalize_synset_group(row.get("synset_group", []))
            primary_synset = row.get("primary_synset", "")
            
            # VALIDATION: Check if examples are valid for synset_group
            validation = validate_examples_for_synset_group(
                lemma=row["lemma"],
                synset_group=synset_group,
                primary_synset=primary_synset,
                examples=examples_with_ids,
                provider=provider,
                cache=cache,
            )
            
            # Skip if no valid examples
            if not validation["has_valid"]:
                logger.debug(f"Skipping {row['lemma']} ({primary_synset}) - no valid examples for synset_group")
                skipped_cards += 1
                continue
            
            # Get valid examples as (sentence_id, sentence) tuples
            valid_examples = [
                (sid, sentences_lookup[sid])
                for sid in validation["valid_sentence_ids"]
                if sid in sentences_lookup
            ]
            
            if not valid_examples:
                # This shouldn't happen if validation worked correctly, but just in case
                logger.warning(f"No valid examples after validation for {row['lemma']}, skipping")
                skipped_cards += 1
                continue

            # STEP 2.5: Filter by length and spoilers BEFORE generation
            # 1. Mark examples by length (don't filter, just mark)
            length_flags = mark_examples_by_length(valid_examples, max_words=50, min_words=6)
            
            # Statistics for length
            # length_flags[sid] = True if min_words <= word_count <= max_words, False otherwise
            too_long_count = 0
            too_short_count = 0
            appropriate_count = 0
            
            for sid, ex in valid_examples:
                word_count = len(ex.split())
                if word_count < 6:
                    too_short_count += 1
                elif word_count > 50:
                    too_long_count += 1
                else:
                    appropriate_count += 1
            
            length_stats["too_long"] += too_long_count
            length_stats["too_short"] += too_short_count
            length_stats["appropriate_length"] += appropriate_count
            
            # 2. Check spoilers (mark with flags)
            spoiler_flags = check_spoilers(
                examples=valid_examples,
                provider=provider,
                cache=cache,
                book_name=BOOK_NAME,
            )
            
            # Statistics for spoilers
            spoiler_count = sum(1 for v in spoiler_flags.values() if v)
            spoiler_stats["has_spoiler"] += spoiler_count
            spoiler_stats["no_spoiler"] += len(spoiler_flags) - spoiler_count
            
            # 3. Select examples for generation based on flags
            selection = select_examples_for_generation(
                all_examples=valid_examples,
                length_flags=length_flags,
                spoiler_flags=spoiler_flags,
                target_count=3,
            )
            
            # Statistics for selection
            selected_count = len(selection["selected_from_book"])
            generate_count = selection["generate_count"]
            if selected_count == 2 and generate_count == 1:
                selection_stats["2+1"] += 1
            elif selected_count >= 1 and generate_count >= 1:
                selection_stats[f"{selected_count}+{generate_count}"] += 1
            elif selected_count == 0:
                selection_stats["0+3"] += 1
            
            # 4. Prepare examples text for generation
            selected_examples_text = [ex for _, ex in selection["selected_from_book"]]

            # Track existing synsets (for duplicate detection, not fallback)
            existing_synsets = {c.primary_synset for c in generated_cards if hasattr(c, 'primary_synset')}
            
            # 5. Generate card with selected examples and generate_count
            card = generator.generate_card(
                lemma=row["lemma"],
                pos=row["pos"],
                supersense=row.get("supersense", "unknown"),
                wn_definition=row["definition"],
                examples=selected_examples_text,  # Only pre-filtered examples
                synset_group=synset_group,
                primary_synset=primary_synset,
                generate_count=generate_count,  # How many to generate (1, 2, or 3)
                existing_synsets=existing_synsets,
            )

            if card:
                generated_cards.append(card)
            
            # Save checkpoint periodically
            if len(generated_cards) % CHECKPOINT_INTERVAL == 0:
                try:
                    partial_data = [card_to_serializable(c) for c in generated_cards]
                    with open(partial_path, "w", encoding="utf-8") as f:
                        json.dump(partial_data, f, ensure_ascii=False, indent=2)
                    logger.debug(f"  Checkpoint saved: {len(generated_cards)} cards")
                except Exception as e:
                    logger.warning(f"  Failed to save checkpoint: {e}")
        
        except ServerError as e:
            # Handle 503 and other server errors with retry
            if "503" in str(e) or "UNAVAILABLE" in str(e):
                logger.warning(f"  Server error (503) at card {i+resume_from}, saving checkpoint and waiting...")
                # Save current progress
                try:
                    partial_data = [card_to_serializable(c) for c in generated_cards]
                    with open(partial_path, "w", encoding="utf-8") as f:
                        json.dump(partial_data, f, ensure_ascii=False, indent=2)
                    logger.info(f"  Progress saved: {len(generated_cards)} cards")
                except:
                    pass
                # Wait and retry
                logger.info("  Waiting 30 seconds before retry...")
                time_module.sleep(30)
                # Retry this card (reuse selection from before error)
                try:
                    card = generator.generate_card(
                        lemma=row["lemma"],
                        pos=row["pos"],
                        supersense=row.get("supersense", "unknown"),
                        wn_definition=row["definition"],
                        examples=selected_examples_text,
                        synset_group=synset_group,
                        primary_synset=primary_synset,
                        generate_count=generate_count,
                        existing_synsets=existing_synsets,
                    )
                    if card:
                        generated_cards.append(card)
                except Exception as retry_e:
                    logger.error(f"  Retry failed for {row['lemma']}: {retry_e}")
                    logger.info(f"  To resume, run script again - it will continue from card {len(generated_cards)}")
                    raise
            else:
                raise
        except Exception as e:
            logger.error(f"  Error generating card for {row.get('lemma', 'unknown')}: {e}")
            # Save progress before failing
            try:
                partial_data = [card_to_serializable(c) for c in generated_cards]
                with open(partial_path, "w", encoding="utf-8") as f:
                    json.dump(partial_data, f, ensure_ascii=False, indent=2)
                logger.info(f"  Progress saved: {len(generated_cards)} cards before error")
            except:
                pass
            raise

    end_time = time.time()
    elapsed = end_time - start_time
    
    # Log validation and filtering statistics
    logger.info(f"\n## Validation Statistics")
    logger.info(f"  Total cards processed: {len(cards_df):,}")
    logger.info(f"  Cards skipped (no valid examples): {skipped_cards:,} ({skipped_cards/max(len(cards_df),1)*100:.1f}%)")
    logger.info(f"  Cards generated: {len(generated_cards):,} ({len(generated_cards)/max(len(cards_df),1)*100:.1f}%)")
    
    logger.info(f"\n## Stage 2.5 Filtering Statistics")
    logger.info(f"  Length filtering:")
    logger.info(f"    - Too long (>50 words): {length_stats['too_long']}")
    logger.info(f"    - Too short (<6 words): {length_stats['too_short']}")
    logger.info(f"    - Appropriate length: {length_stats['appropriate_length']}")
    logger.info(f"  Spoiler filtering:")
    logger.info(f"    - With spoilers: {spoiler_stats['has_spoiler']}")
    logger.info(f"    - Without spoilers: {spoiler_stats['no_spoiler']}")
    logger.info(f"  Selection patterns:")
    for pattern, count in sorted(selection_stats.items()):
        logger.info(f"    - {pattern}: {count}")
    
    # Final checkpoint
    if partial_path.exists():
        partial_path.unlink()  # Remove partial file after successful completion

    # Step 5: LLM WSD Redistribution
    logger.info("\n## Step 5: LLM WSD Redistribution")
    start_wsd = time.time()
    cards_after_wsd = redistribute_empty_cards(generated_cards, provider, cache)
    wsd_time = time.time() - start_wsd
    logger.info(f"  Cards after WSD: {len(cards_after_wsd):,}")
    logger.info(f"  Time: {wsd_time:.1f}s")

    # Step 6: Example Validation
    logger.info("\n## Step 6: Example Validation")
    start_val = time.time()
    final_cards, removed_cards = fix_invalid_cards(
        cards_after_wsd,
        use_generated_example=True,
        remove_unfixable=True,
    )
    val_time = time.time() - start_val
    logger.info(f"  Valid cards: {len(final_cards):,}")
    logger.info(f"  Removed: {len(removed_cards):,}")
    logger.info(f"  Time: {val_time:.1f}s")

    # Step 7: Analyze results
    logger.info("\n## Step 7: Analyze results")
    
    cards_with_examples = [c for c in final_cards if c.selected_examples]
    cards_without_examples = [c for c in final_cards if not c.selected_examples]
    cards_with_translation = [c for c in final_cards if c.translation_ru]

    stats = generator.stats()
    cache_stats = cache.stats()

    logger.info(f"  Total final cards:      {len(final_cards):,}")
    logger.info(f"  With selected examples: {len(cards_with_examples):,}")
    logger.info(f"  Without examples:      {len(cards_without_examples):,} ({len(cards_without_examples)/max(len(final_cards),1)*100:.1f}%)")
    logger.info(f"  With translation:      {len(cards_with_translation):,} ({len(cards_with_translation)/max(len(final_cards),1)*100:.1f}%)")
    logger.info(f"  LLM Cost:              ${stats['total_cost']:.4f}")
    logger.info(f"  Total time:            {elapsed + wsd_time + val_time:.1f}s")
    logger.info(f"  Cache hits:            {cache_stats.get('hits', 0)}")

    # Step 8: Save results
    logger.info("\n## Step 8: Save results")

    # Save full JSON (final cards after all processing)
    json_path = OUTPUT_DIR / "synset_smart_cards_final.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump([card_to_serializable(c) for c in final_cards], f, ensure_ascii=False, indent=2)
    logger.info(f"  Final JSON: {json_path}")

    # Save filtered (with examples only)
    filtered_path = OUTPUT_DIR / "synset_smart_cards_filtered.json"
    with open(filtered_path, "w", encoding="utf-8") as f:
        json.dump([card_to_serializable(c) for c in cards_with_examples], f, ensure_ascii=False, indent=2)
    logger.info(f"  Filtered JSON: {filtered_path}")

    # Export to Anki CSV
    if cards_with_examples:
        anki_rows = [c.to_anki_row() for c in cards_with_examples]
        anki_df = pd.DataFrame(anki_rows)
        anki_path = OUTPUT_DIR / "synset_anki.csv"
        export_to_anki_csv(anki_df, anki_path)
        logger.info(f"  Anki CSV: {anki_path}")

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Input aggregated cards: {len(cards_df):,}")
    print(f"  Cards skipped (validation): {skipped_cards:,} ({skipped_cards/max(len(cards_df),1)*100:.1f}%)")
    print(f"  Total final cards:      {len(final_cards):,}")
    print(f"  Cards with examples:    {len(cards_with_examples):,} ({len(cards_with_examples)/max(len(final_cards),1)*100:.1f}%)")
    print(f"  Cards without examples: {len(cards_without_examples):,} ({len(cards_without_examples)/max(len(final_cards),1)*100:.1f}%)")
    print(f"  Cards with translation: {len(cards_with_translation):,} ({len(cards_with_translation)/max(len(final_cards),1)*100:.1f}%)")
    print(f"  Cards removed:          {len(removed_cards):,}")
    print(f"  LLM Cost:               ${stats['total_cost']:.4f}")
    print(f"  Cache hits:             {cache_stats.get('hits', 0)}")
    print(f"  Total time:             {elapsed + wsd_time + val_time:.1f}s")

    # Compare with supersense-based approach
    print("\n## Comparison with supersense-based approach:")
    print(f"  Old approach: ~16% cards without examples (326/2000)")
    print(f"  New approach: {len(cards_without_examples)/max(len(final_cards),1)*100:.1f}% cards without examples")
    improvement = 16 - len(cards_without_examples)/max(len(final_cards),1)*100
    print(f"  Improvement:  {improvement:.1f}pp")
    
    if len(cards_without_examples) == 0:
        print("  ✅ SUCCESS: 0% cards without examples!")

    return final_cards, cards_with_examples


if __name__ == "__main__":
    import sys
    
    # Check for limit argument
    limit = None
    if len(sys.argv) > 1:
        try:
            limit = int(sys.argv[1])
            logger.info(f"Running with limit: {limit}")
        except ValueError:
            logger.warning(f"Invalid limit argument: {sys.argv[1]}, running full")
    
    run_card_generation(limit=limit)

