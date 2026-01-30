#!/usr/bin/env python3
"""Integration test for WSD Gold Dataset pipeline with real data.

This script tests stages 0-3 with actual data:
1. Stage 1: Extract examples from real tokens_df/sentences_df
2. Stage 2: Apply stratified sampling
3. Stage 3: Make real LLM calls (optional, requires API keys)

Usage:
    python scripts/test_gold_pipeline_real_data.py [--with-llm]
"""

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from eng_words.wsd_gold.collect import (  # noqa: E402
    extract_examples_from_tokens,
)
from eng_words.wsd_gold.models import GoldExample  # noqa: E402
from eng_words.wsd_gold.sample import (  # noqa: E402
    calculate_difficulty_features,
    classify_difficulty,
    get_sampling_stats,
    stratified_sample,
)


def reconstruct_sentences(tokens_df: pd.DataFrame) -> pd.DataFrame:
    """Reconstruct sentence texts from tokens."""
    sentences = []
    for sent_id, group in tokens_df.groupby("sentence_id"):
        group = group.sort_values("position")
        parts = []
        for _, row in group.iterrows():
            parts.append(row["surface"])
            ws = row.get("whitespace", " ")
            if ws:
                parts.append(ws)
        text = "".join(parts).strip()
        sentences.append({"sentence_id": sent_id, "sentence": text})
    return pd.DataFrame(sentences)


def load_real_data() -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """Load real processed data from a book."""
    data_dir = Path("data/processed")

    # Use sense_tokens which has synset_id column
    tokens_file = data_dir / "american_tragedy_wsd_sense_tokens.parquet"
    if not tokens_file.exists():
        tokens_file = data_dir / "american_tragedy_wsd_tokens.parquet"

    if not tokens_file.exists():
        print("❌ No processed data found in data/processed/")
        sys.exit(1)

    print(f"📖 Loading data from: {tokens_file.name}")
    tokens_df = pd.read_parquet(tokens_file)

    # Check for synset_id column
    if "synset_id" not in tokens_df.columns:
        print("❌ No 'synset_id' column - need WSD-processed data")
        sys.exit(1)

    # Reconstruct sentences from tokens
    print("📝 Reconstructing sentences from tokens...")
    sentences_df = reconstruct_sentences(tokens_df)

    book_name = tokens_df["book"].iloc[0] if "book" in tokens_df.columns else "unknown"

    # Count tokens with synset_id
    with_synset = tokens_df[tokens_df["synset_id"].notna() & (tokens_df["synset_id"] != "")]
    print(f"  Tokens: {len(tokens_df):,} rows")
    print(f"  With synset_id: {len(with_synset):,} rows")
    print(f"  Sentences: {len(sentences_df):,} rows")

    return tokens_df, sentences_df, book_name


def test_stage1_collect(
    tokens_df: pd.DataFrame, sentences_df: pd.DataFrame, book_name: str
) -> list[GoldExample]:
    """Test Stage 1: Collection."""
    print("\n" + "=" * 60)
    print("📦 STAGE 1: Collection (extract_examples_from_tokens)")
    print("=" * 60)

    # Sample metadata - keyed by book name from tokens_df
    source_metadata = {
        book_name: {
            "year_bucket": "pre_1950",
            "genre_bucket": "fiction",
            "source_bucket": "classic_fiction",
        }
    }

    try:
        examples = extract_examples_from_tokens(
            tokens_df=tokens_df,
            sentences_df=sentences_df,
            source_metadata=source_metadata,
            min_sense_count=2,  # At least 2 senses for ambiguity
            pos_filter=["NOUN", "VERB"],  # Focus on nouns and verbs
        )

        print(f"✅ Extracted {len(examples)} examples")

        if examples:
            # Show first example
            ex = examples[0]
            print("\n📝 First example:")
            print(f"  ID: {ex.example_id}")
            print(f"  Context: '{ex.context_window[:80]}...'")
            print(f"  Target: {ex.target.lemma} ({ex.target.pos})")
            print(f"  Candidates: {len(ex.candidates)}")
            for i, c in enumerate(ex.candidates[:3]):
                print(f"    {i + 1}. {c.synset_id}: {c.gloss[:50]}...")

            # Show variety stats
            lemmas = set(ex.target.lemma for ex in examples)
            print("\n📊 Stats:")
            print(f"  Unique lemmas: {len(lemmas)}")
            print(f"  Examples per lemma: {len(examples) / len(lemmas):.1f}")

        return examples

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()
        return []


def test_stage2_sample(examples: list[GoldExample]) -> list[GoldExample]:
    """Test Stage 2: Stratified sampling."""
    print("\n" + "=" * 60)
    print("🎲 STAGE 2: Stratified Sampling")
    print("=" * 60)

    if not examples:
        print("⚠️ No examples to sample from")
        return []

    try:
        # Calculate difficulty for all examples
        print("\n📊 Difficulty distribution:")
        difficulties = {"easy": 0, "medium": 0, "hard": 0}
        for ex in examples:
            features = calculate_difficulty_features(ex)
            difficulty = classify_difficulty(features.wn_sense_count, features.baseline_margin)
            difficulties[difficulty] += 1

        total = sum(difficulties.values())
        for d, count in difficulties.items():
            pct = count / total * 100 if total > 0 else 0
            print(f"  {d}: {count} ({pct:.1f}%)")

        # Sample 50 examples
        sample_size = min(50, len(examples))
        print(f"\n🎯 Sampling {sample_size} examples...")

        sampled = stratified_sample(
            examples,
            n=sample_size,
            random_state=42,
        )

        print(f"✅ Sampled {len(sampled)} examples")

        # Show stats
        stats = get_sampling_stats(sampled)
        print("\n📊 Sample stats:")
        print(f"  Difficulty: {stats.get('difficulty_distribution', {})}")
        print(f"  Sources: {stats.get('source_distribution', {})}")

        return sampled

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()
        return []


def test_stage3_providers(examples: list[GoldExample], with_llm: bool) -> dict[str, bool]:
    """Test Stage 3: LLM Providers."""
    print("\n" + "=" * 60)
    print("🤖 STAGE 3: LLM Providers")
    print("=" * 60)

    results = {"openai": False, "anthropic": False, "gemini": False}
    missing_keys = []

    if not with_llm:
        print("⏭️ Skipping LLM calls (use --with-llm to enable)")
        return results

    if not examples:
        print("⚠️ No examples to label")
        return results

    # Check all API keys first
    openai_key = os.environ.get("OPENAI_API_KEY")
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    google_key = os.environ.get("GOOGLE_API_KEY")

    print("\n📋 API Keys status:")
    print(f"  OPENAI_API_KEY: {'✅ set' if openai_key else '❌ missing'}")
    print(f"  ANTHROPIC_API_KEY: {'✅ set' if anthropic_key else '❌ missing'}")
    print(f"  GOOGLE_API_KEY: {'✅ set' if google_key else '❌ missing'}")

    if not openai_key:
        missing_keys.append("OPENAI_API_KEY")
    if not anthropic_key:
        missing_keys.append("ANTHROPIC_API_KEY")
    if not google_key:
        missing_keys.append("GOOGLE_API_KEY")

    if missing_keys:
        print(f"\n⚠️ Missing keys: {', '.join(missing_keys)}")
        print("📖 See docs/LLM_API_KEYS_SETUP.md for instructions")

    # Test OpenAI
    if openai_key:
        print("\n🔵 Testing OpenAI provider...")
        try:
            from eng_words.wsd_gold.providers import OpenAIGoldProvider

            provider = OpenAIGoldProvider()
            test_examples = examples[:3]

            print(f"  Estimated cost: ${provider.estimate_cost(test_examples):.4f}")

            for i, ex in enumerate(test_examples):
                print(f"\n  Example {i + 1}: '{ex.target.lemma}' in context")
                output = provider.label_one(ex)
                print(f"    Chosen: {output.chosen_synset_id}")
                print(f"    Confidence: {output.confidence}")
                print(f"    Flags: {output.flags}")
                print(
                    f"    Tokens: {output.usage.input_tokens} in, {output.usage.output_tokens} out"
                )
                print(f"    Cost: ${output.usage.cost_usd:.4f}")

            print("\n✅ OpenAI provider works!")
            results["openai"] = True

        except Exception as e:
            print(f"❌ OpenAI Error: {e}")
            import traceback

            traceback.print_exc()

    # Test Anthropic
    if anthropic_key:
        print("\n🟠 Testing Anthropic provider...")
        try:
            from eng_words.wsd_gold.providers import AnthropicGoldProvider

            provider = AnthropicGoldProvider()
            test_examples = examples[:3]

            print(f"  Estimated cost: ${provider.estimate_cost(test_examples):.4f}")

            for i, ex in enumerate(test_examples):
                print(f"\n  Example {i + 1}: '{ex.target.lemma}' in context")
                output = provider.label_one(ex)
                print(f"    Chosen: {output.chosen_synset_id}")
                print(f"    Confidence: {output.confidence}")
                print(f"    Flags: {output.flags}")
                print(
                    f"    Tokens: {output.usage.input_tokens} in, {output.usage.output_tokens} out"
                )
                print(f"    Cost: ${output.usage.cost_usd:.4f}")

            print("\n✅ Anthropic provider works!")
            results["anthropic"] = True

        except Exception as e:
            print(f"❌ Anthropic Error: {e}")
            import traceback

            traceback.print_exc()

    # Test Gemini
    if google_key:
        print("\n🟢 Testing Gemini provider...")
        try:
            from eng_words.wsd_gold.providers import GeminiGoldProvider

            provider = GeminiGoldProvider()
            test_examples = examples[:3]

            print(f"  Estimated cost: ${provider.estimate_cost(test_examples):.4f}")

            all_valid = True
            for i, ex in enumerate(test_examples):
                print(f"\n  Example {i + 1}: '{ex.target.lemma}' in context")
                output = provider.label_one(ex)
                print(f"    Chosen: {output.chosen_synset_id}")
                print(f"    Confidence: {output.confidence}")
                print(f"    Flags: {output.flags}")
                print(
                    f"    Tokens: ~{output.usage.input_tokens} in, ~{output.usage.output_tokens} out"
                )
                print(f"    Cost: ${output.usage.cost_usd:.4f}")
                # Check for errors in raw_text
                if "Quota exceeded" in output.raw_text or "Error:" in output.raw_text:
                    print(f"    ⚠️ {output.raw_text[:100]}")
                    all_valid = False
                elif not output.chosen_synset_id:
                    all_valid = False

            if all_valid:
                print("\n✅ Gemini provider works!")
                results["gemini"] = True
            else:
                print("\n⚠️ Gemini returned invalid results (quota exceeded or errors)")
                print("   Check your Google API quota at: https://ai.dev/rate-limit")

        except ImportError as e:
            print(f"❌ Gemini ImportError: {e}")
            print("  Install with: pip install google-generativeai")
        except Exception as e:
            print(f"❌ Gemini Error: {e}")
            import traceback

            traceback.print_exc()

    return results


def save_sample_for_inspection(examples: list[GoldExample], output_path: Path) -> None:
    """Save a sample of examples for manual inspection."""
    print(f"\n💾 Saving sample to {output_path}")

    data = []
    for ex in examples[:10]:
        data.append(
            {
                "example_id": ex.example_id,
                "context": ex.context_window,
                "target_lemma": ex.target.lemma,
                "target_pos": ex.target.pos,
                "candidates": [{"synset_id": c.synset_id, "gloss": c.gloss} for c in ex.candidates],
                "metadata": ex.metadata.to_dict(),
            }
        )

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"✅ Saved {len(data)} examples")


def main():
    parser = argparse.ArgumentParser(description="Test WSD Gold pipeline with real data")
    parser.add_argument("--with-llm", action="store_true", help="Make real LLM API calls")
    args = parser.parse_args()

    print("🧪 WSD Gold Dataset Pipeline - Real Data Test")
    print("=" * 60)

    # Load data
    tokens_df, sentences_df, book_name = load_real_data()

    # Test Stage 1
    examples = test_stage1_collect(tokens_df, sentences_df, book_name)

    if not examples:
        print("\n❌ Stage 1 failed, cannot continue")
        sys.exit(1)

    # Test Stage 2
    sampled = test_stage2_sample(examples)

    # Test Stage 3
    provider_results = test_stage3_providers(sampled or examples[:10], args.with_llm)

    # Save sample
    output_dir = Path("data/processed")
    output_dir.mkdir(exist_ok=True)
    save_sample_for_inspection(sampled or examples, output_dir / "gold_dataset_sample.json")

    # Summary
    print("\n" + "=" * 60)
    print("📊 SUMMARY")
    print("=" * 60)

    print("\n✅ Stage 1 (collect): OK")
    print("✅ Stage 2 (sample): OK")

    if args.with_llm:
        print("\n🤖 LLM Providers:")
        for provider, ok in provider_results.items():
            status = "✅" if ok else "❌"
            print(f"  {status} {provider}")

        all_ok = all(provider_results.values())
        if not all_ok:
            print("\n⚠️ Some providers failed!")
            print("📖 See docs/LLM_API_KEYS_SETUP.md for setup instructions")
            sys.exit(1)
    else:
        print("\n⏭️ Stage 3 (providers): skipped (use --with-llm)")

    print("\n" + "=" * 60)
    print("✅ All tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
