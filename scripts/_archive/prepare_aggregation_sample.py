#!/usr/bin/env python3
"""
Prepare a sample of complex lemmas for testing LLM-based synset aggregation.

Usage:
    uv run python scripts/prepare_aggregation_sample.py --min-synsets 5 --n-lemmas 20
"""

import argparse
import json
import logging
from pathlib import Path

import pandas as pd
from nltk.corpus import wordnet as wn

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

# Default paths
SENSE_TOKENS_PATH = Path("data/processed/american_tragedy_wsd_sense_tokens.parquet")
OUTPUT_DIR = Path("data/synset_aggregation_test")


def get_synset_info(synset_id: str) -> dict | None:
    """Get definition and supersense for a synset."""
    try:
        synset = wn.synset(synset_id)
        return {
            "synset_id": synset_id,
            "definition": synset.definition(),
            "supersense": synset.lexname(),
            "pos": synset.pos(),
        }
    except Exception:
        return None


def prepare_sample(
    sense_tokens_path: Path,
    output_dir: Path,
    min_synsets: int = 5,
    n_lemmas: int = 20,
    min_freq: int = 2,
) -> list[dict]:
    """
    Prepare a sample of complex lemmas for aggregation testing.
    
    Args:
        sense_tokens_path: Path to WSD sense tokens parquet
        output_dir: Output directory
        min_synsets: Minimum number of synsets per lemma
        n_lemmas: Number of lemmas to select
        min_freq: Minimum frequency per synset
        
    Returns:
        List of lemma data dictionaries
    """
    logger.info(f"Loading sense tokens from {sense_tokens_path}")
    tokens_df = pd.read_parquet(sense_tokens_path)
    
    # Filter to valid synsets
    tokens_df = tokens_df[tokens_df["synset_id"].notna()]
    logger.info(f"  {len(tokens_df):,} tokens with synset_id")
    
    # Count synsets per (lemma, synset_id) pair
    synset_freq = (
        tokens_df.groupby(["lemma", "synset_id"])
        .size()
        .reset_index(name="freq")
    )
    
    # Filter by minimum frequency
    synset_freq = synset_freq[synset_freq["freq"] >= min_freq]
    logger.info(f"  {len(synset_freq):,} (lemma, synset) pairs with freq >= {min_freq}")
    
    # Count synsets per lemma
    lemma_synset_count = synset_freq.groupby("lemma").size().reset_index(name="n_synsets")
    
    # Filter to lemmas with enough synsets
    complex_lemmas = lemma_synset_count[lemma_synset_count["n_synsets"] >= min_synsets]
    complex_lemmas = complex_lemmas.sort_values("n_synsets", ascending=False)
    logger.info(f"  {len(complex_lemmas):,} lemmas with {min_synsets}+ synsets")
    
    # Select top n lemmas
    selected_lemmas = complex_lemmas.head(n_lemmas)["lemma"].tolist()
    logger.info(f"  Selected {len(selected_lemmas)} lemmas: {selected_lemmas[:5]}...")
    
    # Prepare output data
    sample_data = []
    
    for lemma in selected_lemmas:
        lemma_synsets = synset_freq[synset_freq["lemma"] == lemma].sort_values(
            "freq", ascending=False
        )
        
        synsets = []
        for _, row in lemma_synsets.iterrows():
            synset_info = get_synset_info(row["synset_id"])
            if synset_info:
                synset_info["freq"] = int(row["freq"])
                synsets.append(synset_info)
        
        if synsets:
            sample_data.append({
                "lemma": lemma,
                "n_synsets": len(synsets),
                "total_freq": sum(s["freq"] for s in synsets),
                "synsets": synsets,
            })
    
    # Save to JSONL
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"sample_{n_lemmas}.jsonl"
    
    with open(output_path, "w", encoding="utf-8") as f:
        for item in sample_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    logger.info(f"Saved {len(sample_data)} lemmas to {output_path}")
    
    # Print summary
    print("\n" + "=" * 70)
    print(f"SAMPLE SUMMARY: {len(sample_data)} lemmas")
    print("=" * 70)
    
    for item in sample_data:
        print(f"\n### {item['lemma']} ({item['n_synsets']} synsets, {item['total_freq']} tokens)")
        for i, s in enumerate(item["synsets"][:5]):
            print(f"  {i+1}. {s['synset_id']:25} ({s['freq']:3}x) | {s['definition'][:45]}...")
        if len(item["synsets"]) > 5:
            print(f"  ... and {len(item['synsets']) - 5} more synsets")
    
    return sample_data


def main():
    parser = argparse.ArgumentParser(description="Prepare aggregation sample")
    parser.add_argument(
        "--min-synsets",
        type=int,
        default=5,
        help="Minimum number of synsets per lemma (default: 5)",
    )
    parser.add_argument(
        "--n-lemmas",
        type=int,
        default=20,
        help="Number of lemmas to select (default: 20)",
    )
    parser.add_argument(
        "--min-freq",
        type=int,
        default=2,
        help="Minimum frequency per synset (default: 2)",
    )
    parser.add_argument(
        "--sense-tokens",
        type=Path,
        default=SENSE_TOKENS_PATH,
        help="Path to sense tokens parquet",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Output directory",
    )
    
    args = parser.parse_args()
    
    prepare_sample(
        sense_tokens_path=args.sense_tokens,
        output_dir=args.output_dir,
        min_synsets=args.min_synsets,
        n_lemmas=args.n_lemmas,
        min_freq=args.min_freq,
    )


if __name__ == "__main__":
    main()

