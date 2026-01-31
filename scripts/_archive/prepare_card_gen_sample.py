#!/usr/bin/env python3
"""Prepare sample for card generation model comparison.

This script selects 50 examples (25 easy, 25 hard) for testing
different LLM models on card generation quality.

Usage:
    uv run python scripts/prepare_card_gen_sample.py --n-easy 25 --n-hard 25
    uv run python scripts/prepare_card_gen_sample.py --book american_tragedy_wsd
"""

import argparse
import json
import random
from pathlib import Path

import pandas as pd
from nltk.corpus import wordnet as wn


def get_pos_from_supersense(supersense: str) -> str | None:
    """Extract WordNet POS from supersense."""
    if supersense.startswith("noun"):
        return "n"
    elif supersense.startswith("verb"):
        return "v"
    elif supersense.startswith("adj"):
        return "a"
    elif supersense.startswith("adv"):
        return "r"
    return None


def get_synset_count(lemma: str, supersense: str) -> int:
    """Get number of WordNet synsets for lemma+pos."""
    pos = get_pos_from_supersense(supersense)
    synsets = wn.synsets(lemma, pos=pos)
    return len(synsets)


def get_wn_definition(lemma: str, supersense: str) -> str:
    """Get first WordNet definition for lemma+pos."""
    pos = get_pos_from_supersense(supersense)
    synsets = wn.synsets(lemma, pos=pos)
    if synsets:
        return synsets[0].definition()
    return ""


def get_pos_tag(supersense: str) -> str:
    """Get human-readable POS from supersense."""
    if supersense.startswith("noun"):
        return "noun"
    elif supersense.startswith("verb"):
        return "verb"
    elif supersense.startswith("adj"):
        return "adj"
    elif supersense.startswith("adv"):
        return "adv"
    return "unknown"


def reconstruct_sentence(tokens_df: pd.DataFrame, sentence_id: int) -> str:
    """Reconstruct sentence text from tokens."""
    sentence_tokens = tokens_df[tokens_df["sentence_id"] == sentence_id].sort_values("position")
    parts = []
    for _, row in sentence_tokens.iterrows():
        parts.append(row["surface"])
        if row.get("whitespace", " "):
            parts.append(" ")
    return "".join(parts).strip()


def get_contexts_for_lemma_supersense(
    sense_tokens_df: pd.DataFrame,
    tokens_df: pd.DataFrame,
    lemma: str,
    supersense: str,
    max_contexts: int = 10,
) -> list[str]:
    """Get example sentences for a (lemma, supersense) pair."""
    # Find tokens with this lemma and supersense
    matching = sense_tokens_df[
        (sense_tokens_df["lemma"] == lemma) & (sense_tokens_df["supersense"] == supersense)
    ]

    # Get unique sentence IDs
    sentence_ids = matching["sentence_id"].unique()

    # Sample if too many
    if len(sentence_ids) > max_contexts:
        sentence_ids = random.sample(list(sentence_ids), max_contexts)

    # Reconstruct sentences
    contexts = []
    for sid in sentence_ids:
        sentence = reconstruct_sentence(tokens_df, sid)
        if sentence and len(sentence) > 20:  # Filter very short
            contexts.append(sentence)

    return contexts[:max_contexts]


def select_easy_examples(
    stats_df: pd.DataFrame,
    n: int = 25,
    min_freq: int = 5,
) -> pd.DataFrame:
    """Select easy examples: 1-2 synsets, common words."""
    # Add synset count
    stats_df = stats_df.copy()
    stats_df["wn_synset_count"] = stats_df.apply(
        lambda r: get_synset_count(r["lemma"], r["supersense"]), axis=1
    )

    # Filter: 1-2 synsets, min frequency
    easy = stats_df[(stats_df["wn_synset_count"] <= 2) & (stats_df["sense_freq"] >= min_freq)]

    # Balance by POS
    result = []
    for pos_prefix in ["noun", "verb", "adj", "adv"]:
        pos_examples = easy[easy["supersense"].str.startswith(pos_prefix)]
        if len(pos_examples) > 0:
            sample_n = min(n // 4 + 1, len(pos_examples))
            result.append(pos_examples.nlargest(sample_n, "sense_freq"))

    combined = pd.concat(result).drop_duplicates()
    return combined.head(n)


def select_hard_examples(
    stats_df: pd.DataFrame,
    n: int = 25,
    min_freq: int = 3,
) -> pd.DataFrame:
    """Select hard examples: 5+ synsets, polysemous words."""
    # Add synset count
    stats_df = stats_df.copy()
    stats_df["wn_synset_count"] = stats_df.apply(
        lambda r: get_synset_count(r["lemma"], r["supersense"]), axis=1
    )

    # Filter: 5+ synsets, min frequency
    hard = stats_df[(stats_df["wn_synset_count"] >= 5) & (stats_df["sense_freq"] >= min_freq)]

    # Balance by POS (verbs are typically harder)
    result = []
    pos_weights = {"verb": 10, "noun": 8, "adj": 4, "adv": 3}

    for pos_prefix, weight in pos_weights.items():
        pos_examples = hard[hard["supersense"].str.startswith(pos_prefix)]
        if len(pos_examples) > 0:
            sample_n = min(n * weight // 25 + 1, len(pos_examples))
            result.append(pos_examples.nlargest(sample_n, "wn_synset_count"))

    combined = pd.concat(result).drop_duplicates()
    return combined.head(n)


def prepare_sample(
    book_name: str = "american_tragedy_wsd",
    n_easy: int = 25,
    n_hard: int = 25,
    output_path: Path | None = None,
) -> list[dict]:
    """Prepare sample for model comparison."""
    # Load data
    data_dir = Path("data/processed")
    supersense_stats = pd.read_parquet(data_dir / f"{book_name}_supersense_stats.parquet")
    tokens = pd.read_parquet(data_dir / f"{book_name}_tokens.parquet")
    sense_tokens = pd.read_parquet(data_dir / f"{book_name}_sense_tokens.parquet")

    print(f"Loaded {len(supersense_stats):,} supersense stats")
    print(f"Loaded {len(tokens):,} tokens")
    print(f"Loaded {len(sense_tokens):,} sense tokens")

    # Filter to content words
    content_stats = supersense_stats[
        (supersense_stats["supersense"] != "unknown")
        & (supersense_stats["lemma"].str.len() > 1)
        & (supersense_stats["lemma"].str.isalpha())
    ].copy()

    print(f"\nContent words: {len(content_stats):,}")

    # Select examples
    print(f"\nSelecting {n_easy} easy examples...")
    easy_examples = select_easy_examples(content_stats, n=n_easy)
    print(f"  Found {len(easy_examples)} easy examples")

    print(f"\nSelecting {n_hard} hard examples...")
    hard_examples = select_hard_examples(content_stats, n=n_hard)
    print(f"  Found {len(hard_examples)} hard examples")

    # Combine and prepare output
    samples = []

    for _, row in easy_examples.iterrows():
        contexts = get_contexts_for_lemma_supersense(
            sense_tokens, tokens, row["lemma"], row["supersense"]
        )
        if len(contexts) >= 3:  # Need at least 3 contexts
            samples.append(
                {
                    "id": f"easy_{len([s for s in samples if s['difficulty'] == 'easy']) + 1}",
                    "lemma": row["lemma"],
                    "pos": get_pos_tag(row["supersense"]),
                    "supersense": row["supersense"],
                    "wn_definition": get_wn_definition(row["lemma"], row["supersense"]),
                    "wn_synset_count": int(row["wn_synset_count"]),
                    "book_freq": int(row["sense_freq"]),
                    "difficulty": "easy",
                    "contexts": contexts,
                    "book_name": book_name,
                }
            )

    for _, row in hard_examples.iterrows():
        contexts = get_contexts_for_lemma_supersense(
            sense_tokens, tokens, row["lemma"], row["supersense"]
        )
        if len(contexts) >= 3:  # Need at least 3 contexts
            samples.append(
                {
                    "id": f"hard_{len([s for s in samples if s['difficulty'] == 'hard']) + 1}",
                    "lemma": row["lemma"],
                    "pos": get_pos_tag(row["supersense"]),
                    "supersense": row["supersense"],
                    "wn_definition": get_wn_definition(row["lemma"], row["supersense"]),
                    "wn_synset_count": int(row["wn_synset_count"]),
                    "book_freq": int(row["sense_freq"]),
                    "difficulty": "hard",
                    "contexts": contexts,
                    "book_name": book_name,
                }
            )

    print(f"\n{'=' * 60}")
    print(f"PREPARED {len(samples)} SAMPLES")
    print(f"{'=' * 60}")

    easy_count = len([s for s in samples if s["difficulty"] == "easy"])
    hard_count = len([s for s in samples if s["difficulty"] == "hard"])
    print(f"  Easy: {easy_count}")
    print(f"  Hard: {hard_count}")

    # POS distribution
    pos_dist = {}
    for s in samples:
        pos_dist[s["pos"]] = pos_dist.get(s["pos"], 0) + 1
    print(f"  POS: {pos_dist}")

    # Avg contexts
    avg_contexts = sum(len(s["contexts"]) for s in samples) / len(samples)
    print(f"  Avg contexts: {avg_contexts:.1f}")

    # Save
    if output_path is None:
        output_path = Path("data/card_gen_test/sample_50.jsonl")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"\nSaved to {output_path}")

    # Show examples
    print(f"\n{'=' * 60}")
    print("EXAMPLE EASY:")
    print(f"{'=' * 60}")
    easy_sample = [s for s in samples if s["difficulty"] == "easy"][0]
    print(f"  Lemma: {easy_sample['lemma']} ({easy_sample['pos']})")
    print(f"  Supersense: {easy_sample['supersense']}")
    print(f"  Definition: {easy_sample['wn_definition']}")
    print(f"  Synsets: {easy_sample['wn_synset_count']}")
    print(f"  Contexts ({len(easy_sample['contexts'])}):")
    for i, ctx in enumerate(easy_sample["contexts"][:3], 1):
        print(f"    {i}. {ctx[:100]}...")

    print(f"\n{'=' * 60}")
    print("EXAMPLE HARD:")
    print(f"{'=' * 60}")
    hard_sample = [s for s in samples if s["difficulty"] == "hard"][0]
    print(f"  Lemma: {hard_sample['lemma']} ({hard_sample['pos']})")
    print(f"  Supersense: {hard_sample['supersense']}")
    print(f"  Definition: {hard_sample['wn_definition']}")
    print(f"  Synsets: {hard_sample['wn_synset_count']}")
    print(f"  Contexts ({len(hard_sample['contexts'])}):")
    for i, ctx in enumerate(hard_sample["contexts"][:3], 1):
        print(f"    {i}. {ctx[:100]}...")

    return samples


def main():
    parser = argparse.ArgumentParser(description="Prepare sample for model comparison")
    parser.add_argument("--book", default="american_tragedy_wsd", help="Book name")
    parser.add_argument("--n-easy", type=int, default=25, help="Number of easy examples")
    parser.add_argument("--n-hard", type=int, default=25, help="Number of hard examples")
    parser.add_argument("--output", type=str, help="Output path")

    args = parser.parse_args()

    output_path = Path(args.output) if args.output else None

    prepare_sample(
        book_name=args.book,
        n_easy=args.n_easy,
        n_hard=args.n_hard,
        output_path=output_path,
    )


if __name__ == "__main__":
    main()
