#!/usr/bin/env python3
"""Analyze WSD results to find issues and insights.

This script runs WSD on 5 chapters and performs detailed analysis:
1. Supersense distribution
2. Confidence score analysis
3. Unknown words investigation
4. Polysemy patterns
5. Potential issues and errors
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from eng_words.pipeline import process_book_stage1  # noqa: E402


def run_wsd_pipeline():
    """Run WSD pipeline on 5 chapters."""
    print("=" * 70)
    print("Running WSD Pipeline on 5 chapters...")
    print("=" * 70)

    book_path = project_root / "data/raw/american_tragedy_chapters_1-5.epub"
    output_dir = project_root / "data/processed/wsd_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    results = process_book_stage1(
        book_path=book_path,
        book_name="wsd_analysis",
        output_dir=output_dir,
        enable_wsd=True,
        wsd_checkpoint_interval=2000,
        detect_phrasals=False,
    )

    return results, output_dir


def analyze_basic_stats(sense_tokens_df, supersense_stats_df):
    """Analyze basic statistics."""
    print("\n" + "=" * 70)
    print("BASIC STATISTICS")
    print("=" * 70)

    total_tokens = len(sense_tokens_df)
    content_tokens = len(sense_tokens_df[sense_tokens_df["supersense"] != "unknown"])
    unknown_tokens = len(sense_tokens_df[sense_tokens_df["supersense"] == "unknown"])

    print("\n📊 Token counts:")
    print(f"  • Total tokens with WSD: {total_tokens:,}")
    print(
        f"  • Content tokens (known): {content_tokens:,} ({100*content_tokens/total_tokens:.1f}%)"
    )
    print(f"  • Unknown tokens: {unknown_tokens:,} ({100*unknown_tokens/total_tokens:.1f}%)")

    print("\n📊 Unique items:")
    print(f"  • Unique lemmas: {sense_tokens_df['lemma'].nunique():,}")
    print(f"  • Unique supersenses: {sense_tokens_df['supersense'].nunique()}")
    print(f"  • (lemma, supersense) pairs: {len(supersense_stats_df):,}")


def analyze_supersense_distribution(sense_tokens_df):
    """Analyze supersense distribution."""
    print("\n" + "=" * 70)
    print("SUPERSENSE DISTRIBUTION")
    print("=" * 70)

    # Count supersenses
    supersense_counts = sense_tokens_df["supersense"].value_counts()

    print("\n📊 Top 20 supersenses by frequency:")
    for i, (ss, count) in enumerate(supersense_counts.head(20).items(), 1):
        pct = 100 * count / len(sense_tokens_df)
        print(f"  {i:2d}. {ss:25s} {count:5d} ({pct:5.1f}%)")

    # Group by category
    print("\n📊 By category:")
    categories = {"noun": 0, "verb": 0, "adj": 0, "adv": 0, "unknown": 0}
    for ss, count in supersense_counts.items():
        if ss.startswith("noun."):
            categories["noun"] += count
        elif ss.startswith("verb."):
            categories["verb"] += count
        elif ss.startswith("adj."):
            categories["adj"] += count
        elif ss.startswith("adv."):
            categories["adv"] += count
        else:
            categories["unknown"] += count

    total = sum(categories.values())
    for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
        pct = 100 * count / total
        print(f"  {cat:10s} {count:6d} ({pct:5.1f}%)")


def analyze_confidence_scores(sense_tokens_df):
    """Analyze confidence score distribution."""
    print("\n" + "=" * 70)
    print("CONFIDENCE SCORE ANALYSIS")
    print("=" * 70)

    # Filter content words (have confidence)
    content_df = sense_tokens_df[sense_tokens_df["supersense"] != "unknown"].copy()

    if "sense_confidence" not in content_df.columns:
        print("⚠️  No sense_confidence column found")
        return

    confidence = content_df["sense_confidence"]

    print("\n📊 Confidence statistics:")
    print(f"  • Mean: {confidence.mean():.3f}")
    print(f"  • Median: {confidence.median():.3f}")
    print(f"  • Std: {confidence.std():.3f}")
    print(f"  • Min: {confidence.min():.3f}")
    print(f"  • Max: {confidence.max():.3f}")

    # Distribution buckets
    print("\n📊 Confidence distribution:")
    buckets = [(0, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5), (0.5, 0.6), (0.6, 1.0)]
    for low, high in buckets:
        count = len(confidence[(confidence >= low) & (confidence < high)])
        pct = 100 * count / len(confidence)
        print(f"  [{low:.1f}-{high:.1f}): {count:5d} ({pct:5.1f}%)")

    # Low confidence examples
    print("\n⚠️  Low confidence examples (< 0.25):")
    low_conf = content_df[content_df["sense_confidence"] < 0.25].copy()
    if not low_conf.empty:
        # Group by lemma and count
        low_conf_counts = low_conf.groupby(["lemma", "supersense"]).size().reset_index(name="count")
        low_conf_counts = low_conf_counts.sort_values("count", ascending=False)
        for _, row in low_conf_counts.head(10).iterrows():
            print(f"  • {row['lemma']:15s} → {row['supersense']:20s} (×{row['count']})")


def analyze_unknown_words(sense_tokens_df, tokens_df):
    """Analyze words marked as 'unknown'."""
    print("\n" + "=" * 70)
    print("UNKNOWN WORDS ANALYSIS")
    print("=" * 70)

    unknown_df = sense_tokens_df[sense_tokens_df["supersense"] == "unknown"].copy()

    if unknown_df.empty:
        print("✅ No unknown words found")
        return

    # Get POS distribution
    if "pos" in unknown_df.columns:
        print("\n📊 Unknown by POS:")
        pos_counts = unknown_df["pos"].value_counts()
        for pos, count in pos_counts.head(10).items():
            pct = 100 * count / len(unknown_df)
            print(f"  {pos:10s} {count:5d} ({pct:5.1f}%)")

    # Top unknown lemmas
    print("\n📊 Top 30 unknown lemmas by frequency:")
    unknown_lemmas = unknown_df["lemma"].value_counts()
    for i, (lemma, count) in enumerate(unknown_lemmas.head(30).items(), 1):
        print(f"  {i:2d}. {lemma:20s} ×{count}")

    # Check if unknown are function words or punctuation
    print("\n📊 Categories of unknown:")
    func_words = {
        "the",
        "a",
        "an",
        "and",
        "or",
        "but",
        "in",
        "on",
        "at",
        "to",
        "of",
        "for",
        "with",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "must",
        "shall",
        "can",
        "it",
        "he",
        "she",
        "they",
        "we",
        "you",
        "i",
        "that",
        "this",
        "these",
        "those",
        "which",
        "who",
        "whom",
        "whose",
        "what",
        "when",
        "where",
        "why",
        "how",
        "if",
        "then",
        "than",
        "so",
        "as",
        "not",
        "no",
        "yes",
    }
    punct = set(".,!?;:'\"-()[]{}…—–")

    func_count = 0
    punct_count = 0
    other_count = 0

    for lemma in unknown_df["lemma"]:
        if lemma.lower() in func_words:
            func_count += 1
        elif all(c in punct for c in lemma):
            punct_count += 1
        else:
            other_count += 1

    total = len(unknown_df)
    print(f"  Function words: {func_count:5d} ({100*func_count/total:.1f}%)")
    print(f"  Punctuation:    {punct_count:5d} ({100*punct_count/total:.1f}%)")
    print(f"  Other:          {other_count:5d} ({100*other_count/total:.1f}%)")


def analyze_polysemy(supersense_stats_df, sense_tokens_df):
    """Analyze polysemy patterns."""
    print("\n" + "=" * 70)
    print("POLYSEMY ANALYSIS")
    print("=" * 70)

    # Filter content words
    content_stats = supersense_stats_df[supersense_stats_df["supersense"] != "unknown"].copy()

    # Count senses per lemma
    senses_per_lemma = (
        content_stats.groupby("lemma")["supersense"].nunique().reset_index(name="sense_count")
    )

    print("\n📊 Polysemy distribution:")
    sense_dist = senses_per_lemma["sense_count"].value_counts().sort_index()
    for n_senses, count in sense_dist.items():
        pct = 100 * count / len(senses_per_lemma)
        print(f"  {n_senses} sense(s): {count:4d} lemmas ({pct:5.1f}%)")

    # Highly polysemous words
    print("\n📊 Most polysemous words (5+ senses):")
    polysemous = senses_per_lemma[senses_per_lemma["sense_count"] >= 5].sort_values(
        "sense_count", ascending=False
    )
    for _, row in polysemous.iterrows():
        lemma = row["lemma"]
        n = row["sense_count"]
        # Get the supersenses for this lemma
        lemma_senses = content_stats[content_stats["lemma"] == lemma][["supersense", "sense_freq"]]
        lemma_senses = lemma_senses.sort_values("sense_freq", ascending=False)
        top_senses = lemma_senses.head(3)["supersense"].tolist()
        print(f"  {lemma:15s} {n} senses: {', '.join(top_senses)}")


def analyze_potential_issues(sense_tokens_df, supersense_stats_df):
    """Look for potential issues and errors."""
    print("\n" + "=" * 70)
    print("POTENTIAL ISSUES")
    print("=" * 70)

    issues_found = []

    # 1. Check for POS/supersense mismatches
    print("\n🔍 Checking POS/supersense consistency...")
    content_df = sense_tokens_df[sense_tokens_df["supersense"] != "unknown"].copy()

    if "pos" in content_df.columns:
        # Nouns with verb supersense
        noun_as_verb = content_df[
            (content_df["pos"] == "NOUN") & (content_df["supersense"].str.startswith("verb."))
        ]
        if not noun_as_verb.empty:
            count = len(noun_as_verb)
            examples = noun_as_verb.groupby("lemma").size().sort_values(ascending=False).head(5)
            issues_found.append(f"⚠️  {count} tokens: NOUN POS but verb. supersense")
            for lemma, cnt in examples.items():
                print(f"    • {lemma}: {cnt}×")

        # Verbs with noun supersense
        verb_as_noun = content_df[
            (content_df["pos"] == "VERB") & (content_df["supersense"].str.startswith("noun."))
        ]
        if not verb_as_noun.empty:
            count = len(verb_as_noun)
            examples = verb_as_noun.groupby("lemma").size().sort_values(ascending=False).head(5)
            issues_found.append(f"⚠️  {count} tokens: VERB POS but noun. supersense")
            for lemma, cnt in examples.items():
                print(f"    • {lemma}: {cnt}×")

    # 2. Check for very high confidence on rare senses
    print("\n🔍 Checking confidence distribution by frequency...")
    if "sense_confidence" in content_df.columns and "sense_freq" in supersense_stats_df.columns:
        # Join to get sense_freq
        merged = content_df.merge(
            supersense_stats_df[["lemma", "supersense", "sense_freq"]],
            on=["lemma", "supersense"],
            how="left",
        )

        # High confidence on rare senses (freq < 3)
        rare_high_conf = merged[(merged["sense_freq"] < 3) & (merged["sense_confidence"] > 0.6)]
        if not rare_high_conf.empty:
            issues_found.append(
                f"⚠️  {len(rare_high_conf)} tokens: rare sense (freq<3) but high confidence (>0.6)"
            )

    # 3. Check for missing definitions
    print("\n🔍 Checking for missing definitions...")
    if "definition" in content_df.columns:
        no_def = content_df[content_df["definition"].isna() | (content_df["definition"] == "")]
        if not no_def.empty:
            issues_found.append(f"⚠️  {len(no_def)} tokens without definition")

    # 4. Check synset_id format
    print("\n🔍 Checking synset_id format...")
    if "synset_id" in content_df.columns:
        # Valid format: word.pos.nn (apostrophes allowed in word names)
        invalid_synsets = content_df[
            ~content_df["synset_id"].str.match(r"^[\w\-\']+\.[nvasr]\.\d+$", na=True)
        ]
        if not invalid_synsets.empty:
            issues_found.append(f"⚠️  {len(invalid_synsets)} tokens with invalid synset_id format")
            print(f"    Examples: {invalid_synsets['synset_id'].head().tolist()}")

    # 5. Check for duplicate sense annotations
    print("\n🔍 Checking for consistency within sentences...")
    # Same lemma, same sentence should ideally have same sense
    grouped = (
        content_df.groupby(["sentence_id", "lemma"])["supersense"]
        .nunique()
        .reset_index(name="n_senses")
    )
    inconsistent = grouped[grouped["n_senses"] > 1]
    if not inconsistent.empty:
        issues_found.append(
            f"⚠️  {len(inconsistent)} (sentence, lemma) pairs with multiple senses in same sentence"
        )
        # This is actually expected and not an error!
        print("    (Note: This is expected - same word can have different senses in one sentence)")

    # Summary
    print("\n" + "-" * 40)
    print("ISSUE SUMMARY")
    print("-" * 40)
    if issues_found:
        for issue in issues_found:
            print(issue)
    else:
        print("✅ No major issues found!")


def analyze_examples(sense_tokens_df, sentences_df):
    """Show example sentences for interesting cases."""
    print("\n" + "=" * 70)
    print("EXAMPLE SENTENCES")
    print("=" * 70)

    # Get some interesting polysemous words
    interesting_words = ["run", "make", "get", "see", "go", "take", "look", "work"]

    for word in interesting_words:
        word_df = sense_tokens_df[
            (sense_tokens_df["lemma"] == word) & (sense_tokens_df["supersense"] != "unknown")
        ]

        if word_df.empty:
            continue

        print(f"\n📖 '{word.upper()}' examples:")

        # Get unique supersenses
        senses = word_df["supersense"].unique()[:3]  # Top 3 senses

        for sense in senses:
            sense_df = word_df[word_df["supersense"] == sense]
            if sense_df.empty:
                continue

            # Get one example sentence
            sent_id = sense_df.iloc[0]["sentence_id"]
            sent_row = sentences_df[sentences_df["sentence_id"] == sent_id]
            if sent_row.empty:
                continue

            sentence = sent_row.iloc[0]["sentence"]
            conf = sense_df.iloc[0].get("sense_confidence", 0)

            # Truncate long sentences
            if len(sentence) > 100:
                sentence = sentence[:100] + "..."

            print(f"  • {sense}: (conf={conf:.2f})")
            print(f'    "{sentence}"')


def main():
    """Main analysis function."""
    # Run pipeline
    results, output_dir = run_wsd_pipeline()

    # Get DataFrames
    tokens_df = results["tokens_df"]
    sense_tokens_df = results["sense_tokens_df"]
    supersense_stats_df = results["supersense_stats_df"]

    # Load sentences
    from eng_words.text_processing import (
        create_sentences_dataframe,
        reconstruct_sentences_from_tokens,
    )

    sentences = reconstruct_sentences_from_tokens(tokens_df)
    sentences_df = create_sentences_dataframe(sentences)

    # Run analyses
    analyze_basic_stats(sense_tokens_df, supersense_stats_df)
    analyze_supersense_distribution(sense_tokens_df)
    analyze_confidence_scores(sense_tokens_df)
    analyze_unknown_words(sense_tokens_df, tokens_df)
    analyze_polysemy(supersense_stats_df, sense_tokens_df)
    analyze_potential_issues(sense_tokens_df, supersense_stats_df)
    analyze_examples(sense_tokens_df, sentences_df)

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"\nResults saved to: {output_dir}")
    print("  • wsd_analysis_sense_tokens.parquet")
    print("  • wsd_analysis_supersense_stats.parquet")

    # Save analysis summary
    print("  • analysis_summary.txt")

    return results


if __name__ == "__main__":
    main()
