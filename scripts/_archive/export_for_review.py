#!/usr/bin/env python3
"""Export lemma/phrasal stats to CSV for manual review in Google Sheets.

Creates a CSV file with all candidates sorted by global_zipf and book_freq,
ready for manual marking in Google Sheets.
"""

import sys
from pathlib import Path
from typing import List

import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from eng_words.constants import (  # noqa: E402
    BOOK_FREQ,
    DEFINITION,
    EXAMPLE,
    GLOBAL_ZIPF,
    ITEM_TYPE,
    ITEM_TYPE_PHRASAL_VERB,
    ITEM_TYPE_WORD,
    LEMMA,
    PHRASAL,
    SCORE,
    SENSE_FREQ,
    STATUS,
    STATUS_IGNORE,
    STOPWORD_COUNT,
    SUPERSENSE,
    SYNSET_ID,
    TAGS,
    TEMPLATE_LEMMA_STATS,
    TEMPLATE_LEMMA_STATS_FULL,
    VERB_COUNT,
    get_included_function_words,
)

AUTO_TAG_ING_VARIANT = "auto_ing_variant"
AUTO_TAG_STOPWORD = "auto_stopword"
AUTO_TAG_FUNCTION_WORD = "auto_function_word"


def _get_function_words_for_export(sense_tokens_df: pd.DataFrame) -> pd.DataFrame:
    """Get function words (ADP/SCONJ) that should be included in export.

    Args:
        sense_tokens_df: DataFrame with tokens including POS column

    Returns:
        DataFrame with function words ready for export (same structure as review_df)
    """
    # Check if POS column exists (may not be present in test data)
    if "pos" not in sense_tokens_df.columns:
        return pd.DataFrame()

    # Filter ADP/SCONJ tokens
    func_tokens = sense_tokens_df[sense_tokens_df["pos"].isin(["ADP", "SCONJ"])]

    if func_tokens.empty:
        return pd.DataFrame()

    # Group by lemma and count frequency
    func_stats = (
        func_tokens.groupby(LEMMA)
        .agg(
            {
                "pos": "count",  # Count occurrences
            }
        )
        .rename(columns={"pos": SENSE_FREQ})
        .reset_index()
    )

    # Filter to only include words with ignore=False
    included_lemmas = get_included_function_words()
    func_stats = func_stats[func_stats[LEMMA].isin(included_lemmas)]

    if func_stats.empty:
        return pd.DataFrame()

    # Add required columns to match review_df structure
    func_stats[SUPERSENSE] = "function_word"  # Special supersense for function words
    func_stats[BOOK_FREQ] = func_stats[SENSE_FREQ]  # Same as sense_freq for function words
    func_stats[DEFINITION] = ""
    func_stats[EXAMPLE] = ""
    func_stats[STATUS] = ""
    func_stats[TAGS] = AUTO_TAG_FUNCTION_WORD
    func_stats[ITEM_TYPE] = ITEM_TYPE_WORD

    return func_stats


def _infer_score_source(lemma_stats_path: Path) -> Path | None:
    """Infer candidate stats path when not provided."""

    if lemma_stats_path.name.endswith(TEMPLATE_LEMMA_STATS_FULL):
        candidate = lemma_stats_path.with_name(
            lemma_stats_path.name.replace(
                TEMPLATE_LEMMA_STATS_FULL,
                TEMPLATE_LEMMA_STATS,
            )
        )
        if candidate.exists():
            return candidate
    return None


def _merge_status_columns(review_df: pd.DataFrame, status_source: Path) -> pd.DataFrame:
    """Merge status/tags columns from an existing CSV."""

    if not status_source.exists():
        raise FileNotFoundError(status_source)

    status_df = pd.read_csv(status_source)
    if "item" not in status_df.columns:
        raise ValueError("status source must contain 'item' column")

    available_cols = ["item"]
    if STATUS in status_df.columns:
        available_cols.append(STATUS)
    if TAGS in status_df.columns:
        available_cols.append(TAGS)

    status_subset = status_df[available_cols].copy()
    if STATUS not in status_subset.columns:
        status_subset[STATUS] = ""
    else:
        status_subset[STATUS] = status_subset[STATUS].fillna("")
    if TAGS not in status_subset.columns:
        status_subset[TAGS] = ""
    else:
        status_subset[TAGS] = status_subset[TAGS].fillna("")

    merged = review_df.merge(
        status_subset,
        on="item",
        how="left",
        suffixes=("", "_existing"),
    )

    for column in [STATUS, TAGS]:
        existing_col = f"{column}_existing"
        if existing_col in merged.columns:
            merged[column] = merged[existing_col].where(
                merged[existing_col].notna() & (merged[existing_col] != ""),
                merged[column],
            )
            merged = merged.drop(columns=[existing_col])

    merged[STATUS] = merged[STATUS].fillna("")
    merged[TAGS] = merged[TAGS].fillna("")
    return merged


def _generate_ing_candidates(word: str) -> list[str]:
    """Return possible base forms for an -ing word."""

    if not word.endswith("ing") or len(word) <= 4:
        return []

    stem = word[:-3]
    candidates = {stem}
    candidates.add(stem + "e")

    if len(stem) >= 2 and stem[-1] == stem[-2]:
        candidates.add(stem[:-1])

    if word.endswith("ying") and len(word) >= 5:
        candidates.add(word[:-4] + "ie")

    return [cand for cand in candidates if len(cand) >= 2]


def _append_tag(existing: str, new_tag: str) -> str:
    """Append tag string if not already present."""

    tokens = existing.split()
    if new_tag in tokens:
        return existing
    if not existing:
        return new_tag
    return f"{existing} {new_tag}"


def _auto_tag_ing_variants(review_df: pd.DataFrame, lemma_words: set[str]) -> pd.DataFrame:
    """Mark non-verb -ing forms as ignore with explanation tag."""

    if VERB_COUNT not in review_df.columns:
        return review_df

    verb_counts = review_df[VERB_COUNT].fillna(0).astype(int)
    mask = (
        (review_df[ITEM_TYPE] == ITEM_TYPE_WORD)
        & review_df["item"].str.endswith("ing")
        & verb_counts.eq(0)
        & review_df[STATUS].astype(str).eq("")
    )

    indices = review_df.index[mask]
    for idx in indices:
        word = review_df.at[idx, "item"]
        for candidate in _generate_ing_candidates(word):
            if candidate in lemma_words:
                review_df.at[idx, STATUS] = STATUS_IGNORE
                review_df.at[idx, TAGS] = _append_tag(review_df.at[idx, TAGS], AUTO_TAG_ING_VARIANT)
                break

    return review_df


def _auto_tag_stopwords(review_df: pd.DataFrame) -> pd.DataFrame:
    """Add tag for rows marked as stop words."""

    if STOPWORD_COUNT not in review_df.columns:
        return review_df

    stop_counts = review_df[STOPWORD_COUNT].fillna(0).astype(int)
    indices = review_df.index[stop_counts.gt(0)]

    for idx in indices:
        review_df.at[idx, TAGS] = _append_tag(review_df.at[idx, TAGS], AUTO_TAG_STOPWORD)

    return review_df


def _get_definitions_from_synsets(sense_tokens_df: pd.DataFrame) -> pd.DataFrame:
    """Extract definitions from sense_tokens DataFrame.

    Args:
        sense_tokens_df: DataFrame with synset_id and definition columns

    Returns:
        DataFrame with synset_id -> definition mapping
    """
    if SYNSET_ID not in sense_tokens_df.columns or DEFINITION not in sense_tokens_df.columns:
        return pd.DataFrame(columns=[SYNSET_ID, DEFINITION])

    # Get unique (synset_id, definition) pairs
    def_df = sense_tokens_df[[SYNSET_ID, DEFINITION]].dropna(subset=[SYNSET_ID])
    def_df = def_df[def_df[SYNSET_ID] != ""].drop_duplicates(subset=[SYNSET_ID])
    return def_df


def export_for_review_with_supersenses(
    supersense_stats_path: Path,
    sense_tokens_path: Path | None = None,
    output_path: Path | None = None,
    *,
    sort_by: List[str] | None = None,
    status_source_path: Path | None = None,
    exclude_unknown: bool = True,
) -> Path:
    """Export supersense stats to CSV for manual review, with one row per (lemma, supersense).

    Args:
        supersense_stats_path: Path to supersense_stats.parquet
        sense_tokens_path: Path to sense_tokens.parquet (for definitions and examples)
        output_path: Output CSV path
        sort_by: Columns to sort by (default: sense_freq, book_freq)
        status_source_path: CSV with existing status/tags to merge
        exclude_unknown: If True, exclude rows where supersense is 'unknown'

    Returns:
        Path to created CSV file
    """
    if sort_by is None:
        sort_by = [SENSE_FREQ, BOOK_FREQ]

    if not supersense_stats_path.exists():
        raise FileNotFoundError(supersense_stats_path)

    # Load supersense stats
    supersense_df = pd.read_parquet(supersense_stats_path)
    required = {LEMMA, SUPERSENSE, SENSE_FREQ, BOOK_FREQ}
    missing = required - set(supersense_df.columns)
    if missing:
        raise ValueError(f"supersense_stats missing required columns: {missing}")

    # Filter out unknown supersenses if requested
    if exclude_unknown:
        # Find lemmas that would be completely lost (only have 'unknown')
        lemmas_with_unknown = set(supersense_df[supersense_df[SUPERSENSE] == "unknown"][LEMMA])
        lemmas_with_known = set(supersense_df[supersense_df[SUPERSENSE] != "unknown"][LEMMA])
        only_unknown_lemmas = lemmas_with_unknown - lemmas_with_known

        before_count = len(supersense_df)
        supersense_df = supersense_df[supersense_df[SUPERSENSE] != "unknown"]
        filtered_count = before_count - len(supersense_df)

        print(f"Filtered out {filtered_count:,} 'unknown' supersense rows")
        print(f"  → {len(only_unknown_lemmas):,} lemmas completely lost (only had 'unknown')")

    # Load sense_tokens if available (used for definitions and function words)
    sense_tokens_df = None
    if sense_tokens_path and sense_tokens_path.exists():
        sense_tokens_df = pd.read_parquet(sense_tokens_path)

    # Check for potentially lost content words (in WordNet but only 'unknown')
    if exclude_unknown and only_unknown_lemmas and sense_tokens_df is not None:
        from nltk.corpus import wordnet as wn

        # Filter out non-content: punct, func words, proper nouns
        punct = set(",.-?!:;\"'()[]{}…—–")
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
            "he",
            "she",
            "it",
            "they",
            "we",
            "you",
            "i",
            "not",
            "from",
            "into",
            "because",
            "'s",
            "n't",
            "'re",
            "'m",
            "'ve",
            "'ll",
        }

        # Get dominant POS for each lemma
        lemma_pos = sense_tokens_df.groupby("lemma")["pos"].agg(
            lambda x: x.mode().iloc[0] if len(x) > 0 else None
        )
        content_pos = {"NOUN", "VERB", "ADJ", "ADV"}

        lost_content_in_wn = []
        for lemma in only_unknown_lemmas:
            if all(c in punct for c in lemma):
                continue
            if lemma.lower() in func_words:
                continue
            if lemma in lemma_pos and lemma_pos[lemma] not in content_pos:
                continue
            # Check WordNet
            if wn.synsets(lemma.lower()):
                lost_content_in_wn.append(lemma)

        if lost_content_in_wn:
            print(f"  ⚠️  {len(lost_content_in_wn)} content words in WordNet but lost!")
            if len(lost_content_in_wn) <= 10:
                print(f"     {', '.join(sorted(lost_content_in_wn)[:10])}")

    # Load definitions from sense_tokens if available
    definitions_df = pd.DataFrame(columns=[SYNSET_ID, DEFINITION])
    if sense_tokens_df is not None:
        definitions_df = _get_definitions_from_synsets(sense_tokens_df)

    # Prepare review DataFrame - one row per (lemma, supersense)
    review_df = supersense_df[[LEMMA, SUPERSENSE, SENSE_FREQ, BOOK_FREQ]].copy()

    # Add definition if available (need to match via synset_id from sense_tokens)
    if sense_tokens_df is not None and not definitions_df.empty:
        # Get most common synset_id for each (lemma, supersense) pair
        synset_mapping = (
            sense_tokens_df[sense_tokens_df[SUPERSENSE] != "unknown"]
            .groupby([LEMMA, SUPERSENSE])[SYNSET_ID]
            .apply(lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else None)
            .reset_index(name=SYNSET_ID)
        )
        review_df = review_df.merge(synset_mapping, on=[LEMMA, SUPERSENSE], how="left")
        review_df = review_df.merge(definitions_df, on=SYNSET_ID, how="left")
        review_df = review_df.drop(columns=[SYNSET_ID], errors="ignore")
    else:
        review_df[DEFINITION] = ""

    # Add example column (placeholder, can be filled later)
    review_df[EXAMPLE] = ""

    # Add status and tags columns
    review_df[STATUS] = ""
    review_df[TAGS] = ""
    review_df[ITEM_TYPE] = ITEM_TYPE_WORD

    # Add function words (ADP/SCONJ) that should be included
    if sense_tokens_df is not None:
        function_words_df = _get_function_words_for_export(sense_tokens_df)
        if not function_words_df.empty:
            review_df = pd.concat([review_df, function_words_df], ignore_index=True)
            print(f"  → Added {len(function_words_df)} function words (ADP/SCONJ)")

    # Rename lemma to item for consistency
    review_df = review_df.rename(columns={LEMMA: "item"})

    # Merge statuses from previous annotations
    if status_source_path:
        review_df = _merge_status_columns(review_df, status_source_path)

    # Sort
    sort_by = [col for col in sort_by if col in review_df.columns]
    ascending = [False] * len(sort_by)
    review_df = review_df.sort_values(by=sort_by, ascending=ascending, na_position="last")

    # Column order
    column_order = [
        "item",
        SUPERSENSE,
        SENSE_FREQ,
        DEFINITION,
        EXAMPLE,
        STATUS,
        ITEM_TYPE,
        TAGS,
        BOOK_FREQ,
    ]
    column_order = [col for col in column_order if col in review_df.columns]
    review_df = review_df[column_order]

    if output_path is None:
        output_path = supersense_stats_path.parent / "review_export_supersenses.csv"

    review_df.to_csv(output_path, index=False)

    print(f"✅ Exported {len(review_df)} (lemma, supersense) pairs to {output_path}")
    print(f"\nColumns: {', '.join(review_df.columns)}")
    print(f"\nSorted by: {', '.join(sort_by)} (descending)")

    return output_path


def export_for_review(
    lemma_stats_path: Path,
    phrasal_stats_path: Path | None = None,
    output_path: Path | None = None,
    *,
    sort_by: List[str] | None = None,
    score_source_path: Path | None = None,
    status_source_path: Path | None = None,
) -> Path:
    """Export stats to CSV for manual review."""

    if sort_by is None:
        sort_by = [GLOBAL_ZIPF, BOOK_FREQ]

    if not lemma_stats_path.exists():
        raise FileNotFoundError(lemma_stats_path)

    lemma_df = pd.read_parquet(lemma_stats_path)
    required = {LEMMA, BOOK_FREQ, GLOBAL_ZIPF}
    missing = required - set(lemma_df.columns)
    if missing:
        raise ValueError(f"lemma stats missing required columns: {missing}")

    if VERB_COUNT not in lemma_df.columns:
        lemma_df[VERB_COUNT] = 0
    if STOPWORD_COUNT not in lemma_df.columns:
        lemma_df[STOPWORD_COUNT] = 0

    lemma_df = lemma_df[[LEMMA, BOOK_FREQ, GLOBAL_ZIPF, VERB_COUNT, STOPWORD_COUNT]].copy()
    lemma_df[ITEM_TYPE] = ITEM_TYPE_WORD

    # Attach scores from filtered stats if provided/inferred
    score_attached = False
    if score_source_path is None:
        score_source_path = _infer_score_source(lemma_stats_path)

    if score_source_path and score_source_path.exists():
        score_df = pd.read_parquet(score_source_path)
        if {LEMMA, SCORE}.issubset(score_df.columns):
            lemma_df = lemma_df.merge(score_df[[LEMMA, SCORE]], on=LEMMA, how="left")
            lemma_df[SCORE] = lemma_df[SCORE].fillna(0.0)
            score_attached = True
        else:
            lemma_df[SCORE] = 0.0
    else:
        lemma_df[SCORE] = 0.0

    # Prepare lemma dataframe
    lemma_review = lemma_df[
        [LEMMA, BOOK_FREQ, GLOBAL_ZIPF, SCORE, ITEM_TYPE, VERB_COUNT, STOPWORD_COUNT]
    ].copy()
    lemma_review[STATUS] = ""
    lemma_review[TAGS] = ""
    lemma_review = lemma_review.rename(columns={LEMMA: "item"})

    # Load phrasal stats if provided
    phrasal_review = None
    if phrasal_stats_path and phrasal_stats_path.exists():
        phrasal_df = pd.read_parquet(phrasal_stats_path)
        phrasal_df[ITEM_TYPE] = ITEM_TYPE_PHRASAL_VERB
        phrasal_review = phrasal_df[[PHRASAL, BOOK_FREQ, SCORE, ITEM_TYPE]].copy()
        phrasal_review[GLOBAL_ZIPF] = None
        phrasal_review[VERB_COUNT] = 0
        phrasal_review[STOPWORD_COUNT] = 0
        phrasal_review[STATUS] = ""
        phrasal_review[TAGS] = ""
        phrasal_review = phrasal_review.rename(columns={PHRASAL: "item"})
        phrasal_review = phrasal_review[
            ["item", BOOK_FREQ, GLOBAL_ZIPF, SCORE, ITEM_TYPE, STATUS, TAGS]
        ]

    # Combine
    if phrasal_review is not None:
        common_cols = set(lemma_review.columns) & set(phrasal_review.columns)
        lemma_review = lemma_review[list(common_cols)]
        phrasal_review = phrasal_review[list(common_cols)]
        review_df = pd.concat([lemma_review, phrasal_review], ignore_index=True)
    else:
        review_df = lemma_review

    # Merge statuses from previous annotations
    if status_source_path:
        review_df = _merge_status_columns(review_df, status_source_path)

    review_df = _auto_tag_ing_variants(review_df, set(lemma_df[LEMMA]))
    review_df = _auto_tag_stopwords(review_df)

    for column in (VERB_COUNT, STOPWORD_COUNT):
        if column in review_df.columns:
            review_df = review_df.drop(columns=[column])

    # Sort
    sort_by = [col for col in sort_by if col in review_df.columns]
    ascending = [False] * len(sort_by)
    review_df = review_df.sort_values(by=sort_by, ascending=ascending, na_position="last")

    column_order = ["item", STATUS, ITEM_TYPE, TAGS, BOOK_FREQ, GLOBAL_ZIPF, SCORE]
    column_order = [col for col in column_order if col in review_df.columns]
    review_df = review_df[column_order]

    if output_path is None:
        output_path = lemma_stats_path.parent / "review_export.csv"

    review_df.to_csv(output_path, index=False)

    print(f"✅ Exported {len(review_df)} items to {output_path}")
    print(f"   - Lemmas: {len(lemma_review)}")
    if phrasal_review is not None:
        print(f"   - Phrasal verbs: {len(phrasal_review)}")
    print(f"\nColumns: {', '.join(review_df.columns)}")
    print(f"\nSorted by: {', '.join(sort_by)} (descending)")

    if not score_attached:
        print("\n⚠️  Score source missing or invalid; scores set to 0 for all lemmas.")

    return output_path


def main() -> None:
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Export lemma/phrasal stats to CSV for manual review"
    )
    parser.add_argument(
        "--lemma-stats",
        type=Path,
        required=False,
        help="Path to lemma_stats.parquet (required if --supersense-stats not provided)",
    )
    parser.add_argument(
        "--phrasal-stats",
        type=Path,
        help="Path to phrasal_verb_stats.parquet (optional)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output CSV path (default: review_export.csv in same dir as lemma_stats)",
    )
    parser.add_argument(
        "--sort-by",
        nargs="+",
        default=None,
        help=f"Columns to sort by (default: {GLOBAL_ZIPF} {BOOK_FREQ} for regular export, {SENSE_FREQ} {BOOK_FREQ} for supersense export)",
    )
    parser.add_argument(
        "--score-source",
        type=Path,
        help="Path to filtered lemma stats with SCORE (defaults to *_lemma_stats.parquet)",
    )
    parser.add_argument(
        "--status-source",
        type=Path,
        help="CSV with existing status/tags to merge (e.g., previous review export)",
    )
    parser.add_argument(
        "--supersense-stats",
        type=Path,
        help="Path to supersense_stats.parquet (enables supersense export mode)",
    )
    parser.add_argument(
        "--sense-tokens",
        type=Path,
        help="Path to sense_tokens.parquet (for definitions, used with --supersense-stats)",
    )
    parser.add_argument(
        "--include-unknown",
        action="store_true",
        help="Include rows with 'unknown' supersense (by default they are excluded)",
    )

    args = parser.parse_args()

    # Use supersense export if supersense_stats is provided
    if args.supersense_stats:
        # lemma_stats not needed for supersense mode
        # Use default sort_by if not specified (will be set to [SENSE_FREQ, BOOK_FREQ] in function)
        export_for_review_with_supersenses(
            supersense_stats_path=args.supersense_stats,
            sense_tokens_path=args.sense_tokens,
            output_path=args.output,
            sort_by=args.sort_by,  # None will use default [SENSE_FREQ, BOOK_FREQ]
            status_source_path=args.status_source,
            exclude_unknown=not args.include_unknown,  # By default, exclude unknown
        )
    else:
        # lemma_stats is required for regular export mode
        if not args.lemma_stats:
            parser.error("--lemma-stats is required when --supersense-stats is not provided")

        # Use default sort_by if not specified (will be set to [GLOBAL_ZIPF, BOOK_FREQ] in function)
        export_for_review(
            lemma_stats_path=args.lemma_stats,
            phrasal_stats_path=args.phrasal_stats,
            output_path=args.output,
            sort_by=args.sort_by or [GLOBAL_ZIPF, BOOK_FREQ],  # Use default if None
            score_source_path=args.score_source,
            status_source_path=args.status_source,
        )


if __name__ == "__main__":
    main()
