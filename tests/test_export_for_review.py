from pathlib import Path

import pandas as pd

from eng_words.constants import (
    BOOK_FREQ,
    DEFINITION,
    GLOBAL_ZIPF,
    LEMMA,
    SCORE,
    SENSE_FREQ,
    STOPWORD_COUNT,
    SUPERSENSE,
    SYNSET_ID,
    VERB_COUNT,
)
from scripts import export_for_review as review_script


def test_export_for_review_merges_scores_and_status(tmp_path: Path) -> None:
    lemma_stats = pd.DataFrame(
        {
            LEMMA: ["and", "reply", "assault"],
            BOOK_FREQ: [16049, 295, 2],
            GLOBAL_ZIPF: [7.2, 4.5, 3.1],
            VERB_COUNT: [0, 5, 0],
            STOPWORD_COUNT: [16049, 0, 0],
        }
    )
    raw_path = tmp_path / "book_lemma_stats_full.parquet"
    lemma_stats.to_parquet(raw_path, index=False)

    candidate_stats = pd.DataFrame(
        {
            LEMMA: ["reply"],
            BOOK_FREQ: [295],
            GLOBAL_ZIPF: [4.5],
            SCORE: [0.21],
        }
    )
    filtered_path = tmp_path / "book_lemma_stats.parquet"
    candidate_stats.to_parquet(filtered_path, index=False)

    status_path = tmp_path / "review_prev.csv"
    status_path.write_text(
        "item,status,tags,book_freq,global_zipf,score\n" "reply,learning,B2,\n" "assault,ignore,,\n"
    )

    output_csv = tmp_path / "review_new.csv"
    review_script.export_for_review(
        lemma_stats_path=raw_path,
        score_source_path=filtered_path,
        status_source_path=status_path,
        output_path=output_csv,
    )

    result = pd.read_csv(output_csv)
    reply_row = result[result["item"] == "reply"].iloc[0]
    assert reply_row["score"] == 0.21
    assert reply_row["status"] == "learning"
    assault_row = result[result["item"] == "assault"].iloc[0]
    assert assault_row["status"] == "ignore"
    assert assault_row["score"] == 0.0


def test_export_for_review_without_score_source(tmp_path: Path) -> None:
    lemma_stats = pd.DataFrame(
        {
            LEMMA: ["word"],
            BOOK_FREQ: [10],
            GLOBAL_ZIPF: [4.0],
            VERB_COUNT: [2],
            STOPWORD_COUNT: [0],
        }
    )
    raw_path = tmp_path / "novel_lemma_stats_full.parquet"
    lemma_stats.to_parquet(raw_path, index=False)

    output_csv = tmp_path / "review_no_score.csv"
    review_script.export_for_review(
        lemma_stats_path=raw_path,
        output_path=output_csv,
    )

    result = pd.read_csv(output_csv)
    assert result.loc[0, "score"] == 0.0


def test_auto_ignore_ing_variants(tmp_path: Path) -> None:
    lemma_stats = pd.DataFrame(
        {
            LEMMA: ["answer", "answering", "ceiling"],
            BOOK_FREQ: [120, 4, 3],
            GLOBAL_ZIPF: [4.6, 3.2, 3.0],
            VERB_COUNT: [120, 0, 0],
            STOPWORD_COUNT: [0, 0, 0],
        }
    )
    raw_path = tmp_path / "novel_lemma_stats_full.parquet"
    lemma_stats.to_parquet(raw_path, index=False)

    filtered_path = tmp_path / "novel_lemma_stats.parquet"
    pd.DataFrame({LEMMA: [], SCORE: []}).to_parquet(filtered_path, index=False)

    output_csv = tmp_path / "review_auto_tag.csv"
    review_script.export_for_review(
        lemma_stats_path=raw_path,
        output_path=output_csv,
    )

    result = pd.read_csv(output_csv)
    answering = result[result["item"] == "answering"].iloc[0]
    assert answering["status"] == "ignore"
    assert "auto_ing_variant" in answering["tags"]

    ceiling = result[result["item"] == "ceiling"].iloc[0]
    # Status can be empty string or NaN for items without auto-tags
    assert pd.isna(ceiling["status"]) or ceiling["status"] == ""


def test_auto_tag_stopwords(tmp_path: Path) -> None:
    lemma_stats = pd.DataFrame(
        {
            LEMMA: ["the", "run"],
            BOOK_FREQ: [5000, 20],
            GLOBAL_ZIPF: [7.8, 4.2],
            VERB_COUNT: [0, 20],
            STOPWORD_COUNT: [5000, 0],
        }
    )
    raw_path = tmp_path / "book_lemma_stats_full.parquet"
    lemma_stats.to_parquet(raw_path, index=False)

    filtered_path = tmp_path / "book_lemma_stats.parquet"
    pd.DataFrame({LEMMA: ["run"], SCORE: [0.5]}).to_parquet(filtered_path, index=False)

    output_csv = tmp_path / "review_stop.csv"
    review_script.export_for_review(
        lemma_stats_path=raw_path,
        score_source_path=filtered_path,
        output_path=output_csv,
    )

    result = pd.read_csv(output_csv)
    the_row = result[result["item"] == "the"].iloc[0]
    assert "auto_stopword" in the_row["tags"]
    # Status can be empty string or NaN for items with only auto_stopword tag
    assert pd.isna(the_row["status"]) or the_row["status"] == ""


def test_export_for_review_with_supersenses(tmp_path: Path) -> None:
    """Test export with supersense statistics."""
    supersense_stats = pd.DataFrame(
        {
            LEMMA: ["run", "run", "bank", "bank"],
            SUPERSENSE: ["verb.motion", "verb.social", "noun.group", "noun.object"],
            SENSE_FREQ: [17, 10, 15, 5],
            BOOK_FREQ: [27, 27, 20, 20],
            "sense_ratio": [0.63, 0.37, 0.75, 0.25],
            "sense_count": [2, 2, 2, 2],
        }
    )
    stats_path = tmp_path / "book_supersense_stats.parquet"
    supersense_stats.to_parquet(stats_path, index=False)

    output_csv = tmp_path / "review_supersenses.csv"
    review_script.export_for_review_with_supersenses(
        supersense_stats_path=stats_path,
        output_path=output_csv,
    )

    result = pd.read_csv(output_csv)
    assert len(result) == 4  # One row per (lemma, supersense)
    assert "item" in result.columns
    assert SUPERSENSE in result.columns
    assert SENSE_FREQ in result.columns
    assert DEFINITION in result.columns
    assert "example" in result.columns

    # Check that run has 2 rows
    run_rows = result[result["item"] == "run"]
    assert len(run_rows) == 2
    assert set(run_rows[SUPERSENSE]) == {"verb.motion", "verb.social"}


def test_export_for_review_with_supersenses_and_definitions(tmp_path: Path) -> None:
    """Test export with definitions from sense_tokens."""
    supersense_stats = pd.DataFrame(
        {
            LEMMA: ["run", "run"],
            SUPERSENSE: ["verb.motion", "verb.social"],
            SENSE_FREQ: [17, 10],
            BOOK_FREQ: [27, 27],
            "sense_ratio": [0.63, 0.37],
            "sense_count": [2, 2],
        }
    )
    stats_path = tmp_path / "book_supersense_stats.parquet"
    supersense_stats.to_parquet(stats_path, index=False)

    # Create sense_tokens with definitions
    # Note: 'pos' column is optional but needed for function words export
    sense_tokens = pd.DataFrame(
        {
            LEMMA: ["run", "run", "run"],
            SUPERSENSE: ["verb.motion", "verb.motion", "verb.social"],
            SYNSET_ID: ["run.v.01", "run.v.01", "run.v.02"],
            DEFINITION: [
                "move fast by using legs",
                "move fast by using legs",
                "be in charge of",
            ],
            "pos": ["VERB", "VERB", "VERB"],  # Add POS for function words check
        }
    )
    tokens_path = tmp_path / "book_sense_tokens.parquet"
    sense_tokens.to_parquet(tokens_path, index=False)

    output_csv = tmp_path / "review_supersenses_defs.csv"
    review_script.export_for_review_with_supersenses(
        supersense_stats_path=stats_path,
        sense_tokens_path=tokens_path,
        output_path=output_csv,
    )

    result = pd.read_csv(output_csv)
    assert len(result) == 2

    # Check definitions are included
    run_motion = result[(result["item"] == "run") & (result[SUPERSENSE] == "verb.motion")].iloc[0]
    assert "move fast" in run_motion[DEFINITION].lower()

    run_social = result[(result["item"] == "run") & (result[SUPERSENSE] == "verb.social")].iloc[0]
    assert "in charge" in run_social[DEFINITION].lower()


def test_export_for_review_with_supersenses_sorted_by_sense_freq(tmp_path: Path) -> None:
    """Test that sorting works correctly."""
    supersense_stats = pd.DataFrame(
        {
            LEMMA: ["run", "run", "bank"],
            SUPERSENSE: ["verb.motion", "verb.social", "noun.group"],
            SENSE_FREQ: [10, 17, 15],  # verb.social has highest freq
            BOOK_FREQ: [27, 27, 20],
            "sense_ratio": [0.37, 0.63, 0.75],
            "sense_count": [2, 2, 1],
        }
    )
    stats_path = tmp_path / "book_supersense_stats.parquet"
    supersense_stats.to_parquet(stats_path, index=False)

    output_csv = tmp_path / "review_sorted.csv"
    review_script.export_for_review_with_supersenses(
        supersense_stats_path=stats_path,
        output_path=output_csv,
        sort_by=[SENSE_FREQ],
    )

    result = pd.read_csv(output_csv)
    # Should be sorted by sense_freq descending
    assert result.iloc[0][SENSE_FREQ] == 17  # verb.social
    assert result.iloc[1][SENSE_FREQ] == 15  # noun.group
    assert result.iloc[2][SENSE_FREQ] == 10  # verb.motion
