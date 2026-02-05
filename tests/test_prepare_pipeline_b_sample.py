"""Tests for prepare_pipeline_b_sample: sample from Stage 1 tokens + sentences.parquet."""

from pathlib import Path

import pandas as pd
import pytest

from eng_words.pipeline import process_book_stage1


def test_prepare_sample_uses_sentences_parquet_and_keeps_invariant(tmp_path: Path) -> None:
    """After sample, tokens_sample.sentence_id is subset of sentences_sample.sentence_id."""
    # 1. Stage 1 output into data/processed layout
    book_name = "sample_book"
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir(parents=True)
    text = "First sentence here. Second one there. Third for good measure."
    book_path = tmp_path / "book.txt"
    book_path.write_text(text)

    try:
        result = process_book_stage1(
            book_path,
            book_name=book_name,
            output_dir=processed_dir,
        )
    except OSError as exc:
        if "Can't find model" in str(exc):
            pytest.skip("spaCy model en_core_web_sm is not installed")
        raise

    assert (processed_dir / f"{book_name}_tokens.parquet").exists()
    assert (processed_dir / f"{book_name}_sentences.parquet").exists()

    # 2. Run prepare_pipeline_b_sample (import main and call with patched argv)
    import sys

    script_path = Path(__file__).resolve().parent.parent / "scripts" / "prepare_pipeline_b_sample.py"
    assert script_path.exists(), f"Script not found: {script_path}"

    # Run script via subprocess so --data-dir points to tmp_path
    import subprocess

    env = {**__import__("os").environ}
    r = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--data-dir",
            str(tmp_path),
            "--book",
            book_name,
            "--size",
            "2",
        ],
        cwd=str(Path(__file__).resolve().parent.parent),
        capture_output=True,
        text=True,
        env=env,
    )
    assert r.returncode == 0, f"Script failed: {r.stderr}"

    # 3. Check outputs and invariant
    experiment_dir = tmp_path / "experiment"
    tokens_sample_path = experiment_dir / "tokens_sample.parquet"
    sentences_sample_path = experiment_dir / "sentences_sample.parquet"
    assert tokens_sample_path.exists()
    assert sentences_sample_path.exists()

    tokens_sample = pd.read_parquet(tokens_sample_path)
    sentences_sample = pd.read_parquet(sentences_sample_path)
    assert "sentence_id" in tokens_sample.columns
    assert "sentence_id" in sentences_sample.columns and "text" in sentences_sample.columns

    token_sids = set(tokens_sample["sentence_id"].unique())
    sentence_sids = set(sentences_sample["sentence_id"].unique())
    assert token_sids.issubset(sentence_sids), (
        "After sample, tokens_sample.sentence_id must be subset of sentences_sample.sentence_id"
    )
