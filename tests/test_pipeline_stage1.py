import sys
from pathlib import Path

import pytest

from eng_words.pipeline import process_book, process_book_stage1, run_full_pipeline_cli
from tests.test_integration import _create_excerpt_from_epub


def test_process_book_stage1_creates_parquet(tmp_path: Path) -> None:
    excerpt_path = _create_excerpt_from_epub(tmp_path)
    output_dir = tmp_path / "outputs"

    try:
        result = process_book_stage1(
            excerpt_path,
            book_name="american_tragedy_excerpt",
            output_dir=output_dir,
        )
    except OSError as exc:
        if "Can't find model" in str(exc):
            pytest.skip("spaCy model en_core_web_sm is not installed")
        raise

    tokens_df = result["tokens_df"]
    tokens_path = result["tokens_path"]
    stats_df = result["lemma_stats_df"]
    stats_path = result["lemma_stats_path"]

    assert tokens_path.exists()
    assert len(tokens_df) > 0
    assert tokens_df["book"].iloc[0] == "american_tragedy_excerpt"
    assert stats_path.exists()
    assert len(stats_df) > 0
    assert "global_zipf" in stats_df.columns


def test_process_book_stage1_handles_large_txt(tmp_path: Path) -> None:
    text = " ".join([f"This is sentence number {i}." for i in range(80000)])
    book_path = tmp_path / "large.txt"
    book_path.write_text(text)
    output_dir = tmp_path / "large_outputs"

    try:
        result = process_book_stage1(
            book_path,
            book_name="large_book",
            output_dir=output_dir,
        )
    except OSError as exc:
        if "Can't find model" in str(exc):
            pytest.skip("spaCy model en_core_web_sm is not installed")
        raise

    tokens_df = result["tokens_df"]
    tokens_path = result["tokens_path"]
    stats_df = result["lemma_stats_df"]
    stats_path = result["lemma_stats_path"]

    assert tokens_path.exists()
    assert stats_path.exists()
    assert len(tokens_df) > 0
    assert len(stats_df) > 0


def test_process_book_stage1_detects_phrasals(tmp_path: Path) -> None:
    text = "She gave up quickly. Turn the lights on now."
    book_path = tmp_path / "phrasal.txt"
    book_path.write_text(text)
    output_dir = tmp_path / "phrasal_outputs"

    try:
        result = process_book_stage1(
            book_path,
            book_name="phrasal_book",
            output_dir=output_dir,
            detect_phrasals=True,
        )
    except OSError as exc:
        if "Can't find model" in str(exc):
            pytest.skip("spaCy model en_core_web_sm is not installed")
        raise

    phrasal_df = result["phrasal_df"]
    phrasal_path = result["phrasal_path"]
    phrasal_stats_df = result["phrasal_stats_df"]
    phrasal_stats_path = result["phrasal_stats_path"]

    assert phrasal_path and phrasal_path.exists()
    assert not phrasal_df.empty
    assert "give up" in set(phrasal_df["phrasal"])
    assert "turn on" in set(phrasal_df["phrasal"])
    assert phrasal_stats_path and phrasal_stats_path.exists()
    assert "phrasal" in phrasal_stats_df.columns


def test_process_book_full_pipeline(tmp_path: Path) -> None:
    text = "She gave up quickly. Turn the lights on now."
    book_path = tmp_path / "full.txt"
    book_path.write_text(text)
    output_dir = tmp_path / "full_outputs"

    try:
        outputs = process_book(
            book_path=book_path,
            book_name="full_book",
            output_dir=output_dir,
            detect_phrasals=True,
            top_n=10,
        )
    except OSError as exc:
        if "Can't find model" in str(exc):
            pytest.skip("spaCy model en_core_web_sm is not installed")
        raise

    for key, path in outputs.items():
        if path:
            assert Path(path).exists(), f"{key} file missing"


def test_run_full_pipeline_cli(tmp_path: Path, monkeypatch, capsys) -> None:
    text = "Books open minds."
    book_path = tmp_path / "cli.txt"
    book_path.write_text(text)
    output_dir = tmp_path / "cli_outputs"

    argv = [
        "eng_words.pipeline",
        "--book-path",
        str(book_path),
        "--book-name",
        "cli_book",
        "--output-dir",
        str(output_dir),
        "--no-phrasals",
        "--top-n",
        "5",
    ]

    monkeypatch.setattr(sys, "argv", argv)

    try:
        run_full_pipeline_cli()
    except OSError as exc:
        if "Can't find model" in str(exc):
            pytest.skip("spaCy model en_core_web_sm is not installed")
        raise

    captured = capsys.readouterr()
    assert "Pipeline completed" in captured.out
    assert (output_dir / "cli_book_tokens.parquet").exists()
    assert (output_dir / "cli_book_lemma_stats.parquet").exists()
