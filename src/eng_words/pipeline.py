"""Stage 1 pipeline orchestration."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import spacy
from dotenv import load_dotenv

from eng_words.anki_export import export_to_anki_csv, prepare_anki_export
from eng_words.constants import (
    DEFAULT_MODEL_NAME,
    DIR_ANKI_EXPORTS,
    EXAMPLE,
    ITEM_TYPE,
    ITEM_TYPE_PHRASAL_VERB,
    LEMMA,
    MAX_ZIPF_DEFAULT,
    MIN_BOOK_FREQ_DEFAULT,
    MIN_ZIPF_DEFAULT,
    PHRASAL,
    SCORE,
    SENTENCE,
    SENTENCE_ID,
    STAGE1_MANIFEST,
    TEMPLATE_ANKI,
    TEMPLATE_LEMMA_STATS,
    TEMPLATE_LEMMA_STATS_FULL,
    TEMPLATE_MWE_CANDIDATES,
    TEMPLATE_PHRASAL_VERB_STATS,
    TEMPLATE_PHRASAL_VERBS,
    TEMPLATE_SENTENCES,
    TEMPLATE_TOKENS,
    TOP_N_DEFAULT,
)
from eng_words.mwe_candidates import build_mwe_candidates_from_phrasal
from eng_words.examples import get_examples_for_lemmas, get_examples_for_phrasal_verbs
from eng_words.filtering import (
    filter_by_frequency,
    filter_known_words,
    rank_candidates,
)
from eng_words.phrasal_verbs import (
    calculate_phrasal_verb_stats,
    detect_phrasal_verbs,
    filter_phrasal_verbs,
    initialize_phrasal_model,
    rank_phrasal_verbs,
)
from eng_words.statistics import (
    add_global_frequency,
    calculate_lemma_frequency,
    save_lemma_stats_to_parquet,
)
from eng_words.storage import load_known_words
from eng_words.text_io import load_book_text
from eng_words.text_processing import (
    create_sentences_dataframe,
    create_tokens_dataframe,
    initialize_spacy_model,
    reconstruct_sentences_from_tokens,
    save_tokens_to_parquet,
    tokenize_text_in_chunks,
)


def _get_pipeline_version() -> str | None:
    """Return git HEAD sha if repo is available, else None."""
    try:
        r = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=2,
            cwd=Path(__file__).resolve().parent.parent.parent,
        )
        if r.returncode == 0 and r.stdout:
            return r.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


def process_book_stage1(
    book_path: Path,
    book_name: str,
    output_dir: Path,
    *,
    model_name: str = DEFAULT_MODEL_NAME,
    known_words_path: Path | str | None = None,
    min_book_freq: int = MIN_BOOK_FREQ_DEFAULT,
    min_zipf: float = MIN_ZIPF_DEFAULT,
    max_zipf: float = MAX_ZIPF_DEFAULT,
    detect_phrasals: bool = False,
    phrasal_model_name: str | None = None,
    extract_mwe_candidates: bool = True,
    text: str | None = None,
) -> dict[str, pd.DataFrame | Path | None]:
    """Run Stage 1 pipeline and return dataframes + parquet paths.

    Args:
        book_path: Path to the book file (e.g., EPUB)
        book_name: Logical identifier for the book
        output_dir: Directory to store output files
        model_name: spaCy model name for tokenization
        known_words_path: Path to known words CSV or Google Sheets URL
        min_book_freq: Minimum book frequency for filtering
        min_zipf: Minimum global Zipf frequency
        max_zipf: Maximum global Zipf frequency
        detect_phrasals: Whether to detect phrasal verbs
        phrasal_model_name: spaCy model for phrasal detection
        extract_mwe_candidates: If True and detect_phrasals, write unified _mwe_candidates.parquet
        text: Pre-loaded text (optional, for testing)

    Returns:
        Dictionary with DataFrames and paths for all outputs
    """
    if not book_path or not book_path.exists():
        raise FileNotFoundError(book_path)
    if not book_name:
        raise ValueError("book_name must be provided")
    if not output_dir:
        raise ValueError("output_dir must be provided")

    if text is None:
        text = load_book_text(book_path)
    nlp = initialize_spacy_model(model_name=model_name)
    tokens = tokenize_text_in_chunks(text, nlp)
    tokens_df = create_tokens_dataframe(tokens, book_name)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tokens_path = output_dir / f"{book_name}{TEMPLATE_TOKENS}"
    save_tokens_to_parquet(tokens_df, tokens_path)

    # Stage 1 artifact: sentences.parquet (sentence_id, text) for Pipeline B
    sentences_list = reconstruct_sentences_from_tokens(tokens_df)
    sentences_df = create_sentences_dataframe(sentences_list)
    # Fail-fast: invariants for tokens/sentences consistency
    if not sentences_df[SENTENCE_ID].is_unique:
        raise ValueError(
            "Invariant violated: sentence_id must be unique in sentences. "
            "Re-run Stage 1 to rebuild outputs."
        )
    token_sids = set(tokens_df[SENTENCE_ID].unique())
    sentence_sids = set(sentences_df[SENTENCE_ID].unique())
    if not token_sids.issubset(sentence_sids):
        missing = token_sids - sentence_sids
        raise ValueError(
            f"Invariant violated: tokens.sentence_id must be subset of sentences.sentence_id. "
            f"Missing sentence_id in sentences: {sorted(missing)[:20]}{'...' if len(missing) > 20 else ''}. "
            "Re-run Stage 1 to rebuild outputs."
        )
    sentences_path = output_dir / f"{book_name}{TEMPLATE_SENTENCES}"
    sentences_export = sentences_df.rename(columns={SENTENCE: "text"})[
        [SENTENCE_ID, "text"]
    ]
    sentences_export.to_parquet(sentences_path, index=False)

    # Stage 1 manifest (required fields for QC/reproducibility; book_name/book_id for batch)
    manifest = {
        "schema_version": "1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "pipeline_version": _get_pipeline_version(),
        "spacy_model_name": model_name,
        "spacy_model": model_name,  # backward compat
        "spacy_version": spacy.__version__,
        "token_count": int(len(tokens_df)),
        "sentence_count": int(sentences_df[SENTENCE_ID].nunique()),
        "random_seed": None,
        "book_name": book_name,
        "book_id": book_name,  # alias for grouping/batch
    }
    # Optional checksums for deterministic runs (same book + params => same hashes)
    try:
        manifest["file_checksums"] = {
            "tokens_sha256": hashlib.sha256(tokens_path.read_bytes()).hexdigest(),
            "sentences_sha256": hashlib.sha256(sentences_path.read_bytes()).hexdigest(),
        }
    except OSError:
        pass
    manifest_path = output_dir / STAGE1_MANIFEST
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    full_lemma_stats = add_global_frequency(
        calculate_lemma_frequency(
            tokens_df,
            include_stopwords=True,
            include_proper_nouns=True,
        )
    )
    stats_full_path = output_dir / f"{book_name}{TEMPLATE_LEMMA_STATS_FULL}"
    save_lemma_stats_to_parquet(full_lemma_stats, stats_full_path)

    lemma_stats = add_global_frequency(calculate_lemma_frequency(tokens_df))
    lemma_stats = filter_by_frequency(
        lemma_stats,
        min_book_freq=min_book_freq,
        min_zipf=min_zipf,
        max_zipf=max_zipf,
    )
    lemma_stats = rank_candidates(lemma_stats)
    if known_words_path:
        known_df = load_known_words(known_words_path)
        if not known_df.empty:
            lemma_stats = filter_known_words(lemma_stats, known_df)
    stats_path = output_dir / f"{book_name}{TEMPLATE_LEMMA_STATS}"
    save_lemma_stats_to_parquet(lemma_stats, stats_path)

    phrasal_df = None
    phrasal_path = None
    phrasal_stats_df = None
    phrasal_stats_path = None
    mwe_candidates_path = None
    mwe_candidates_df = None
    if detect_phrasals:
        parser_nlp = initialize_phrasal_model(phrasal_model_name or model_name)
        phrasal_df = detect_phrasal_verbs(tokens_df, parser_nlp)
        phrasal_path = output_dir / f"{book_name}{TEMPLATE_PHRASAL_VERBS}"
        phrasal_df.to_parquet(phrasal_path, index=False)
        phrasal_stats_df = calculate_phrasal_verb_stats(phrasal_df)
        phrasal_known = None
        if known_words_path:
            known_df = load_known_words(known_words_path)
            if not known_df.empty:
                phrasal_known = known_df[known_df[ITEM_TYPE] == ITEM_TYPE_PHRASAL_VERB]
        phrasal_stats_df = filter_phrasal_verbs(phrasal_stats_df, phrasal_known)
        phrasal_stats_df = rank_phrasal_verbs(phrasal_stats_df)
        phrasal_stats_path = output_dir / f"{book_name}{TEMPLATE_PHRASAL_VERB_STATS}"
        phrasal_stats_df.to_parquet(phrasal_stats_path, index=False)

        if extract_mwe_candidates:
            mwe_candidates_df = build_mwe_candidates_from_phrasal(phrasal_df, phrasal_stats_df)
            mwe_candidates_path = output_dir / f"{book_name}{TEMPLATE_MWE_CANDIDATES}"
            mwe_candidates_df.to_parquet(mwe_candidates_path, index=False)

    return {
        "tokens_df": tokens_df,
        "tokens_path": tokens_path,
        "sentences_df": sentences_df,
        "sentences_path": sentences_path,
        "manifest_path": manifest_path,
        "lemma_stats_df": lemma_stats,
        "lemma_stats_path": stats_path,
        "phrasal_df": phrasal_df,
        "phrasal_path": phrasal_path,
        "phrasal_stats_df": phrasal_stats_df,
        "phrasal_stats_path": phrasal_stats_path,
        "mwe_candidates_df": mwe_candidates_df,
        "mwe_candidates_path": mwe_candidates_path,
    }


def process_book(
    book_path: Path,
    book_name: str,
    output_dir: Path,
    *,
    known_words_path: Path | str | None = None,
    model_name: str = DEFAULT_MODEL_NAME,
    phrasal_model_name: str | None = None,
    min_book_freq: int = MIN_BOOK_FREQ_DEFAULT,
    min_zipf: float = MIN_ZIPF_DEFAULT,
    max_zipf: float = MAX_ZIPF_DEFAULT,
    top_n: int = TOP_N_DEFAULT,
    detect_phrasals: bool = True,
    extract_mwe_candidates: bool = True,
) -> dict[str, Path | None]:
    """Full pipeline: runs tokenization, stats, examples, and Anki export.

    Args:
        book_path: Path to the book file (e.g., EPUB)
        book_name: Logical identifier for the book
        output_dir: Directory to store output files
        known_words_path: Path to known words CSV or Google Sheets URL
        model_name: spaCy model name for tokenization
        phrasal_model_name: spaCy model for phrasal detection
        min_book_freq: Minimum book frequency for filtering
        min_zipf: Minimum global Zipf frequency
        max_zipf: Maximum global Zipf frequency
        top_n: Number of top items for examples/Anki
        detect_phrasals: Whether to detect phrasal verbs
        extract_mwe_candidates: When phrasals detected, write _mwe_candidates.parquet

    Returns:
        Dictionary with paths to all output files
    """
    text = load_book_text(book_path)
    stage1_results = process_book_stage1(
        book_path=book_path,
        book_name=book_name,
        output_dir=output_dir,
        model_name=model_name,
        known_words_path=known_words_path,
        min_book_freq=min_book_freq,
        min_zipf=min_zipf,
        max_zipf=max_zipf,
        detect_phrasals=detect_phrasals,
        phrasal_model_name=phrasal_model_name,
        extract_mwe_candidates=extract_mwe_candidates,
        text=text,
    )

    tokens_df = stage1_results["tokens_df"]
    tokens_path = stage1_results["tokens_path"]
    lemma_stats_df = stage1_results["lemma_stats_df"]
    lemma_stats_path = stage1_results["lemma_stats_path"]
    phrasal_df = stage1_results["phrasal_df"]
    phrasal_path = stage1_results["phrasal_path"]
    phrasal_stats_df = stage1_results["phrasal_stats_df"]
    phrasal_stats_path = stage1_results["phrasal_stats_path"]

    # Sentence reconstruction and examples
    # Reconstruct sentences from tokens for accurate whitespace handling
    sentences = reconstruct_sentences_from_tokens(tokens_df)
    sentences_df = create_sentences_dataframe(sentences)

    lemma_with_examples = get_examples_for_lemmas(
        lemma_stats_df,
        tokens_df[[LEMMA, SENTENCE_ID]],
        sentences_df,
        top_n=top_n,
        examples_per_item=3,
    )

    phrasal_examples_df = None
    if detect_phrasals and phrasal_df is not None:
        phrasal_examples_df = get_examples_for_phrasal_verbs(
            phrasal_df[[PHRASAL, SENTENCE_ID]],
            sentences_df,
            examples_per_item=3,
        )
        phrasal_stats_df = phrasal_stats_df.merge(phrasal_examples_df, on=PHRASAL, how="left")

    # Anki export
    anki_frames = []
    if not lemma_with_examples.empty:
        top_lemmas = lemma_with_examples.sort_values(SCORE, ascending=False).head(top_n)
        anki_frames.append(prepare_anki_export(top_lemmas[[LEMMA, EXAMPLE]], book_name=book_name))
    if (
        phrasal_stats_df is not None
        and not phrasal_stats_df.empty
        and EXAMPLE in phrasal_stats_df.columns
    ):
        top_phrasals = phrasal_stats_df.sort_values(SCORE, ascending=False).head(top_n)
        anki_frames.append(
            prepare_anki_export(top_phrasals[[PHRASAL, EXAMPLE]], book_name=book_name)
        )

    anki_path = None
    if anki_frames:
        anki_export_df = pd.concat(anki_frames, ignore_index=True)
        anki_path = Path(output_dir) / DIR_ANKI_EXPORTS / f"{book_name}{TEMPLATE_ANKI}"
        anki_path.parent.mkdir(parents=True, exist_ok=True)
        export_to_anki_csv(anki_export_df, anki_path)

    lemma_stats_full_path = Path(output_dir) / f"{book_name}{TEMPLATE_LEMMA_STATS_FULL}"

    return {
        "tokens": tokens_path,
        "lemma_stats": lemma_stats_path,
        "lemma_stats_full": lemma_stats_full_path,
        "phrasal_verbs": phrasal_path,
        "phrasal_stats": phrasal_stats_path,
        "anki_csv": anki_path,
    }


def run_full_pipeline_cli() -> None:
    """CLI entry point for running the full pipeline and Anki export."""

    # Load environment variables from .env file
    load_dotenv()

    parser = argparse.ArgumentParser(description="Full pipeline: book -> stats -> Anki CSV")
    parser.add_argument(
        "--book-path", type=Path, required=True, help="Path to source book (e.g., EPUB)"
    )
    parser.add_argument("--book-name", type=str, required=True, help="Logical book identifier")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory to store outputs")
    parser.add_argument(
        "--known-words",
        type=str,
        default=None,
        help="Known words source: CSV file path or Google Sheets URL (gsheets://spreadsheet_id/worksheet_name). "
        "If not provided, will try to use GOOGLE_SHEETS_URL from .env file.",
    )
    parser.add_argument(
        "--model-name", type=str, default=DEFAULT_MODEL_NAME, help="spaCy model for tokens"
    )
    parser.add_argument("--phrasal-model", type=str, help="spaCy model for phrasal detection")
    parser.add_argument(
        "--min-book-freq",
        type=int,
        default=MIN_BOOK_FREQ_DEFAULT,
        help="Minimal book frequency threshold",
    )
    parser.add_argument(
        "--min-zipf", type=float, default=MIN_ZIPF_DEFAULT, help="Minimal global Zipf frequency"
    )
    parser.add_argument(
        "--max-zipf", type=float, default=MAX_ZIPF_DEFAULT, help="Maximal global Zipf frequency"
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=TOP_N_DEFAULT,
        help="Top lemmas to attach examples/Anki cards",
    )
    parser.add_argument(
        "--no-phrasals",
        action="store_true",
        help="Disable phrasal verb detection and stats",
    )
    parser.add_argument(
        "--no-extract-mwe-candidates",
        action="store_true",
        help="Do not write _mwe_candidates.parquet when phrasals are detected (default: write it)",
    )

    args = parser.parse_args()

    # Use --known-words if provided, otherwise try GOOGLE_SHEETS_URL from .env
    known_words_source = args.known_words
    if known_words_source is None:
        known_words_source = os.getenv("GOOGLE_SHEETS_URL")
        if known_words_source:
            print(f"Using GOOGLE_SHEETS_URL from .env: {known_words_source}")

    outputs = process_book(
        book_path=args.book_path,
        book_name=args.book_name,
        output_dir=args.output_dir,
        known_words_path=known_words_source,
        model_name=args.model_name,
        phrasal_model_name=args.phrasal_model,
        min_book_freq=args.min_book_freq,
        min_zipf=args.min_zipf,
        max_zipf=args.max_zipf,
        top_n=args.top_n,
        detect_phrasals=not args.no_phrasals,
        extract_mwe_candidates=not args.no_extract_mwe_candidates,
    )

    print("Pipeline completed. Outputs:")
    for key, value in outputs.items():
        print(f"- {key}: {value}")


if __name__ == "__main__":
    run_full_pipeline_cli()
