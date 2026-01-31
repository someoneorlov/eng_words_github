"""Stage 1 pipeline orchestration."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from .anki_export import export_to_anki_csv, prepare_anki_export
from .constants import (
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
    SENTENCE_ID,
    TEMPLATE_ANKI,
    TEMPLATE_LEMMA_STATS,
    TEMPLATE_LEMMA_STATS_FULL,
    TEMPLATE_PHRASAL_VERB_STATS,
    TEMPLATE_PHRASAL_VERBS,
    TEMPLATE_SENSE_TOKENS,
    TEMPLATE_SUPERSENSE_STATS,
    TEMPLATE_TOKENS,
    TOP_N_DEFAULT,
)
from .examples import get_examples_for_lemmas, get_examples_for_phrasal_verbs
from .filtering import (
    filter_by_frequency,
    filter_by_supersense,
    filter_known_words,
    rank_candidates,
)
from .phrasal_verbs import (
    calculate_phrasal_verb_stats,
    detect_phrasal_verbs,
    filter_phrasal_verbs,
    initialize_phrasal_model,
    rank_phrasal_verbs,
)
from .statistics import (
    add_global_frequency,
    calculate_lemma_frequency,
    save_lemma_stats_to_parquet,
)
from .storage import load_known_words
from .text_io import load_book_text
from .text_processing import (
    create_sentences_dataframe,
    create_tokens_dataframe,
    initialize_spacy_model,
    reconstruct_sentences_from_tokens,
    save_tokens_to_parquet,
    tokenize_text_in_chunks,
)

# Constants for smart card generation
DEFAULT_SMART_CARD_PROVIDER = "gemini"
DEFAULT_SMART_CARD_MODEL = "gemini-3-flash-preview"
SMART_CARDS_CACHE_DIR = Path("data/cache/llm_responses")


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
    text: str | None = None,
    enable_wsd: bool = False,
    wsd_checkpoint_interval: int = 5000,
    min_sense_freq: int | None = None,
    max_senses: int | None = None,
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
        text: Pre-loaded text (optional, for testing)
        enable_wsd: Whether to run Word Sense Disambiguation
        wsd_checkpoint_interval: Checkpoint interval for WSD (tokens)
        min_sense_freq: Minimum frequency for a sense to be included (WSD only)
        max_senses: Maximum number of senses per lemma (WSD only)

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

    # WSD annotation (optional)
    sense_tokens_df = None
    sense_tokens_path = None
    supersense_stats_df = None
    supersense_stats_path = None

    if enable_wsd:
        from .wsd import WordNetSenseBackend

        # Reconstruct sentences for WSD
        sentences = reconstruct_sentences_from_tokens(tokens_df)
        sentences_df = create_sentences_dataframe(sentences)

        print("Running Word Sense Disambiguation...")
        wsd_backend = WordNetSenseBackend()
        checkpoint_path = output_dir / f"{book_name}_wsd_checkpoint.parquet"

        sense_tokens_df = wsd_backend.annotate(
            tokens_df,
            sentences_df,
            checkpoint_path=checkpoint_path,
            checkpoint_interval=wsd_checkpoint_interval,
            show_progress=True,
        )

        sense_tokens_path = output_dir / f"{book_name}{TEMPLATE_SENSE_TOKENS}"
        sense_tokens_df.to_parquet(sense_tokens_path, index=False)

        # Remove checkpoint after successful completion
        if checkpoint_path.exists():
            checkpoint_path.unlink()

        # Aggregate supersense statistics
        supersense_stats_df = wsd_backend.aggregate(sense_tokens_df)

        # Apply supersense filtering if parameters provided
        if min_sense_freq is not None or max_senses is not None:
            supersense_stats_df = filter_by_supersense(
                supersense_stats_df,
                min_sense_freq=min_sense_freq,
                max_senses=max_senses,
            )

        supersense_stats_path = output_dir / f"{book_name}{TEMPLATE_SUPERSENSE_STATS}"
        supersense_stats_df.to_parquet(supersense_stats_path, index=False)
        print(f"WSD complete. Saved to {sense_tokens_path}")

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
    if known_words_path:
        try:
            known_df = load_known_words(known_words_path)
            if not known_df.empty:
                lemma_stats = filter_known_words(lemma_stats, known_df)
        except (ValueError, FileNotFoundError, Exception) as e:
            print(f"⚠️  Warning: Failed to load known words from {known_words_path}: {e}")
            print("   Continuing without known words filtering...")
    lemma_stats = filter_by_frequency(
        lemma_stats,
        min_book_freq=min_book_freq,
        min_zipf=min_zipf,
        max_zipf=max_zipf,
    )
    lemma_stats = rank_candidates(lemma_stats)
    stats_path = output_dir / f"{book_name}{TEMPLATE_LEMMA_STATS}"
    save_lemma_stats_to_parquet(lemma_stats, stats_path)

    phrasal_df = None
    phrasal_path = None
    phrasal_stats_df = None
    phrasal_stats_path = None
    if detect_phrasals:
        parser_nlp = initialize_phrasal_model(phrasal_model_name or model_name)
        phrasal_df = detect_phrasal_verbs(tokens_df, parser_nlp)
        phrasal_path = output_dir / f"{book_name}{TEMPLATE_PHRASAL_VERBS}"
        phrasal_df.to_parquet(phrasal_path, index=False)
        phrasal_stats_df = calculate_phrasal_verb_stats(phrasal_df)
        phrasal_known = None
        if known_words_path:
            known_df = load_known_words(known_words_path)
            phrasal_known = known_df[known_df[ITEM_TYPE] == ITEM_TYPE_PHRASAL_VERB]
        phrasal_stats_df = filter_phrasal_verbs(phrasal_stats_df, phrasal_known)
        phrasal_stats_df = rank_phrasal_verbs(phrasal_stats_df)
        phrasal_stats_path = output_dir / f"{book_name}{TEMPLATE_PHRASAL_VERB_STATS}"
        phrasal_stats_df.to_parquet(phrasal_stats_path, index=False)

    return {
        "tokens_df": tokens_df,
        "tokens_path": tokens_path,
        "lemma_stats_df": lemma_stats,
        "lemma_stats_path": stats_path,
        "phrasal_df": phrasal_df,
        "phrasal_path": phrasal_path,
        "phrasal_stats_df": phrasal_stats_df,
        "phrasal_stats_path": phrasal_stats_path,
        "sense_tokens_df": sense_tokens_df,
        "sense_tokens_path": sense_tokens_path,
        "supersense_stats_df": supersense_stats_df,
        "supersense_stats_path": supersense_stats_path,
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
    enable_wsd: bool = False,
    wsd_checkpoint_interval: int = 5000,
    min_sense_freq: int | None = None,
    max_senses: int | None = None,
    smart_cards: bool = False,
    smart_cards_provider: str = DEFAULT_SMART_CARD_PROVIDER,
    smart_cards_model: str | None = None,
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
        enable_wsd: Whether to run Word Sense Disambiguation
        wsd_checkpoint_interval: Checkpoint interval for WSD

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
        text=text,
        enable_wsd=enable_wsd,
        wsd_checkpoint_interval=wsd_checkpoint_interval,
        min_sense_freq=min_sense_freq,
        max_senses=max_senses,
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

    # Smart Cards generation (requires WSD)
    smart_cards_path = None
    if smart_cards:
        supersense_stats_df = stage1_results.get("supersense_stats_df")
        if supersense_stats_df is None or supersense_stats_df.empty:
            print("⚠️  Smart cards require WSD. Use --enable-wsd flag.")
        else:
            smart_cards_path = generate_smart_cards(
                supersense_stats_df=supersense_stats_df,
                tokens_df=tokens_df,
                sentences_df=sentences_df,
                book_name=book_name,
                output_dir=Path(output_dir),
                provider_name=smart_cards_provider,
                model_name=smart_cards_model,
                top_n=top_n,
            )

    return {
        "tokens": tokens_path,
        "lemma_stats": lemma_stats_path,
        "lemma_stats_full": lemma_stats_full_path,
        "phrasal_verbs": phrasal_path,
        "phrasal_stats": phrasal_stats_path,
        "anki_csv": anki_path,
        "sense_tokens": stage1_results["sense_tokens_path"],
        "supersense_stats": stage1_results["supersense_stats_path"],
        "smart_cards": smart_cards_path,
    }


def generate_smart_cards(
    supersense_stats_df: pd.DataFrame,
    tokens_df: pd.DataFrame,
    sentences_df: pd.DataFrame,
    book_name: str,
    output_dir: Path,
    *,
    provider_name: str = DEFAULT_SMART_CARD_PROVIDER,
    model_name: str | None = None,
    top_n: int = TOP_N_DEFAULT,
    examples_per_sense: int = 10,
) -> Path | None:
    """Generate smart flashcards using LLM.

    Args:
        supersense_stats_df: DataFrame with supersense statistics from WSD.
        tokens_df: DataFrame with tokens.
        sentences_df: DataFrame with sentences.
        book_name: Book name for context.
        output_dir: Output directory for cards.
        provider_name: LLM provider name (gemini, openai, anthropic).
        model_name: Specific model name (optional).
        top_n: Number of top senses to generate cards for.
        examples_per_sense: Max examples per sense to send to LLM.

    Returns:
        Path to generated cards JSON file, or None if failed.
    """
    from eng_words.llm.base import get_provider
    from eng_words.llm.response_cache import ResponseCache
    from eng_words.llm.smart_card_generator import SmartCardGenerator

    print(f"\n📝 Generating Smart Cards with {provider_name}...")

    # Initialize provider and cache
    provider = get_provider(provider_name, model=model_name)
    cache = ResponseCache(cache_dir=SMART_CARDS_CACHE_DIR)

    generator = SmartCardGenerator(
        provider=provider,
        cache=cache,
        book_name=book_name,
    )

    # Prepare items from supersense stats
    # Group tokens by (lemma, supersense) and get examples
    items = []

    # Get top senses by frequency
    top_senses = supersense_stats_df.nlargest(top_n, "freq")

    for _, row in top_senses.iterrows():
        lemma = row["lemma"]
        pos = row["pos"]
        supersense = row["supersense"]
        wn_definition = row.get("wn_definition", "")

        # Get sentence IDs for this lemma
        lemma_tokens = tokens_df[(tokens_df[LEMMA] == lemma)]

        if lemma_tokens.empty:
            continue

        # Get unique sentences
        sentence_ids = lemma_tokens[SENTENCE_ID].unique()[:examples_per_sense]
        examples = []
        for sid in sentence_ids:
            sent_row = sentences_df[sentences_df[SENTENCE_ID] == sid]
            if not sent_row.empty:
                examples.append(sent_row.iloc[0]["sentence_text"])

        if not examples:
            continue

        items.append(
            {
                "lemma": lemma,
                "pos": pos,
                "supersense": supersense,
                "wn_definition": wn_definition,
                "examples": examples,
            }
        )

    if not items:
        print("⚠️  No items to generate cards for")
        return None

    print(f"   Generating {len(items)} cards...")

    # Generate cards
    cards = generator.generate_batch(items, progress=True)

    # Save results
    import json

    output_path = output_dir / f"{book_name}_smart_cards.json"
    cards_data = [card.to_dict() for card in cards]

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(cards_data, f, ensure_ascii=False, indent=2)

    # Also export Anki-ready CSV
    anki_rows = [card.to_anki_row() for card in cards]
    anki_df = pd.DataFrame(anki_rows)
    anki_path = output_dir / DIR_ANKI_EXPORTS / f"{book_name}_smart_anki.csv"
    anki_path.parent.mkdir(parents=True, exist_ok=True)
    anki_df.to_csv(anki_path, index=False, sep="\t")

    # Print stats
    stats = generator.stats()
    print("\n✅ Smart Cards generated:")
    print(f"   Total: {stats['total_cards']}")
    print(f"   Successful: {stats['successful']}")
    print(f"   Failed: {stats['failed']}")
    print(f"   Cost: ${stats['total_cost']:.4f}")
    print(f"   Cache hits: {stats['cache_stats']['hits']}")
    print(f"   Saved to: {output_path}")
    print(f"   Anki CSV: {anki_path}")

    return output_path


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
        "--enable-wsd",
        action="store_true",
        help="Enable Word Sense Disambiguation (adds supersense annotations)",
    )
    parser.add_argument(
        "--wsd-checkpoint-interval",
        type=int,
        default=5000,
        help="Checkpoint interval for WSD (number of tokens)",
    )
    parser.add_argument(
        "--min-sense-freq",
        type=int,
        default=None,
        help="Minimum frequency for a sense to be included (WSD only)",
    )
    parser.add_argument(
        "--max-senses",
        type=int,
        default=None,
        help="Maximum number of senses per lemma (WSD only, keeps top N by frequency)",
    )
    parser.add_argument(
        "--smart-cards",
        action="store_true",
        help="Generate smart flashcards using LLM (requires --enable-wsd)",
    )
    parser.add_argument(
        "--smart-cards-provider",
        type=str,
        default=DEFAULT_SMART_CARD_PROVIDER,
        choices=["gemini", "openai", "anthropic"],
        help="LLM provider for smart cards (default: gemini)",
    )
    parser.add_argument(
        "--smart-cards-model",
        type=str,
        default=None,
        help="Specific LLM model for smart cards (optional)",
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
        enable_wsd=args.enable_wsd,
        wsd_checkpoint_interval=args.wsd_checkpoint_interval,
        min_sense_freq=args.min_sense_freq,
        max_senses=args.max_senses,
        smart_cards=args.smart_cards,
        smart_cards_provider=args.smart_cards_provider,
        smart_cards_model=args.smart_cards_model,
    )

    print("Pipeline completed. Outputs:")
    for key, value in outputs.items():
        print(f"- {key}: {value}")


if __name__ == "__main__":
    run_full_pipeline_cli()
