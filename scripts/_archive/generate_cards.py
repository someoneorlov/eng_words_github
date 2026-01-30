"""CLI script for generating Anki flashcards from WSD results.

This script:
- Loads `*_sense_tokens.parquet` and optionally `*_tokens.parquet`
- Collects book examples for each synset
- Generates SenseCards via LLM (definition, translation, examples)
- Filters book examples by spoiler_risk
- Saves cards to JSON and/or exports to Anki CSV
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import typer
from dotenv import load_dotenv

# Load .env file before importing modules that need API keys
load_dotenv()

from eng_words.constants import LEMMA, POS, SENTENCE_ID, SUPERSENSE, SYNSET_ID  # noqa: E402
from eng_words.constants.llm_config import (  # noqa: E402
    DEFAULT_CACHE_DIR,
    DEFAULT_MODEL_OPENAI,
)
from eng_words.llm import SenseCache  # noqa: E402
from eng_words.llm.card_generator import (  # noqa: E402
    CardGenerator,
    collect_book_examples,
    export_to_anki,
)
from eng_words.llm.providers.openai import OpenAIProvider  # noqa: E402
from eng_words.wsd.wordnet_utils import get_definition  # noqa: E402

app = typer.Typer(add_completion=False)


def _build_sentences_df(tokens_df: pd.DataFrame) -> pd.DataFrame:
    """Reconstruct sentences from tokens by grouping on sentence_id."""
    sorted_df = tokens_df.sort_values([SENTENCE_ID, "position"])

    if "whitespace" in sorted_df.columns:

        def join_with_ws(group: pd.DataFrame) -> str:
            parts = []
            for _, row in group.iterrows():
                parts.append(row["surface"])
                if row.get("whitespace", True):
                    parts.append(" ")
            return "".join(parts).strip()

        sentences = sorted_df.groupby(SENTENCE_ID).apply(join_with_ws, include_groups=False)
    else:
        sentences = sorted_df.groupby(SENTENCE_ID)["surface"].apply(" ".join)

    return sentences.reset_index().rename(columns={0: "text"})


def _get_unique_synsets(sense_tokens_df: pd.DataFrame) -> list[dict]:
    """Extract unique synsets with their metadata from sense_tokens."""
    # Filter valid synsets
    valid = sense_tokens_df[
        (sense_tokens_df[SYNSET_ID].notna())
        & (sense_tokens_df[SYNSET_ID] != "")
        & (sense_tokens_df[SUPERSENSE] != "unknown")
    ].copy()

    # Get unique synsets with first occurrence metadata
    unique_synsets = (
        valid.groupby(SYNSET_ID)
        .agg(
            {
                LEMMA: "first",
                POS: "first",
                SUPERSENSE: "first",
            }
        )
        .reset_index()
    )

    # Add WordNet definitions
    result = []
    for _, row in unique_synsets.iterrows():
        synset_id = row[SYNSET_ID]
        try:
            from nltk.corpus import wordnet as wn

            synset = wn.synset(synset_id)
            definition = get_definition(synset)
        except Exception:
            definition = ""

        result.append(
            {
                "synset_id": synset_id,
                "lemma": row[LEMMA],
                "pos": row[POS],
                "supersense": row[SUPERSENSE],
                "definition": definition,
            }
        )

    return result


@app.command()
def main(
    sense_tokens_path: Path = typer.Option(..., help="Path to *_sense_tokens.parquet"),
    tokens_path: Path = typer.Option(
        None, help="Path to *_tokens.parquet (optional; for building sentences)"
    ),
    book_name: str = typer.Option(..., help="Name of the book for card context"),
    cache_dir: Path = typer.Option(DEFAULT_CACHE_DIR, help="Directory for LLM cache"),
    out_dir: Path = typer.Option(Path("data/processed"), help="Output directory for cards JSON"),
    out_name: str = typer.Option("sense_cards", help="Output file name prefix"),
    model: str = typer.Option(DEFAULT_MODEL_OPENAI, help="OpenAI model to use"),
    max_synsets: int = typer.Option(None, help="Max synsets to process (for testing)"),
    dry_run: bool = typer.Option(False, help="Show stats without generating cards"),
    anki_export: bool = typer.Option(True, help="Export to Anki CSV format"),
    anki_dir: Path = typer.Option(Path("data/anki_exports"), help="Directory for Anki CSV exports"),
):
    """Generate Anki flashcards from WSD results using LLM.

    Processes unique senses from the book, generates definitions and translations,
    and selects best examples (filtering spoilers).
    """
    # Load data
    typer.echo(f"Loading sense tokens from {sense_tokens_path}")
    sense_tokens_df = pd.read_parquet(sense_tokens_path)

    # Build sentences
    if tokens_path is not None:
        tokens_df = pd.read_parquet(tokens_path)
        sentences_df = _build_sentences_df(tokens_df)
        typer.echo(f"Built {len(sentences_df)} sentences from tokens")
    else:
        inferred = str(sense_tokens_path).replace("_sense_tokens.parquet", "_tokens.parquet")
        if Path(inferred).exists():
            tokens_df = pd.read_parquet(inferred)
            sentences_df = _build_sentences_df(tokens_df)
            typer.echo(f"Inferred tokens path: {inferred}")
            typer.echo(f"Built {len(sentences_df)} sentences from tokens")
        else:
            typer.echo("ERROR: No tokens_path provided and couldn't infer")
            raise typer.Exit(1)

    # Get unique synsets
    synset_infos = _get_unique_synsets(sense_tokens_df)
    typer.echo(f"Found {len(synset_infos)} unique synsets with WSD")

    if max_synsets:
        synset_infos = synset_infos[:max_synsets]
        typer.echo(f"Limited to {max_synsets} synsets for testing")

    # Collect book examples
    typer.echo("Collecting book examples...")
    book_examples = collect_book_examples(sense_tokens_df, sentences_df)
    typer.echo(f"Collected examples for {len(book_examples)} synsets")

    # Stats
    total_examples = sum(len(exs) for exs in book_examples.values())
    avg_examples = total_examples / len(book_examples) if book_examples else 0
    typer.echo(f"Total examples: {total_examples}, avg per synset: {avg_examples:.1f}")

    if dry_run:
        typer.echo("Dry run - not generating cards")
        return

    # Initialize cache and generator
    cache = SenseCache(cache_dir=cache_dir)
    provider = OpenAIProvider(model=model)
    generator = CardGenerator(provider=provider, cache=cache)

    typer.echo(f"Cache has {len(cache)} existing cards")
    uncached = cache.get_uncached_synsets([s["synset_id"] for s in synset_infos])
    typer.echo(
        f"Will generate {len(uncached)} new cards, update {len(synset_infos) - len(uncached)} existing"
    )

    # Generate cards
    typer.echo(f"Generating cards with {model}...")
    cards = generator.generate_all(synset_infos, book_examples, book_name)
    typer.echo(f"Generated/updated {len(cards)} cards")

    # Save to JSON
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{out_name}.json"

    cards_data = [card.to_dict() for card in cards]
    out_path.write_text(json.dumps(cards_data, ensure_ascii=False, indent=2), encoding="utf-8")
    typer.echo(f"Saved cards to {out_path}")

    # Summary
    with_book_examples = sum(1 for c in cards if c.book_examples)
    typer.echo(f"Cards with book examples: {with_book_examples}/{len(cards)}")

    # Export to Anki
    if anki_export:
        anki_dir = Path(anki_dir)
        anki_dir.mkdir(parents=True, exist_ok=True)
        anki_path = anki_dir / f"{out_name}_anki.csv"

        exported = export_to_anki(cards, anki_path, book_name=book_name)
        typer.echo(f"Exported {exported} cards to Anki: {anki_path}")


if __name__ == "__main__":
    app()
