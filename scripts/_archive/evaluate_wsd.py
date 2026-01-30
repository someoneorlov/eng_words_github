"""CLI script for WSD evaluation using LLM jury.

This script:
- Loads `*_sense_tokens.parquet` and optionally `*_tokens.parquet` (for sentences)
- Samples tokens for evaluation (stratified by supersense)
- Builds blind evaluation prompts (A/B/C candidate senses)
- Runs 1 or 2 LLM providers as a jury
- Saves per-sample results + summary metrics to JSON
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import typer
from dotenv import load_dotenv

# Load .env file before importing modules that need API keys
load_dotenv()

from nltk.corpus import wordnet as wn  # noqa: E402

from eng_words.constants import POS, SENTENCE_ID, SYNSET_ID  # noqa: E402
from eng_words.constants.llm_config import DEFAULT_CACHE_DIR, DEFAULT_MODEL_OPENAI  # noqa: E402
from eng_words.llm import WSDEvaluator, compute_metrics, sample_tokens_for_evaluation  # noqa: E402
from eng_words.llm.providers.openai import OpenAIProvider  # noqa: E402
from eng_words.wsd.wordnet_utils import get_definition, get_synsets  # noqa: E402

app = typer.Typer(add_completion=False)


def _build_sentences_df(tokens_df: pd.DataFrame) -> pd.DataFrame:
    """Reconstruct sentences from tokens by grouping on sentence_id.

    Returns DataFrame with columns: sentence_id, text.
    """
    # Sort by position to get correct token order
    sorted_df = tokens_df.sort_values([SENTENCE_ID, "position"])

    # Join surfaces with whitespace info if available
    if "whitespace" in sorted_df.columns:
        # Build sentence with proper whitespace
        def join_with_ws(group: pd.DataFrame) -> str:
            parts = []
            for _, row in group.iterrows():
                parts.append(row["surface"])
                if row.get("whitespace", True):
                    parts.append(" ")
            return "".join(parts).strip()

        sentences = sorted_df.groupby(SENTENCE_ID).apply(join_with_ws, include_groups=False)
    else:
        # Simple space join
        sentences = sorted_df.groupby(SENTENCE_ID)["surface"].apply(" ".join)

    return sentences.reset_index().rename(columns={0: "text"})


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _build_candidates(lemma: str, pos: str | None, max_candidates: int) -> list[dict]:
    """Get candidate synsets for lemma and build (synset_id, definition) dicts."""
    synsets = get_synsets(lemma, pos)
    synsets = synsets[:max_candidates]
    candidates: list[dict] = []
    for s in synsets:
        candidates.append({"synset_id": s.name(), "definition": get_definition(s)})
    return candidates


@app.command()
def main(
    sense_tokens_path: Path = typer.Option(..., help="Path to *_sense_tokens.parquet"),
    tokens_path: Path = typer.Option(
        None, help="Path to *_tokens.parquet (optional; for building sentences)"
    ),
    out_dir: Path = typer.Option(DEFAULT_CACHE_DIR / "evaluations", help="Output directory"),
    out_name: str = typer.Option("wsd_eval", help="Output file name prefix"),
    n_samples: int = typer.Option(200, help="Number of samples for evaluation"),
    random_state: int = typer.Option(42, help="Random seed for reproducibility"),
    max_candidates: int = typer.Option(5, help="Max candidate synsets per item"),
    models: str = typer.Option(
        DEFAULT_MODEL_OPENAI,
        help="Comma-separated list of models for jury voting (e.g., 'gpt-4o-mini,gpt-4o,gpt-4.1-mini')",
    ),
):
    """Evaluate WSD quality using LLM jury voting.

    Confidence is computed as agreement between models (not LLM self-report).
    Use multiple models for reliable confidence estimates.
    """
    _ensure_dir(out_dir)

    sense_tokens_df = pd.read_parquet(sense_tokens_path)

    # Build sentences from tokens if tokens_path is provided
    if tokens_path is not None:
        tokens_df = pd.read_parquet(tokens_path)
        sentences_df = _build_sentences_df(tokens_df)
        typer.echo(f"Built {len(sentences_df)} sentences from tokens")
    else:
        # Try to infer tokens_path from sense_tokens_path
        inferred = str(sense_tokens_path).replace("_sense_tokens.parquet", "_tokens.parquet")
        if Path(inferred).exists():
            tokens_df = pd.read_parquet(inferred)
            sentences_df = _build_sentences_df(tokens_df)
            typer.echo(f"Inferred tokens path: {inferred}")
            typer.echo(f"Built {len(sentences_df)} sentences from tokens")
        else:
            typer.echo("ERROR: No tokens_path provided and couldn't infer from sense_tokens_path")
            raise typer.Exit(1)

    sample_df = sample_tokens_for_evaluation(
        sense_tokens_df=sense_tokens_df,
        sentences_df=sentences_df,
        n_samples=n_samples,
        random_state=random_state,
    )

    # Create providers from comma-separated model list
    model_list = [m.strip() for m in models.split(",") if m.strip()]
    providers = [OpenAIProvider(model=m) for m in model_list]
    typer.echo(f"Using {len(providers)} model(s) for jury: {model_list}")

    evaluator = WSDEvaluator(providers=providers)

    results = []
    for row in sample_df.to_dict(orient="records"):
        lemma = row["lemma"]
        pos = row.get(POS, None)
        assigned_synset = row.get(SYNSET_ID, "")
        sentence_text = row["sentence_text"]

        candidates = _build_candidates(lemma=lemma, pos=pos, max_candidates=max_candidates)
        # If WSD picked a synset not in candidates (e.g., due to POS fallback), add it.
        if assigned_synset and assigned_synset not in {c["synset_id"] for c in candidates}:
            # Load definition for the assigned synset
            try:
                assigned_synset_obj = wn.synset(assigned_synset)
                assigned_definition = get_definition(assigned_synset_obj)
            except Exception:
                assigned_definition = f"[Synset {assigned_synset} - definition not available]"
            candidates = [
                {"synset_id": assigned_synset, "definition": assigned_definition}
            ] + candidates
            candidates = candidates[:max_candidates]

        res = evaluator.evaluate_sample(
            lemma=lemma,
            sentence_text=sentence_text,
            assigned_synset=assigned_synset,
            candidate_synsets=candidates,
            pos=pos or "NOUN",
        )
        results.append(res)

    metrics = compute_metrics(results)

    payload = {
        "inputs": {
            "sense_tokens_path": str(sense_tokens_path),
            "tokens_path": str(tokens_path) if tokens_path else "inferred",
            "n_samples": n_samples,
            "random_state": random_state,
            "max_candidates": max_candidates,
            "models": model_list,
            "jury_size": len(model_list),
        },
        "metrics": metrics,
        "results": [asdict(r) for r in results],
    }

    out_path = out_dir / f"{out_name}.json"
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))

    typer.echo(f"Saved: {out_path}")
    typer.echo(
        f"Accuracy(strict): {metrics['accuracy_strict']:.3f} | Coverage: {metrics['coverage']:.3f}"
    )


if __name__ == "__main__":
    app()
