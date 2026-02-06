#!/usr/bin/env python3
"""Pipeline B (Word Family) via Gemini Batch API — production script.

Full dataset: run this script (Batch API, ~50% cheaper).
Small samples: same script with --limit N.

No first-example fallback: when indices invalid, examples stay empty and are flagged for retry.
Use --retry-empty (prompt+reminder) and optionally --retry-thinking (Pro model with Thinking).
On full book (~4200 cards): ~27 cards with invalid indices, all go to retry (cost: pennies).
Thinking retry uses THINKING_MODEL (2.5 Pro or 3 Pro), not Flash — Flash has no Thinking mode.

Usage (from project root):
    uv run python scripts/run_pipeline_b_batch.py create [--limit N] [--max-examples 50]
    uv run python scripts/run_pipeline_b_batch.py create --no-batch [--limit 50]  # small test via Standard API
    uv run python scripts/run_pipeline_b_batch.py status
    uv run python scripts/run_pipeline_b_batch.py download
    uv run python scripts/run_pipeline_b_batch.py wait

Requires: GOOGLE_API_KEY. Input: data/experiment/tokens_sample.parquet,
data/experiment/sentences_sample.parquet (see scripts/prepare_pipeline_b_sample.py).
"""

import json
import sys
from pathlib import Path

import typer
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eng_words.word_family.batch import (
    BATCH_DIR,
    DATA_DIR,
    get_client,
    LEMMA_EXAMPLES_PATH,
    OUTPUT_CARDS_PATH,
    RESULTS_PATH,
    SENTENCES_PATH,
    TOKENS_PATH,
)
from eng_words.word_family.batch_io import (
    create_batch as _create_batch,
    download_batch as _download_batch,
    get_batch_status as _get_batch_status,
    list_retry_candidates as _list_retry_candidates,
    parse_results as _parse_results,
    render_requests as _render_requests,
    run_standard_api as _run_standard_api,
    wait_for_batch as _wait_for_batch,
)
from eng_words.word_family.batch_schemas import BatchConfig

app = typer.Typer(help="Pipeline B via Gemini Batch API")

# Same as rest of project (pipeline, run_synset_card_generation, etc.)
MODEL = "gemini-3-flash-preview"
# For --retry-thinking: Pro models have Thinking mode (2.5 Flash has no Thinking)
THINKING_MODEL = "gemini-2.5-pro"


def _batch_config(
    limit: int | None = None,
    max_examples: int = 50,
    model: str = MODEL,
    strict: bool = True,
) -> BatchConfig:
    """Build BatchConfig from module default paths (for use with batch_io)."""
    return BatchConfig(
        tokens_path=TOKENS_PATH,
        sentences_path=SENTENCES_PATH,
        batch_dir=BATCH_DIR,
        output_cards_path=OUTPUT_CARDS_PATH,
        limit=limit if limit and limit > 0 else None,
        max_examples=max_examples,
        model=model,
        strict=strict,
    )


@app.command()
def create(
    limit: int = typer.Option(0, help="Limit number of lemmas (0=all)"),
    max_examples: int = typer.Option(
        50,
        help="Max examples per lemma (50 recommended to avoid MAX_TOKENS)",
    ),
    model: str = typer.Option(MODEL, help="Gemini model"),
    overwrite: bool = typer.Option(False, "--overwrite", help="Overwrite existing batch_info.json"),
    no_batch: bool = typer.Option(
        False,
        "--no-batch/--batch",
        help="Use Standard API instead of Batch (for small test runs, e.g. --limit 50)",
    ),
):
    """Create batch requests for Pipeline B (one request per lemma). With --no-batch: run Standard API and write cards (no batch job)."""
    get_client()  # fail fast if no API key
    config = _batch_config(limit=limit if limit > 0 else None, max_examples=max_examples, model=model)
    if no_batch:
        print("Running via Standard API (no batch job)...")
        _run_standard_api(config)
        print("Parsing results and writing cards...")
        _download_batch(config, from_file=True)
        print("\n✅ Done. Cards written (Standard API).")
        return
    print("Loading data and creating batch...")
    info = _create_batch(config, overwrite=overwrite)
    n = info.lemmas_count
    print("\n✅ Batch created!")
    print(f"   Name: {info.batch_name}")
    print(f"   Lemmas: {n}")
    print("\nRun: uv run python scripts/run_pipeline_b_batch.py status")


@app.command("render-requests")
def render_requests_cli(
    limit: int = typer.Option(0, help="Limit number of lemmas (0=all)"),
    max_examples: int = typer.Option(50, help="Max examples per lemma"),
    model: str = typer.Option(MODEL, help="Gemini model (for config only)"),
):
    """Build requests.jsonl and lemma_examples.json from tokens/sentences. No API calls."""
    config = _batch_config(limit=limit if limit > 0 else None, max_examples=max_examples, model=model)
    try:
        _render_requests(config)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
        raise SystemExit(1) from e
    print(f"Wrote {BATCH_DIR / 'requests.jsonl'} and {LEMMA_EXAMPLES_PATH}")


@app.command("parse-results")
def parse_results_cli(
    skip_validation: bool = typer.Option(False, "--skip-validation", help="Do not validate card fields"),
):
    """Parse existing results.jsonl and write cards. No API download. Requires results.jsonl and lemma_examples.json."""
    config = _batch_config()
    try:
        _parse_results(config, skip_validation=skip_validation)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
        raise SystemExit(1) from e
    print(f"Output: {OUTPUT_CARDS_PATH}")


@app.command()
def status():
    """Check batch job status."""
    try:
        s = _get_batch_status(BATCH_DIR)
    except FileNotFoundError:
        print("No batch info. Run 'create' first.")
        raise SystemExit(1)
    print(f"Batch: {s.get('batch_name')}")
    print(f"State: {s['state']}")
    print(f"Model: {s.get('model')}")
    print(f"Lemmas: {s.get('lemmas_count')}")
    if s["state"] == "JOB_STATE_SUCCEEDED":
        print("\n✅ Complete. Run 'download' to get results.")
    elif s["state"] == "JOB_STATE_FAILED":
        print("\n❌ Failed.")
        if s.get("error"):
            print(f"Error: {s['error']}")


@app.command()
def download(
    skip_validation: bool = typer.Option(
        False,
        "--skip-validation",
        help="Do not validate card fields (definition_en, definition_ru, part_of_speech)",
    ),
    retry_empty: bool = typer.Option(
        True,
        "--retry-empty/--no-retry-empty",
        help="Retry lemmas with empty/fallback examples via Standard API (prompt + reminder)",
    ),
    retry_thinking: bool = typer.Option(
        False,
        "--retry-thinking/--no-retry-thinking",
        help="Second retry with Thinking model for lemmas still empty after first retry",
    ),
    from_file: bool = typer.Option(
        False,
        "--from-file/--no-from-file",
        help="Use existing results.jsonl (no API download). For testing retry on same data.",
    ),
    run_gate: bool = typer.Option(
        False,
        "--run-gate/--no-run-gate",
        help="After download, run QC gate (validation_errors vs thresholds); exit 1 if FAIL (Stage 7).",
    ),
    no_strict: bool = typer.Option(
        False,
        "--no-strict",
        help="Write cards even if QC gate fails (e.g. pos_mismatch); gate result still in download_log.",
    ),
):
    """Download batch results and write cards_B_batch.json.

    To test new retry logic on same data (no API download): use --from-file.
    Requires: data/experiment/batch_b/results.jsonl and lemma_examples.json.
    Example: uv run python scripts/run_pipeline_b_batch.py download --from-file
    With QC gate: download --run-gate (or run scripts/run_quality_investigation.py --gate separately).
    """
    config = _batch_config(strict=not no_strict)
    try:
        _download_batch(
            config,
            from_file=from_file,
            skip_validation=skip_validation,
            retry_empty=retry_empty,
            retry_thinking=retry_thinking,
            thinking_model=THINKING_MODEL,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        print(f"Error: {e}")
        raise SystemExit(1) from e

    log_path = BATCH_DIR / "download_log.json"
    result_path = config.output_cards_path
    if not result_path.exists():
        print("Download completed but output file was not written.")
        raise SystemExit(1)
    result = json.loads(result_path.read_text(encoding="utf-8"))
    stats = result.get("stats", {})
    n_cards = stats.get("cards_generated", 0)
    n_errors = stats.get("errors", 0)
    val_errors = result.get("validation_errors", [])
    if val_errors:
        print(f"\n⚠ Validation issues: {len(val_errors)} cards (not in output)")
        for e in val_errors[:5]:
            lemma = e.get("lemma", "?")
            stage = e.get("stage", "?")
            msg = e.get("message", "")[:60]
            print(f"   {lemma} [{stage}]: {msg}")
        if len(val_errors) > 5:
            print(f"   ... and {len(val_errors) - 5} more — see download_log.json validation_errors")
    qc_list = result.get("stats", {}).get("cards_lemma_not_in_example_count", 0)
    if qc_list:
        print(f"\n⚠ Cards for review (lemma not in example): {qc_list} — see download_log.json")
    print(f"\n✅ Done. {n_cards} cards, {n_errors} parse errors.")
    print(f"Output: {OUTPUT_CARDS_PATH}")
    print(f"Log: {log_path}")

    if run_gate:
        from eng_words.word_family.qc_gate import evaluate_gate
        passed, summary, msg = evaluate_gate(result)
        print(f"\nQC gate: {msg}")
        if not passed:
            print("Gate FAIL. Fix validation_errors or relax thresholds. Exiting with code 1.")
            raise SystemExit(1)
        print("Gate PASS.")


@app.command("list-retry-candidates")
def list_retry_candidates():
    """Parse results.jsonl and list lemmas that would be retried (empty/fallback examples). No API calls."""
    if not RESULTS_PATH.exists():
        print(f"Missing {RESULTS_PATH}. Run batch download first.")
        raise SystemExit(1)
    if not LEMMA_EXAMPLES_PATH.exists():
        print(f"Missing {LEMMA_EXAMPLES_PATH}. Re-run 'create' first.")
        raise SystemExit(1)
    lemmas_empty, lemmas_fallback = _list_retry_candidates(RESULTS_PATH, LEMMA_EXAMPLES_PATH)
    lemmas_to_retry = sorted(lemmas_empty | lemmas_fallback)
    print("=== Retry candidates (0 examples or examples_fallback) ===\n")
    print(f"Lemmas with empty examples (lemma had examples): {len(lemmas_empty)}")
    print(f"Lemmas with examples_fallback (out-of-range indices): {len(lemmas_fallback)}")
    print(f"Total unique lemmas to retry: {len(lemmas_to_retry)}")
    if lemmas_to_retry:
        print("\nLemmas (first 40):")
        for lemma in lemmas_to_retry[:40]:
            flags = []
            if lemma in lemmas_empty:
                flags.append("empty")
            if lemma in lemmas_fallback:
                flags.append("fallback")
            print(f"  {lemma} [{', '.join(flags)}]")
        if len(lemmas_to_retry) > 40:
            print(f"  ... and {len(lemmas_to_retry) - 40} more")
        print("\nRun: uv run python scripts/run_pipeline_b_batch.py download")
        print("to retry these via Standard API (prompt + reminder).")


@app.command()
def wait():
    """Wait for batch to complete, then download."""
    try:
        print("Waiting for batch...")
        _wait_for_batch(BATCH_DIR)
    except FileNotFoundError:
        print("No batch info. Run 'create' first.")
        raise SystemExit(1)
    except RuntimeError as e:
        print(f"\n❌ {e}")
        raise SystemExit(1)
    print("\n✅ Complete!")
    config = _batch_config(strict=False)  # write cards even if gate fails (e.g. 1 pos_mismatch)
    _download_batch(config, from_file=False, retry_empty=True, retry_thinking=False, skip_validation=False)
    print(f"Output: {OUTPUT_CARDS_PATH}")


if __name__ == "__main__":
    app()
