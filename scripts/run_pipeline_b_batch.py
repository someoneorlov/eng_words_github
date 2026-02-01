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
    uv run python scripts/run_pipeline_b_batch.py status
    uv run python scripts/run_pipeline_b_batch.py download
    uv run python scripts/run_pipeline_b_batch.py wait

Requires: GOOGLE_API_KEY. Input: data/experiment/tokens_sample.parquet,
data/experiment/sentences_sample.parquet (see scripts/prepare_pipeline_b_sample.py).
"""

import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
import typer
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eng_words.word_family import (
    CLUSTER_PROMPT_TEMPLATE,
    group_examples_by_lemma,
)

try:
    from eng_words.validation.example_validator import _get_word_forms, _word_in_text
except ImportError:
    _get_word_forms = _word_in_text = None

app = typer.Typer(help="Pipeline B via Gemini Batch API")

# Forms that look like the lemma but belong to another word (e.g. hop vs hope)
HOMONYM_EXCLUDE_FORMS: dict[str, set[str]] = {
    "hop": {"hoped", "hoping"},
    "hope": {"hopped", "hopping"},
}

# Paths (relative to project root)
DATA_DIR = PROJECT_ROOT / "data" / "experiment"
BATCH_DIR = DATA_DIR / "batch_b"
TOKENS_PATH = DATA_DIR / "tokens_sample.parquet"
SENTENCES_PATH = DATA_DIR / "sentences_sample.parquet"
BATCH_INFO_PATH = BATCH_DIR / "batch_info.json"
REQUESTS_PATH = BATCH_DIR / "requests.jsonl"
RESULTS_PATH = BATCH_DIR / "results.jsonl"
LEMMA_EXAMPLES_PATH = BATCH_DIR / "lemma_examples.json"
OUTPUT_CARDS_PATH = DATA_DIR / "cards_B_batch.json"

# Same as rest of project (pipeline, run_synset_card_generation, etc.)
MODEL = "gemini-3-flash-preview"
# For --retry-thinking: Pro models have Thinking mode (2.5 Flash has no Thinking)
THINKING_MODEL = "gemini-2.5-pro"

# Required card keys for validation
REQUIRED_CARD_KEYS = {"definition_en", "definition_ru", "part_of_speech"}


def get_client() -> genai.Client:
    """Return authenticated Gemini client. Raises ValueError if GOOGLE_API_KEY not set."""
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not set. Set it in .env or environment.")
    return genai.Client(api_key=api_key)


def load_lemma_groups(
    limit: int | None = None,
    max_examples: int = 50,
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Load tokens + sentences, group by lemma; return DataFrame and lemma -> examples map."""
    if not TOKENS_PATH.exists():
        raise FileNotFoundError(
            f"Tokens not found: {TOKENS_PATH}. "
            "Run: uv run python scripts/prepare_pipeline_b_sample.py [--book NAME] [--size N]"
        )
    if not SENTENCES_PATH.exists():
        raise FileNotFoundError(
            f"Sentences not found: {SENTENCES_PATH}. "
            "Run: uv run python scripts/prepare_pipeline_b_sample.py [--book NAME] [--size N]"
        )
    tokens = pd.read_parquet(TOKENS_PATH)
    sentences = pd.read_parquet(SENTENCES_PATH)
    lemma_groups = group_examples_by_lemma(tokens, sentences)

    if limit:
        lemma_groups = lemma_groups.head(limit)

    lemma_examples: dict[str, list[str]] = {}
    for _, row in lemma_groups.iterrows():
        ex = list(row["examples"])
        if len(ex) > max_examples:
            ex = ex[:max_examples]
        lemma_examples[row["lemma"]] = ex

    return lemma_groups, lemma_examples


def build_prompt(lemma: str, examples: list[str]) -> str:
    """Build Pipeline B cluster prompt for one lemma."""
    numbered = "\n".join(f"{i+1}. {ex}" for i, ex in enumerate(examples))
    prompt = CLUSTER_PROMPT_TEMPLATE.format(lemma=lemma, numbered_examples=numbered)
    n = len(examples)
    if n > 0:
        prompt += (
            f"\n\nIMPORTANT: selected_example_indices must be 1-based (1 = first example, 2 = second, ...). "
            f"Use only integers from 1 to {n} — you have exactly {n} examples above."
        )
    return prompt


def build_retry_prompt(lemma: str, examples: list[str]) -> str:
    """Same as build_prompt plus a concatenated line asking to fix indices (for retry)."""
    prompt = build_prompt(lemma, examples)
    n = len(examples)
    if n > 0:
        prompt += (
            f"\n\nCRITICAL: Your previous response had invalid selected_example_indices. "
            f"Use only 1-based indices from 1 to {n} (you have exactly {n} examples above)."
        )
    return prompt


def _call_standard_api_for_retry(
    client: genai.Client,
    lemma: str,
    lemma_examples: dict[str, list[str]],
    model: str,
    use_thinking: bool = False,
) -> dict | None:
    """Call Gemini Standard API with retry prompt; return response dict (batch shape) or None.
    When use_thinking=True, calls THINKING_MODEL (2.5 Pro or 3 Pro), not Flash — Flash has no Thinking."""
    examples = lemma_examples.get(lemma, [])
    if not examples:
        return None
    prompt = build_retry_prompt(lemma, examples)
    actual_model = THINKING_MODEL if use_thinking else model
    config_kwargs: dict = {
        "temperature": 0,
        "max_output_tokens": 16384,
        "response_mime_type": "application/json",
    }
    if "flash" in actual_model:
        config_kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
    try:
        response = client.models.generate_content(
            model=f"models/{actual_model}",
            contents=prompt,
            config=types.GenerateContentConfig(**config_kwargs),
        )
    except Exception:
        return None
    text = (response.text or "").strip()
    if not text:
        return None
    return {"candidates": [{"content": {"parts": [{"text": text}]}}]}


def _parse_one_result(
    key: str,
    response: dict,
    lemma_examples: dict[str, list[str]],
) -> tuple[str, list[dict], str | None]:
    """Parse one batch result line. Returns (lemma, cards, error)."""
    if not key.startswith("lemma:"):
        return "", [], "bad_key"
    lemma = key[6:]

    if "candidates" not in response or not response["candidates"]:
        return lemma, [], "no_candidates"
    parts = response["candidates"][0].get("content", {}).get("parts", [])
    if not parts:
        return lemma, [], "empty_content"
    text = parts[0].get("text", "").strip()

    if text.startswith("```json"):
        text = text[7:]
    if text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        return lemma, [], f"json_error: {e}"

    cards = data.get("cards", [])
    examples = lemma_examples.get(lemma, [])

    for card in cards:
        if not isinstance(card, dict):
            continue
        card["lemma"] = lemma
        card["source"] = "pipeline_b_batch"
        raw = [i for i in card.get("selected_example_indices", []) if isinstance(i, int)]
        # Support both 1-based (prompt) and 0-based (LLM may still return 0-based)
        if not raw:
            idxs = []
        elif 0 in raw and max(raw) < len(examples):
            idxs = [i for i in raw if 0 <= i < len(examples)]
        else:
            idxs = [i - 1 for i in raw if 0 < i <= len(examples)]
        card["examples"] = [examples[j] for j in idxs]
        # No first-example substitution: when indices invalid, leave empty and flag for retry (all 27 ~cards go to retry).
        if not card["examples"] and examples:
            card["examples_fallback"] = True

    return lemma, cards, None


def _validate_card(card: dict, lemma: str) -> list[str]:
    """Return list of validation errors for a card."""
    errs = []
    for k in REQUIRED_CARD_KEYS:
        if k not in card or not (card[k] and str(card[k]).strip()):
            errs.append(f"missing or empty '{k}'")
    if "selected_example_indices" not in card:
        errs.append("missing selected_example_indices")
    return errs


def _cards_lemma_not_in_example(all_cards: list[dict]) -> list[dict]:
    """Cards where at least one example does not contain the lemma (or a valid word form). Flag for review."""
    if _get_word_forms is None or _word_in_text is None:
        return []
    out = []
    for c in all_cards:
        lemma = c.get("lemma", "")
        examples = c.get("examples") or []
        if not examples:
            continue
        forms = _get_word_forms(lemma) - HOMONYM_EXCLUDE_FORMS.get(lemma.lower(), set())
        for i, ex in enumerate(examples):
            if not any(_word_in_text(f, ex) for f in forms):
                out.append({
                    "lemma": lemma,
                    "definition_en": (c.get("definition_en") or "")[:80],
                    "example_index": i + 1,
                    "example_preview": (ex[:100] + "...") if len(ex) > 100 else ex,
                })
                break
    return out


@app.command()
def create(
    limit: int = typer.Option(0, help="Limit number of lemmas (0=all)"),
    max_examples: int = typer.Option(
        50,
        help="Max examples per lemma (50 recommended to avoid MAX_TOKENS)",
    ),
    model: str = typer.Option(MODEL, help="Gemini model"),
):
    """Create batch requests for Pipeline B (one request per lemma)."""
    get_client()  # fail fast if no API key
    BATCH_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    lemma_groups, lemma_examples = load_lemma_groups(
        limit=limit if limit > 0 else None,
        max_examples=max_examples,
    )

    lemmas = lemma_groups["lemma"].tolist()
    n = len(lemmas)
    if n == 0:
        raise SystemExit("No lemmas to process. Check tokens/sentences sample.")
    print(f"Total lemmas: {n} (max_examples={max_examples})")

    requests = []
    for lemma in lemmas:
        examples = lemma_examples[lemma]
        prompt = build_prompt(lemma, examples)
        requests.append({
            "key": f"lemma:{lemma}",
            "request": {
                "model": f"models/{model}",
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "temperature": 0.0,
                    "maxOutputTokens": 16384,
                    "responseMimeType": "application/json",
                },
            },
        })

    with open(REQUESTS_PATH, "w", encoding="utf-8") as f:
        for req in requests:
            f.write(json.dumps(req, ensure_ascii=False) + "\n")

    with open(LEMMA_EXAMPLES_PATH, "w", encoding="utf-8") as f:
        json.dump(lemma_examples, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(requests)} requests to {REQUESTS_PATH}")

    client = get_client()
    print("Uploading batch file...")
    uploaded = client.files.upload(
        file=REQUESTS_PATH,
        config=types.UploadFileConfig(mime_type="application/jsonl"),
    )
    print(f"Uploaded: {uploaded.name}")

    print("Creating batch job...")
    batch_job = client.batches.create(
        model=f"models/{model}",
        src=uploaded.name,
        config=types.CreateBatchJobConfig(
            display_name=f"pipeline_b_{n}_lemmas",
        ),
    )

    batch_info = {
        "batch_name": batch_job.name,
        "model": model,
        "lemmas_count": n,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "uploaded_file": uploaded.name,
    }
    with open(BATCH_INFO_PATH, "w") as f:
        json.dump(batch_info, f, indent=2)

    print("\n✅ Batch created!")
    print(f"   Name: {batch_job.name}")
    print(f"   State: {batch_job.state}")
    print(f"   Lemmas: {n}")
    print("\nRun: uv run python scripts/run_pipeline_b_batch.py status")


@app.command()
def status():
    """Check batch job status."""
    if not BATCH_INFO_PATH.exists():
        print("No batch info. Run 'create' first.")
        raise SystemExit(1)
    with open(BATCH_INFO_PATH) as f:
        info = json.load(f)
    client = get_client()
    job = client.batches.get(name=info["batch_name"])
    print(f"Batch: {job.name}")
    print(f"State: {job.state.name}")
    print(f"Model: {info['model']}")
    print(f"Lemmas: {info['lemmas_count']}")
    if job.state.name == "JOB_STATE_SUCCEEDED":
        print("\n✅ Complete. Run 'download' to get results.")
    elif job.state.name == "JOB_STATE_FAILED":
        print("\n❌ Failed.")
        if hasattr(job, "error"):
            print(f"Error: {job.error}")


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
):
    """Download batch results and write cards_B_batch.json.

    To test new retry logic on same data (no API download): use --from-file.
    Requires: data/experiment/batch_b/results.jsonl and lemma_examples.json.
    Example: uv run python scripts/run_pipeline_b_batch.py download --from-file
    Then optionally: download --from-file --retry-thinking
    """
    if not LEMMA_EXAMPLES_PATH.exists():
        print("Lemma examples file missing. Re-run 'create' or copy lemma_examples.json to batch_b.")
        raise SystemExit(1)
    if not from_file:
        if not BATCH_INFO_PATH.exists():
            print("No batch info. Run 'create' first (or use --from-file to use existing results.jsonl).")
            raise SystemExit(1)
        with open(BATCH_INFO_PATH) as f:
            info = json.load(f)
        client = get_client()
        job = client.batches.get(name=info["batch_name"])
        if job.state.name != "JOB_STATE_SUCCEEDED":
            print(f"Batch not complete. State: {job.state.name}")
            raise SystemExit(1)
        if not hasattr(job, "dest") or not job.dest:
            print("No destination file in batch job.")
            raise SystemExit(1)
        file_name = job.dest.file_name
        content = client.files.download(file=file_name)
        with open(RESULTS_PATH, "wb") as f:
            f.write(content)
        print(f"Downloaded to {RESULTS_PATH}")
    else:
        if not RESULTS_PATH.exists():
            print(f"Missing {RESULTS_PATH}. Run download without --from-file first, or copy results.jsonl.")
            raise SystemExit(1)
        info = {"batch_name": None, "lemmas_count": None}
        print(f"Using existing {RESULTS_PATH} (--from-file).")

    with open(LEMMA_EXAMPLES_PATH, encoding="utf-8") as f:
        lemma_examples = json.load(f)
    if from_file and info["lemmas_count"] is None:
        info["lemmas_count"] = len(lemma_examples)

    client = get_client() if (retry_empty or retry_thinking) else None

    all_cards = []
    errors = []
    validation_errors = []
    success_count = 0
    lemmas_with_zero_cards: list[str] = []
    cards_with_empty_examples: list[dict] = []
    cards_with_examples_fallback: list[dict] = []

    with open(RESULTS_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            key = row.get("key", "")
            resp = row.get("response", {})

            lemma, cards, err = _parse_one_result(key, resp, lemma_examples)
            if err:
                errors.append({"lemma": lemma, "error": err})
            else:
                success_count += 1
                if len(cards) == 0:
                    lemmas_with_zero_cards.append(lemma)
                total_ex = len(lemma_examples.get(lemma, []))
                for c in cards:
                    c["total_lemma_examples"] = total_ex
                    if not c.get("examples"):
                        cards_with_empty_examples.append({
                            "lemma": lemma,
                            "definition_en": (c.get("definition_en") or "")[:100],
                            "selected_example_indices": c.get("selected_example_indices", []),
                            "total_lemma_examples": total_ex,
                        })
                    elif c.get("examples_fallback"):
                        cards_with_examples_fallback.append({
                            "lemma": lemma,
                            "definition_en": (c.get("definition_en") or "")[:60],
                            "selected_example_indices": c.get("selected_example_indices", []),
                            "total_lemma_examples": total_ex,
                        })
                    if not skip_validation:
                        ve = _validate_card(c, lemma)
                        if ve:
                            validation_errors.append({"lemma": lemma, "card": c, "errors": ve})
                all_cards.extend(cards)

    # Retry lemmas that have cards with empty examples or examples_fallback: concatenate
    # a reminder line to the prompt and call Standard API again (new response even at temp 0).
    lemmas_to_retry = set(e["lemma"] for e in cards_with_empty_examples) | set(
        e["lemma"] for e in cards_with_examples_fallback
    )
    retry_success = 0
    retry_failed = 0
    if retry_empty and lemmas_to_retry:
        print(f"\nRetrying {len(lemmas_to_retry)} lemmas (prompt + reminder) via Standard API...")
        for lemma in sorted(lemmas_to_retry):
            resp = _call_standard_api_for_retry(client, lemma, lemma_examples, MODEL)
            if not resp:
                retry_failed += 1
                continue
            _, new_cards, err = _parse_one_result(f"lemma:{lemma}", resp, lemma_examples)
            if err or not new_cards:
                retry_failed += 1
                continue
            if all(c.get("examples") and not c.get("examples_fallback") for c in new_cards):
                all_cards[:] = [c for c in all_cards if c.get("lemma") != lemma]
                total_ex = len(lemma_examples.get(lemma, []))
                for c in new_cards:
                    c["total_lemma_examples"] = total_ex
                all_cards.extend(new_cards)
                retry_success += 1
            else:
                retry_failed += 1
        if retry_success or retry_failed:
            print(f"Retry: {retry_success} lemmas fixed, {retry_failed} still empty/fallback.")

    # Second retry with Thinking (same model, thinking_budget>0) for lemmas still with empty examples
    lemmas_still_empty = {c["lemma"] for c in all_cards if not c.get("examples")}
    retry_thinking_success = 0
    retry_thinking_failed = 0
    if retry_thinking and lemmas_still_empty:
        print(f"\nRetrying {len(lemmas_still_empty)} lemmas with Thinking model ({THINKING_MODEL})...")
        for lemma in sorted(lemmas_still_empty):
            resp = _call_standard_api_for_retry(
                client, lemma, lemma_examples, MODEL, use_thinking=True
            )
            if not resp:
                retry_thinking_failed += 1
                continue
            _, new_cards, err = _parse_one_result(f"lemma:{lemma}", resp, lemma_examples)
            if err or not new_cards:
                retry_thinking_failed += 1
                continue
            if all(c.get("examples") for c in new_cards):
                all_cards[:] = [c for c in all_cards if c.get("lemma") != lemma]
                total_ex = len(lemma_examples.get(lemma, []))
                for c in new_cards:
                    c["total_lemma_examples"] = total_ex
                all_cards.extend(new_cards)
                retry_thinking_success += 1
            else:
                retry_thinking_failed += 1
        if retry_thinking_success or retry_thinking_failed:
            print(f"Thinking retry: {retry_thinking_success} fixed, {retry_thinking_failed} still empty.")

    # Post-check: cards where at least one example does not contain the lemma (or valid form)
    cards_lemma_not_in_example = _cards_lemma_not_in_example(all_cards)

    if validation_errors:
        print(f"\n⚠ Validation issues: {len(validation_errors)} cards")
        for e in validation_errors[:5]:
            print(f"   {e['lemma']}: {e['errors']}")
        if len(validation_errors) > 5:
            print(f"   ... and {len(validation_errors) - 5} more")

    result = {
        "pipeline": "B",
        "source": "batch_api",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "config": {
            "batch_name": info["batch_name"],
            "lemmas_count": info["lemmas_count"],
            "retry_empty": retry_empty,
            "retry_success": retry_success if retry_empty else None,
            "retry_failed": retry_failed if retry_empty else None,
            "retry_thinking": retry_thinking,
            "retry_thinking_success": retry_thinking_success if retry_thinking else None,
            "retry_thinking_failed": retry_thinking_failed if retry_thinking else None,
            "cards_lemma_not_in_example_count": len(cards_lemma_not_in_example),
        },
        "stats": {
            "lemmas_processed": success_count,
            "cards_generated": len(all_cards),
            "errors": len(errors),
            "lemmas_with_zero_cards": len(lemmas_with_zero_cards),
            "validation_issues": len(validation_errors) if not skip_validation else None,
        },
        "cards": all_cards,
        "errors": errors,
        "lemmas_with_zero_cards": lemmas_with_zero_cards,
    }

    OUTPUT_CARDS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_CARDS_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    # Log summary to batch_b for debugging (which lemmas got 0 cards, parse errors, empty examples)
    log_path = BATCH_DIR / "download_log.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "timestamp": result["timestamp"],
                "lemmas_processed": success_count,
                "cards_generated": len(all_cards),
                "errors_count": len(errors),
                "lemmas_with_zero_cards": lemmas_with_zero_cards,
                "cards_with_empty_examples": cards_with_empty_examples,
                "cards_with_examples_fallback": cards_with_examples_fallback,
                "retry_empty": retry_empty,
                "retry_success": retry_success if retry_empty else None,
                "retry_failed": retry_failed if retry_empty else None,
                "retry_thinking": retry_thinking,
                "retry_thinking_success": retry_thinking_success if retry_thinking else None,
                "retry_thinking_failed": retry_thinking_failed if retry_thinking else None,
                "cards_lemma_not_in_example": cards_lemma_not_in_example,
                "errors": errors,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"Log: {log_path}")

    if lemmas_with_zero_cards:
        print(f"\n⚠ Lemmas with 0 cards (LLM returned empty): {len(lemmas_with_zero_cards)} — see download_log.json")
    if cards_with_empty_examples:
        print(f"\n⚠ Cards with empty examples (A3): {len(cards_with_empty_examples)} — see download_log.json 'cards_with_empty_examples'")
    if cards_with_examples_fallback:
        print(f"\n📌 Cards with examples fallback (out-of-range indices): {len(cards_with_examples_fallback)} — see download_log.json 'cards_with_examples_fallback'")
    if cards_lemma_not_in_example:
        print(f"\n⚠ Cards for review (lemma not in example): {len(cards_lemma_not_in_example)} — see download_log.json 'cards_lemma_not_in_example'")

    print(f"\n✅ Done. Parsed {success_count} lemmas, {len(all_cards)} cards, {len(errors)} errors.")
    print(f"Output: {OUTPUT_CARDS_PATH}")


@app.command("list-retry-candidates")
def list_retry_candidates():
    """Parse results.jsonl and list lemmas that would be retried (empty/fallback examples). No API calls."""
    if not RESULTS_PATH.exists():
        print(f"Missing {RESULTS_PATH}. Run batch download first.")
        raise SystemExit(1)
    if not LEMMA_EXAMPLES_PATH.exists():
        print(f"Missing {LEMMA_EXAMPLES_PATH}. Re-run 'create' first.")
        raise SystemExit(1)
    with open(LEMMA_EXAMPLES_PATH, encoding="utf-8") as f:
        lemma_examples = json.load(f)
    lemmas_empty: list[str] = []
    lemmas_fallback: list[str] = []
    with open(RESULTS_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            key = row.get("key", "")
            resp = row.get("response", {})
            if not key.startswith("lemma:"):
                continue
            lemma, cards, err = _parse_one_result(key, resp, lemma_examples)
            if err or not cards:
                continue
            n_ex = len(lemma_examples.get(lemma, []))
            for c in cards:
                if not c.get("examples") and n_ex > 0:
                    lemmas_empty.append(lemma)
                    break
            for c in cards:
                if c.get("examples_fallback"):
                    lemmas_fallback.append(lemma)
                    break
    lemmas_to_retry = sorted(set(lemmas_empty) | set(lemmas_fallback))
    print("=== Retry candidates (0 examples or examples_fallback) ===\n")
    print(f"Lemmas with empty examples (lemma had examples): {len(set(lemmas_empty))}")
    print(f"Lemmas with examples_fallback (out-of-range indices): {len(set(lemmas_fallback))}")
    print(f"Total unique lemmas to retry: {len(lemmas_to_retry)}")
    if lemmas_to_retry:
        print("\nLemmas (first 40):")
        for lemma in lemmas_to_retry[:40]:
            flags = []
            if lemma in set(lemmas_empty):
                flags.append("empty")
            if lemma in set(lemmas_fallback):
                flags.append("fallback")
            print(f"  {lemma} [{', '.join(flags)}]")
        if len(lemmas_to_retry) > 40:
            print(f"  ... and {len(lemmas_to_retry) - 40} more")
        print("\nRun: uv run python scripts/run_pipeline_b_batch.py download")
        print("to retry these via Standard API (prompt + reminder).")


@app.command()
def wait():
    """Wait for batch to complete, then download."""
    if not BATCH_INFO_PATH.exists():
        print("No batch info. Run 'create' first.")
        raise SystemExit(1)
    with open(BATCH_INFO_PATH) as f:
        info = json.load(f)
    client = get_client()
    name = info["batch_name"]
    print(f"Waiting for batch {name}...")
    while True:
        job = client.batches.get(name=name)
        state = job.state.name
        print(f"State: {state}")
        if state == "JOB_STATE_SUCCEEDED":
            print("\n✅ Complete!")
            download()
            break
        if state in ("JOB_STATE_FAILED", "JOB_STATE_CANCELLED"):
            print(f"\n❌ {state}")
            if hasattr(job, "error"):
                print(job.error)
            raise SystemExit(1)
        time.sleep(60)


if __name__ == "__main__":
    app()
