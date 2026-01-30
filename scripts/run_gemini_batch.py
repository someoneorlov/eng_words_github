#!/usr/bin/env python3
"""Run Pipeline v2 using Gemini Batch API (50% cheaper).

Usage:
    uv run python scripts/run_gemini_batch.py --help
    uv run python scripts/run_gemini_batch.py create
    uv run python scripts/run_gemini_batch.py status
    uv run python scripts/run_gemini_batch.py download
    uv run python scripts/run_gemini_batch.py wait

Batch API provides 50% discount:
- gemini-3-flash-preview: $0.25/$1.50 vs $0.50/$3.00
- gemini-2.5-flash: $0.15/$1.25 vs $0.30/$2.50
"""

import json
import os
import time
from pathlib import Path

import pandas as pd
import typer
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

app = typer.Typer(help="Gemini Batch API for Pipeline v2")

# Paths
DATA_DIR = Path("data/experiment")
BATCH_DIR = DATA_DIR / "batch"
SENTENCES_PATH = DATA_DIR / "sentences_sample.parquet"
TOKENS_PATH = DATA_DIR / "tokens_sample.parquet"

# Batch files
STAGE1_REQUESTS_PATH = BATCH_DIR / "stage1_requests.jsonl"
STAGE2_REQUESTS_PATH = BATCH_DIR / "stage2_requests.jsonl"
BATCH_INFO_PATH = BATCH_DIR / "batch_info.json"
RESULTS_PATH = DATA_DIR / "cards_batch.json"

MODEL = "gemini-3-flash-preview"


def get_client() -> genai.Client:
    """Get authenticated Gemini client."""
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not set")
    return genai.Client(api_key=api_key)


def load_lemma_groups(
    skip_stopwords: bool = True,
    max_examples: int = 50,
) -> dict[str, list[tuple[int, str]]]:
    """Load lemma -> [(sentence_id, text), ...] mapping.
    
    Args:
        skip_stopwords: Skip common stop words that have too many examples
        max_examples: Limit examples per lemma to avoid huge prompts
    """
    sentences_df = pd.read_parquet(SENTENCES_PATH)
    tokens_df = pd.read_parquet(TOKENS_PATH)
    
    sentence_lookup = dict(zip(sentences_df["sentence_id"], sentences_df["text"]))
    
    # Skip very common function words and punctuation that cause MAX_TOKENS issues
    STOPWORDS = {
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
        "of", "with", "by", "from", "as", "is", "was", "are", "were", "be",
        "been", "being", "have", "has", "had", "do", "does", "did", "will",
        "would", "could", "should", "may", "might", "must", "shall", "can",
        "it", "its", "this", "that", "these", "those", "he", "she", "they",
        "him", "her", "them", "his", "their", "i", "you", "we", "me", "us",
        "my", "your", "our", "who", "which", "what", "where", "when", "how",
        "not", "no", "so", "if", "then", "than", "more", "very", "just",
        "one", "all", "any", "some", "each", "other", "such", "only", "own",
        # Punctuation
        ",", ".", "!", "?", ";", ":", "-", "'", '"', "...",
    }
    
    groups: dict[str, list[tuple[int, str]]] = {}
    for _, row in tokens_df.iterrows():
        lemma = row["lemma"]
        
        # Skip stopwords if requested
        if skip_stopwords and lemma.lower() in STOPWORDS:
            continue
            
        sent_id = row["sentence_id"]
        text = sentence_lookup.get(sent_id, "")
        if text:
            if lemma not in groups:
                groups[lemma] = []
            groups[lemma].append((sent_id, text))
    
    # Limit examples per lemma
    if max_examples:
        for lemma in groups:
            if len(groups[lemma]) > max_examples:
                groups[lemma] = groups[lemma][:max_examples]
    
    return groups


def build_extraction_prompt(lemma: str, examples: list[tuple[int, str]]) -> str:
    """Build Stage 1 extraction prompt."""
    # Import from our module
    from eng_words.pipeline_v2.meaning_extractor import EXTRACTION_PROMPT
    
    numbered_examples = "\n".join(
        f"{i+1}. (id={sent_id}) {text}"
        for i, (sent_id, text) in enumerate(examples)
    )
    
    return EXTRACTION_PROMPT.format(lemma=lemma, numbered_examples=numbered_examples)


@app.command()
def create(
    limit: int = typer.Option(0, help="Limit number of lemmas (0=all)"),
    model: str = typer.Option(MODEL, help="Gemini model to use"),
    max_examples: int = typer.Option(
        50,
        help="Max examples per lemma (50 recommended to avoid MAX_TOKENS)",
    ),
):
    """Create batch requests for all lemmas."""
    BATCH_DIR.mkdir(parents=True, exist_ok=True)
    
    print("Loading data...")
    groups = load_lemma_groups(max_examples=max_examples)
    
    if limit > 0:
        groups = dict(list(groups.items())[:limit])
    
    print(f"Total lemmas: {len(groups)} (max_examples={max_examples})")
    
    # Build Stage 1 requests (meaning extraction)
    requests = []
    for lemma, examples in groups.items():
        prompt = build_extraction_prompt(lemma, examples)
        
        request = {
            "key": f"extract_{lemma}",
            "request": {
                "model": f"models/{model}",
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "temperature": 0.0,
                    "maxOutputTokens": 16384,
                    "responseMimeType": "application/json",
                },
            }
        }
        requests.append(request)
    
    # Save requests to JSONL
    with open(STAGE1_REQUESTS_PATH, "w") as f:
        for req in requests:
            f.write(json.dumps(req) + "\n")
    
    print(f"Saved {len(requests)} Stage 1 requests to {STAGE1_REQUESTS_PATH}")
    
    # Upload file and create batch
    client = get_client()
    
    print("Uploading batch file...")
    uploaded_file = client.files.upload(
        file=STAGE1_REQUESTS_PATH,
        config=types.UploadFileConfig(mime_type="application/jsonl"),
    )
    print(f"Uploaded: {uploaded_file.name}")
    
    print("Creating batch job...")
    batch_job = client.batches.create(
        model=f"models/{model}",
        src=uploaded_file.name,
        config=types.CreateBatchJobConfig(
            display_name=f"pipeline_v2_stage1_{len(requests)}_lemmas",
        ),
    )
    
    # Save batch info
    batch_info = {
        "stage1_batch_name": batch_job.name,
        "model": model,
        "lemmas_count": len(requests),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "uploaded_file": uploaded_file.name,
    }
    
    with open(BATCH_INFO_PATH, "w") as f:
        json.dump(batch_info, f, indent=2)
    
    print(f"\n✅ Batch created!")
    print(f"   Name: {batch_job.name}")
    print(f"   State: {batch_job.state}")
    print(f"   Lemmas: {len(requests)}")
    print(f"\nRun 'uv run python scripts/run_gemini_batch.py status' to check progress")


@app.command()
def status():
    """Check batch job status."""
    if not BATCH_INFO_PATH.exists():
        print("No batch info found. Run 'create' first.")
        return
    
    with open(BATCH_INFO_PATH) as f:
        batch_info = json.load(f)
    
    client = get_client()
    batch_name = batch_info["stage1_batch_name"]
    
    batch_job = client.batches.get(name=batch_name)
    
    print(f"Batch: {batch_job.name}")
    print(f"State: {batch_job.state}")
    print(f"Model: {batch_info['model']}")
    print(f"Lemmas: {batch_info['lemmas_count']}")
    
    if hasattr(batch_job, "create_time"):
        print(f"Created: {batch_job.create_time}")
    
    if batch_job.state.name == "JOB_STATE_SUCCEEDED":
        print("\n✅ Batch complete! Run 'download' to get results.")
    elif batch_job.state.name == "JOB_STATE_FAILED":
        print("\n❌ Batch failed!")
        if hasattr(batch_job, "error"):
            print(f"Error: {batch_job.error}")


@app.command()
def download():
    """Download batch results and generate cards."""
    if not BATCH_INFO_PATH.exists():
        print("No batch info found. Run 'create' first.")
        return
    
    with open(BATCH_INFO_PATH) as f:
        batch_info = json.load(f)
    
    client = get_client()
    batch_name = batch_info["stage1_batch_name"]
    
    batch_job = client.batches.get(name=batch_name)
    
    if batch_job.state.name != "JOB_STATE_SUCCEEDED":
        print(f"Batch not complete yet. State: {batch_job.state.name}")
        return
    
    print(f"Downloading results from {batch_name}...")
    
    # Get destination file
    if not hasattr(batch_job, "dest") or not batch_job.dest:
        print("No destination file in batch job!")
        return
    
    dest = batch_job.dest
    print(f"Destination: {dest}")
    
    # Download results
    results_file = BATCH_DIR / "stage1_results.jsonl"
    
    # The dest.file_name contains the file resource name
    file_name = dest.file_name
    print(f"File name: {file_name}")
    
    # Download using the file resource
    file_content = client.files.download(file=file_name)
    
    with open(results_file, "wb") as f:
        f.write(file_content)
    
    print(f"Downloaded to {results_file}")
    
    # Parse results
    print("\nParsing extraction results...")
    extraction_results = {}
    success_count = 0
    error_count = 0
    
    with open(results_file) as f:
        for line in f:
            result = json.loads(line)
            key = result.get("key", "")
            
            if key.startswith("extract_"):
                lemma = key[8:]  # Remove "extract_" prefix
                
                response = result.get("response", {})
                if "candidates" in response:
                    try:
                        content = response["candidates"][0]["content"]["parts"][0]["text"]
                        meanings = json.loads(content)
                        extraction_results[lemma] = meanings
                        success_count += 1
                    except (KeyError, json.JSONDecodeError, IndexError) as e:
                        print(f"Parse error for {lemma}: {e}")
                        error_count += 1
                else:
                    error_count += 1
    
    print(f"Parsed: {success_count} lemmas, {error_count} errors")
    
    # Save intermediate results
    with open(BATCH_DIR / "extraction_results.json", "w") as f:
        json.dump(extraction_results, f, indent=2)
    
    print(f"\n✅ Stage 1 complete! Extracted {success_count} lemmas.")
    print("TODO: Run Stage 2 (card generation) - can be done via batch or real-time")


@app.command()
def wait():
    """Wait for batch to complete and download."""
    if not BATCH_INFO_PATH.exists():
        print("No batch info found. Run 'create' first.")
        return
    
    with open(BATCH_INFO_PATH) as f:
        batch_info = json.load(f)
    
    client = get_client()
    batch_name = batch_info["stage1_batch_name"]
    
    print(f"Waiting for batch {batch_name}...")
    
    while True:
        batch_job = client.batches.get(name=batch_name)
        state = batch_job.state.name
        
        print(f"State: {state}")
        
        if state == "JOB_STATE_SUCCEEDED":
            print("\n✅ Batch complete!")
            download()
            break
        elif state == "JOB_STATE_FAILED":
            print("\n❌ Batch failed!")
            if hasattr(batch_job, "error"):
                print(f"Error: {batch_job.error}")
            break
        elif state == "JOB_STATE_CANCELLED":
            print("\n❌ Batch cancelled!")
            break
        
        time.sleep(60)  # Check every minute


@app.command()
def estimate():
    """Estimate batch cost."""
    groups = load_lemma_groups()
    
    total_lemmas = len(groups)
    avg_examples = sum(len(ex) for ex in groups.values()) / total_lemmas
    
    # Rough estimates based on test runs
    # Stage 1: ~500 input tokens, ~200 output tokens per lemma
    # Stage 2: ~300 input tokens, ~150 output tokens per meaning
    # Average ~4 meanings per lemma
    
    stage1_input = 500 * total_lemmas
    stage1_output = 200 * total_lemmas
    
    avg_meanings = 4
    stage2_input = 300 * total_lemmas * avg_meanings
    stage2_output = 150 * total_lemmas * avg_meanings
    
    total_input = stage1_input + stage2_input
    total_output = stage1_output + stage2_output
    
    # Batch pricing for gemini-3-flash-preview
    # Input: $0.25/M, Output: $1.50/M
    batch_input_price = 0.25
    batch_output_price = 1.50
    
    # Standard pricing
    standard_input_price = 0.50
    standard_output_price = 3.00
    
    batch_cost = (total_input * batch_input_price + total_output * batch_output_price) / 1_000_000
    standard_cost = (total_input * standard_input_price + total_output * standard_output_price) / 1_000_000
    
    print(f"=== BATCH COST ESTIMATE ===")
    print(f"Total lemmas: {total_lemmas}")
    print(f"Avg examples/lemma: {avg_examples:.1f}")
    print(f"Avg meanings/lemma: {avg_meanings} (estimated)")
    print()
    print(f"Tokens:")
    print(f"  Input:  {total_input:,}")
    print(f"  Output: {total_output:,}")
    print()
    print(f"Cost comparison (gemini-3-flash-preview):")
    print(f"  Standard API: ${standard_cost:.2f}")
    print(f"  Batch API:    ${batch_cost:.2f}")
    print(f"  Savings:      ${standard_cost - batch_cost:.2f} ({(1 - batch_cost/standard_cost)*100:.0f}%)")


if __name__ == "__main__":
    app()
