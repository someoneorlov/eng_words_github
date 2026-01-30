#!/usr/bin/env python3
"""Test different approaches for high-frequency lemmas in Batch API.

Tests:
1. maxOutputTokens=16384, all examples
2. maxOutputTokens=16384, max_examples=50
3. maxOutputTokens=16384, max_examples=100
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

from eng_words.pipeline_v2.meaning_extractor import EXTRACTION_PROMPT

load_dotenv()

app = typer.Typer()

DATA_DIR = Path("data/experiment")
BATCH_DIR = DATA_DIR / "batch" / "test_approaches"
SENTENCES_PATH = DATA_DIR / "sentences_sample.parquet"
TOKENS_PATH = DATA_DIR / "tokens_sample.parquet"

# Test lemmas (high frequency, caused MAX_TOKENS errors)
TEST_LEMMAS = ["now", "well", "up", "about", "after"]

MODEL = "gemini-2.5-flash"


def get_client() -> genai.Client:
    """Get authenticated Gemini client."""
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not set")
    return genai.Client(api_key=api_key)


def load_lemma_examples(lemma: str, max_examples: int | None = None) -> list[tuple[int, str]]:
    """Load examples for a lemma."""
    sentences_df = pd.read_parquet(SENTENCES_PATH)
    tokens_df = pd.read_parquet(TOKENS_PATH)
    
    sentence_lookup = dict(zip(sentences_df["sentence_id"], sentences_df["text"]))
    
    examples = []
    for _, row in tokens_df[tokens_df["lemma"] == lemma].iterrows():
        sent_id = row["sentence_id"]
        text = sentence_lookup.get(sent_id, "")
        if text:
            examples.append((sent_id, text))
    
    if max_examples and len(examples) > max_examples:
        examples = examples[:max_examples]
    
    return examples


def build_request(lemma: str, examples: list[tuple[int, str]], max_output_tokens: int) -> dict:
    """Build batch request for a lemma."""
    numbered_examples = "\n".join(
        f"{i+1}. (id={sent_id}) {text}"
        for i, (sent_id, text) in enumerate(examples)
    )
    
    prompt = EXTRACTION_PROMPT.format(lemma=lemma, numbered_examples=numbered_examples)
    
    return {
        "key": f"test_{lemma}",
        "request": {
            "model": f"models/{MODEL}",
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.0,
                "maxOutputTokens": max_output_tokens,
                "responseMimeType": "application/json",
            },
        }
    }


@app.command()
def test(
    approach: str = typer.Option(..., help="Approach: all_examples, limit_50, limit_100"),
):
    """Test a specific approach."""
    BATCH_DIR.mkdir(parents=True, exist_ok=True)
    
    # Configure approach
    if approach == "all_examples":
        max_examples = None
        max_output_tokens = 16384
        suffix = "all"
    elif approach == "limit_50":
        max_examples = 50
        max_output_tokens = 16384
        suffix = "50"
    elif approach == "limit_100":
        max_examples = 100
        max_output_tokens = 16384
        suffix = "100"
    else:
        raise ValueError(f"Unknown approach: {approach}")
    
    print(f"=== Testing approach: {approach} ===")
    print(f"  max_examples: {max_examples}")
    print(f"  max_output_tokens: {max_output_tokens}")
    print()
    
    # Build requests
    requests = []
    stats = {}
    
    for lemma in TEST_LEMMAS:
        examples = load_lemma_examples(lemma, max_examples)
        stats[lemma] = len(examples)
        
        request = build_request(lemma, examples, max_output_tokens)
        requests.append(request)
        
        print(f"{lemma:10} {len(examples):4} examples")
    
    print(f"\nTotal requests: {len(requests)}")
    
    # Save requests
    requests_file = BATCH_DIR / f"requests_{suffix}.jsonl"
    with open(requests_file, "w") as f:
        for req in requests:
            f.write(json.dumps(req) + "\n")
    
    print(f"Saved to {requests_file}")
    
    # Upload and create batch
    client = get_client()
    
    print("Uploading...")
    uploaded_file = client.files.upload(
        file=requests_file,
        config=types.UploadFileConfig(mime_type="application/jsonl"),
    )
    print(f"Uploaded: {uploaded_file.name}")
    
    print("Creating batch...")
    batch_job = client.batches.create(
        model=f"models/{MODEL}",
        src=uploaded_file.name,
        config=types.CreateBatchJobConfig(
            display_name=f"test_{approach}_{len(requests)}_lemmas",
        ),
    )
    
    # Save batch info
    batch_info = {
        "approach": approach,
        "max_examples": max_examples,
        "max_output_tokens": max_output_tokens,
        "batch_name": batch_job.name,
        "lemmas": TEST_LEMMAS,
        "stats": stats,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    
    info_file = BATCH_DIR / f"batch_info_{suffix}.json"
    with open(info_file, "w") as f:
        json.dump(batch_info, f, indent=2)
    
    print(f"\n✅ Batch created: {batch_job.name}")
    print(f"   State: {batch_job.state}")
    print(f"   Info saved to: {info_file}")
    print(f"\nRun: uv run python scripts/test_batch_approaches.py check --approach {approach}")


@app.command()
def check(approach: str):
    """Check batch status and download results."""
    BATCH_DIR.mkdir(parents=True, exist_ok=True)
    
    # Determine suffix
    if approach == "all_examples":
        suffix = "all"
    elif approach == "limit_50":
        suffix = "50"
    elif approach == "limit_100":
        suffix = "100"
    else:
        raise ValueError(f"Unknown approach: {approach}")
    
    info_file = BATCH_DIR / f"batch_info_{suffix}.json"
    if not info_file.exists():
        print(f"No batch info found for {approach}")
        return
    
    with open(info_file) as f:
        batch_info = json.load(f)
    
    client = get_client()
    batch_name = batch_info["batch_name"]
    
    batch_job = client.batches.get(name=batch_name)
    
    print(f"Batch: {batch_name}")
    print(f"State: {batch_job.state.name}")
    print(f"Approach: {approach}")
    print()
    
    if batch_job.state.name == "JOB_STATE_SUCCEEDED":
        print("✅ Batch complete! Downloading results...")
        
        dest = batch_job.dest
        file_name = dest.file_name
        
        results_file = BATCH_DIR / f"results_{suffix}.jsonl"
        file_content = client.files.download(file=file_name)
        
        with open(results_file, "wb") as f:
            f.write(file_content)
        
        print(f"Downloaded to {results_file}")
        
        # Parse results
        success_count = 0
        error_count = 0
        errors = []
        
        with open(results_file) as f:
            for line in f:
                result = json.loads(line)
                key = result.get("key", "")
                lemma = key.replace("test_", "")
                
                response = result.get("response", {})
                if "candidates" in response:
                    candidate = response["candidates"][0]
                    finish_reason = candidate.get("finishReason", "")
                    content = candidate.get("content", {}).get("parts", [{}])[0]
                    text = content.get("text", "")
                    
                    if finish_reason == "STOP" and text:
                        try:
                            meanings = json.loads(text)
                            if "meanings" in meanings:
                                success_count += 1
                                continue
                        except json.JSONDecodeError:
                            pass
                    
                    error_count += 1
                    errors.append({
                        "lemma": lemma,
                        "finish_reason": finish_reason,
                        "text_length": len(text),
                        "text_preview": text[:100] if text else "NO TEXT",
                    })
                else:
                    error_count += 1
                    errors.append({"lemma": lemma, "error": "no candidates"})
        
        print()
        print(f"Results:")
        print(f"  ✅ Success: {success_count}/{len(TEST_LEMMAS)}")
        print(f"  ❌ Errors: {error_count}/{len(TEST_LEMMAS)}")
        print()
        
        if errors:
            print("Errors:")
            for err in errors:
                print(f"  {err['lemma']}: {err.get('finish_reason', 'unknown')} (text_len={err.get('text_length', 0)})")
        
        # Save summary
        summary = {
            "approach": approach,
            "success_count": success_count,
            "error_count": error_count,
            "errors": errors,
        }
        
        summary_file = BATCH_DIR / f"summary_{suffix}.json"
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)
        
        print(f"\nSummary saved to {summary_file}")
        
    elif batch_job.state.name == "JOB_STATE_FAILED":
        print("❌ Batch failed!")
        if hasattr(batch_job, "error"):
            print(f"Error: {batch_job.error}")
    else:
        print(f"Batch still processing. State: {batch_job.state.name}")


@app.command()
def wait_all():
    """Wait for all 3 batch jobs to complete, then check each."""
    client = get_client()
    
    for approach, suffix in [
        ("all_examples", "all"),
        ("limit_50", "50"),
        ("limit_100", "100"),
    ]:
        info_file = BATCH_DIR / f"batch_info_{suffix}.json"
        if not info_file.exists():
            print(f"Skip {approach}: no batch info")
            continue
        
        with open(info_file) as f:
            batch_info = json.load(f)
        
        batch_name = batch_info["batch_name"]
        print(f"\n=== {approach} ===")
        
        while True:
            batch_job = client.batches.get(name=batch_name)
            state = batch_job.state.name
            
            if state == "JOB_STATE_SUCCEEDED":
                print(f"  ✅ Complete")
                break
            elif state == "JOB_STATE_FAILED":
                print(f"  ❌ Failed")
                break
            elif state == "JOB_STATE_CANCELLED":
                print(f"  ❌ Cancelled")
                break
            
            print(f"  State: {state}")
            time.sleep(30)
    
    print("\nChecking results...")
    for approach in ["all_examples", "limit_50", "limit_100"]:
        check(approach)


@app.command()
def compare():
    """Compare all approaches."""
    BATCH_DIR.mkdir(parents=True, exist_ok=True)
    
    print("=== COMPARISON OF ALL APPROACHES ===\n")
    
    approaches = ["all_examples", "limit_50", "limit_100"]
    results = {}
    
    for approach in approaches:
        suffix = "all" if approach == "all_examples" else ("50" if approach == "limit_50" else "100")
        summary_file = BATCH_DIR / f"summary_{suffix}.json"
        
        if summary_file.exists():
            with open(summary_file) as f:
                summary = json.load(f)
                results[approach] = summary
        else:
            results[approach] = None
    
    # Print comparison
    print(f"{'Approach':<20} {'Success':<10} {'Errors':<10} {'Status'}")
    print("-" * 60)
    
    for approach in approaches:
        if results[approach]:
            r = results[approach]
            success = r["success_count"]
            errors = r["error_count"]
            status = "✅" if errors == 0 else "⚠️"
            print(f"{approach:<20} {success:<10} {errors:<10} {status}")
        else:
            print(f"{approach:<20} {'-':<10} {'-':<10} ⏳ Not tested")
    
    print()
    
    # Best approach
    best = None
    best_score = -1
    
    for approach, result in results.items():
        if result and result["error_count"] == 0:
            if result["success_count"] > best_score:
                best_score = result["success_count"]
                best = approach
    
    if best:
        print(f"✅ Best approach: {best} (100% success)")
    else:
        print("⚠️ No approach achieved 100% success yet")


if __name__ == "__main__":
    app()
