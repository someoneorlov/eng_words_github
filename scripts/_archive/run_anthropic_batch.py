#!/usr/bin/env python3
"""Run Anthropic labeling using Batch API (50% cheaper).

Usage:
    uv run python scripts/run_anthropic_batch.py --help
    uv run python scripts/run_anthropic_batch.py create
    uv run python scripts/run_anthropic_batch.py status
    uv run python scripts/run_anthropic_batch.py download
"""

import json
import os
import time
from pathlib import Path

import anthropic
import typer
from dotenv import load_dotenv

from eng_words.wsd_gold.models import GoldExample, LLMUsage, ModelOutput
from eng_words.wsd_gold.providers.prompts import (
    build_gold_labeling_prompt,
    parse_gold_label_response,
)

load_dotenv()

app = typer.Typer(help="Anthropic Batch API for gold labeling")

INPUT_PATH = Path("data/wsd_gold/examples_all.jsonl")
OUTPUT_PATH = Path("data/wsd_gold/labels_full/anthropic.jsonl")
BATCH_REQUEST_PATH = Path("data/wsd_gold/batch_requests.jsonl")
BATCH_ID_PATH = Path("data/wsd_gold/batch_id.txt")

MODEL = "claude-opus-4-5-20251101"


def load_examples() -> list[GoldExample]:
    """Load all examples."""
    examples = []
    with open(INPUT_PATH) as f:
        for line in f:
            examples.append(GoldExample.from_dict(json.loads(line)))
    return examples


def load_existing_ids() -> set[str]:
    """Load already labeled example IDs."""
    if not OUTPUT_PATH.exists():
        return set()
    
    ids = set()
    with open(OUTPUT_PATH) as f:
        for line in f:
            data = json.loads(line)
            ids.add(data.get("example_id", ""))
    return ids


@app.command()
def create():
    """Create a batch request for remaining examples."""
    examples = load_examples()
    existing_ids = load_existing_ids()
    
    # Filter to unlabeled examples
    to_label = [ex for ex in examples if ex.example_id not in existing_ids]
    
    print(f"Total examples: {len(examples)}")
    print(f"Already labeled: {len(existing_ids)}")
    print(f"To label: {len(to_label)}")
    
    if not to_label:
        print("All examples already labeled!")
        return
    
    # Build batch requests
    requests = []
    for ex in to_label:
        prompt = build_gold_labeling_prompt(ex)
        
        # Sanitize custom_id: only alphanumeric, underscore, hyphen allowed
        # book:american_tragedy_wsd|sent:12008|tok:1 -> book_american_tragedy_wsd_sent_12008_tok_1
        safe_id = ex.example_id.replace(":", "_").replace("|", "_")
        
        request = {
            "custom_id": safe_id,
            "params": {
                "model": MODEL,
                "max_tokens": 512,
                "messages": [
                    {"role": "user", "content": prompt}
                ]
            }
        }
        requests.append(request)
    
    # Save requests to file
    BATCH_REQUEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(BATCH_REQUEST_PATH, "w") as f:
        for req in requests:
            f.write(json.dumps(req) + "\n")
    
    print(f"Saved {len(requests)} requests to {BATCH_REQUEST_PATH}")
    
    # Create batch
    client = anthropic.Anthropic()
    
    print("Creating batch...")
    batch = client.messages.batches.create(requests=requests)
    
    # Save batch ID
    with open(BATCH_ID_PATH, "w") as f:
        f.write(batch.id)
    
    print(f"✅ Batch created!")
    print(f"   ID: {batch.id}")
    print(f"   Status: {batch.processing_status}")
    print(f"   Requests: {len(requests)}")
    print(f"\nRun 'uv run python scripts/run_anthropic_batch.py status' to check progress")


@app.command()
def status():
    """Check batch status."""
    if not BATCH_ID_PATH.exists():
        print("No batch ID found. Run 'create' first.")
        return
    
    batch_id = BATCH_ID_PATH.read_text().strip()
    
    client = anthropic.Anthropic()
    batch = client.messages.batches.retrieve(batch_id)
    
    print(f"Batch ID: {batch.id}")
    print(f"Status: {batch.processing_status}")
    print(f"Created: {batch.created_at}")
    
    if hasattr(batch, 'request_counts'):
        counts = batch.request_counts
        print(f"Requests:")
        print(f"  Processing: {counts.processing}")
        print(f"  Succeeded: {counts.succeeded}")
        print(f"  Errored: {counts.errored}")
        print(f"  Canceled: {counts.canceled}")
        print(f"  Expired: {counts.expired}")
    
    if batch.processing_status == "ended":
        print("\n✅ Batch complete! Run 'download' to get results.")


@app.command()
def download():
    """Download batch results."""
    if not BATCH_ID_PATH.exists():
        print("No batch ID found. Run 'create' first.")
        return
    
    batch_id = BATCH_ID_PATH.read_text().strip()
    
    client = anthropic.Anthropic()
    batch = client.messages.batches.retrieve(batch_id)
    
    if batch.processing_status != "ended":
        print(f"Batch not complete yet. Status: {batch.processing_status}")
        return
    
    print(f"Downloading results from batch {batch_id}...")
    
    # Stream results
    success_count = 0
    error_count = 0
    
    # Load examples map once
    examples = load_examples()
    ex_map = {ex.example_id: ex for ex in examples}
    # Also map sanitized IDs
    safe_id_map = {ex.example_id.replace(":", "_").replace("|", "_"): ex.example_id for ex in examples}
    
    with open(OUTPUT_PATH, "a") as out_f:
        for result in client.messages.batches.results(batch_id):
            safe_id = result.custom_id
            # Convert back to original ID
            example_id = safe_id_map.get(safe_id, safe_id)
            
            if result.result.type == "succeeded":
                message = result.result.message
                raw_text = message.content[0].text if message.content else ""
                
                # Parse response
                usage = LLMUsage(
                    input_tokens=message.usage.input_tokens,
                    output_tokens=message.usage.output_tokens,
                    cached_tokens=0,
                    cost_usd=0,  # Batch pricing calculated separately
                )
                
                # Get candidates from example
                ex = ex_map.get(example_id)
                
                if ex:
                    candidates = ex.get_candidate_ids()
                    output = parse_gold_label_response(raw_text, candidates, usage)
                    
                    if output:
                        result_data = {
                            "example_id": example_id,
                            "output": output.to_dict(),
                        }
                        out_f.write(json.dumps(result_data) + "\n")
                        success_count += 1
                    else:
                        error_count += 1
                else:
                    error_count += 1
            else:
                error_count += 1
                print(f"Error for {example_id}: {result.result.type}")
    
    print(f"\n✅ Downloaded {success_count} results")
    print(f"   Errors: {error_count}")
    print(f"   Saved to: {OUTPUT_PATH}")
    
    # Final count
    total = sum(1 for _ in open(OUTPUT_PATH))
    print(f"\nTotal labeled: {total}/3000")


@app.command()
def wait():
    """Wait for batch to complete and download."""
    if not BATCH_ID_PATH.exists():
        print("No batch ID found. Run 'create' first.")
        return
    
    batch_id = BATCH_ID_PATH.read_text().strip()
    client = anthropic.Anthropic()
    
    print(f"Waiting for batch {batch_id}...")
    
    while True:
        batch = client.messages.batches.retrieve(batch_id)
        status = batch.processing_status
        
        if hasattr(batch, 'request_counts'):
            counts = batch.request_counts
            print(f"Status: {status} | Succeeded: {counts.succeeded} | Processing: {counts.processing}")
        else:
            print(f"Status: {status}")
        
        if status == "ended":
            print("\n✅ Batch complete!")
            download()
            break
        
        time.sleep(30)


if __name__ == "__main__":
    app()

