#!/usr/bin/env python3
"""Run LLM labeling on gold examples.

This script runs LLM providers to label WSD examples for the gold dataset.

Usage:
    uv run python scripts/run_gold_labeling.py --help
    uv run python scripts/run_gold_labeling.py --provider openai
    uv run python scripts/run_gold_labeling.py --provider all --dry-run
"""

import json
import logging
import os
from pathlib import Path

import typer
from dotenv import load_dotenv

from eng_words.wsd_gold.models import GoldExample, ModelOutput

# Load environment variables from .env
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

app = typer.Typer(help="Run LLM labeling on gold examples")

# Provider names
PROVIDERS = ["openai", "anthropic", "gemini"]


def load_examples(input_path: Path) -> list[GoldExample]:
    """Load examples from JSONL file."""
    examples = []
    with open(input_path) as f:
        for line in f:
            data = json.loads(line)
            examples.append(GoldExample.from_dict(data))
    return examples


def load_existing_labels(output_path: Path) -> set[str]:
    """Load example IDs that have already been labeled."""
    if not output_path.exists():
        return set()

    labeled_ids = set()
    with open(output_path) as f:
        for line in f:
            data = json.loads(line)
            labeled_ids.add(data.get("example_id", ""))
    return labeled_ids


def save_label(output_path: Path, example_id: str, output: ModelOutput) -> None:
    """Append a label to the output file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    result = {
        "example_id": example_id,
        "output": output.to_dict(),
    }

    with open(output_path, "a") as f:
        f.write(json.dumps(result) + "\n")


def get_provider(name: str, model: str | None = None):
    """Get a provider instance by name.

    Args:
        name: Provider name (openai, anthropic, gemini)
        model: Optional model override
    """
    if name == "openai":
        from eng_words.wsd_gold.providers import OpenAIGoldProvider

        return OpenAIGoldProvider(model=model) if model else OpenAIGoldProvider()
    elif name == "anthropic":
        from eng_words.wsd_gold.providers import AnthropicGoldProvider

        return AnthropicGoldProvider(model=model) if model else AnthropicGoldProvider()
    elif name == "gemini":
        from eng_words.wsd_gold.providers import GeminiGoldProvider

        return GeminiGoldProvider(model=model) if model else GeminiGoldProvider()
    else:
        raise ValueError(f"Unknown provider: {name}")


def check_api_key(provider_name: str) -> bool:
    """Check if API key is available for provider."""
    key_names = {
        "openai": ["OPENAI_API_KEY"],
        "anthropic": ["ANTHROPIC_API_KEY"],
        "gemini": ["GEMINI_API_KEY", "GOOGLE_API_KEY", "GOOGLE_GEMINI_API_KEY"],
    }
    possible_keys = key_names.get(provider_name, [])
    for key_name in possible_keys:
        if os.getenv(key_name):
            return True
    return False


def estimate_cost(examples: list[GoldExample], provider_name: str) -> float:
    """Estimate cost for labeling examples."""
    provider = get_provider(provider_name)
    return provider.estimate_cost(examples)


@app.command()
def main(
    input_path: Path = typer.Option(
        Path("data/wsd_gold/pilot_examples.jsonl"),
        "--input",
        "-i",
        help="Input JSONL file with examples",
    ),
    output_dir: Path = typer.Option(
        Path("data/wsd_gold/labels"),
        "--output-dir",
        "-o",
        help="Output directory for labels",
    ),
    provider: str = typer.Option(
        "openai",
        "--provider",
        "-p",
        help="Provider to use: openai, anthropic, gemini, or all",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Estimate cost without running",
    ),
    limit: int = typer.Option(
        0,
        "--limit",
        "-n",
        help="Limit number of examples (0 = all)",
    ),
    model: str = typer.Option(
        None,
        "--model",
        "-m",
        help="Model override for the provider",
    ),
) -> None:
    """Run LLM labeling on gold examples.

    Supports resume capability - already labeled examples are skipped.
    """
    # Load examples
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        raise typer.Exit(1)

    examples = load_examples(input_path)
    logger.info(f"Loaded {len(examples)} examples from {input_path}")

    if limit > 0:
        examples = examples[:limit]
        logger.info(f"Limited to {len(examples)} examples")

    # Determine providers to run
    providers_to_run = PROVIDERS if provider == "all" else [provider]

    # Dry run: estimate cost
    if dry_run:
        print("\n" + "=" * 50)
        print("💰 COST ESTIMATION (dry run)")
        print("=" * 50)
        total_cost = 0.0
        for prov_name in providers_to_run:
            if not check_api_key(prov_name):
                print(f"  {prov_name:12s}: ⚠️  No API key")
                continue
            cost = estimate_cost(examples, prov_name)
            total_cost += cost
            print(f"  {prov_name:12s}: ${cost:.4f}")
        print("-" * 50)
        print(f"  {'Total':12s}: ${total_cost:.4f}")
        print("=" * 50 + "\n")
        return

    # Run labeling for each provider
    for prov_name in providers_to_run:
        if not check_api_key(prov_name):
            logger.warning(f"Skipping {prov_name}: no API key")
            continue

        output_path = output_dir / f"{prov_name}.jsonl"
        labeled_ids = load_existing_labels(output_path)

        # Filter to unlabeled examples
        to_label = [ex for ex in examples if ex.example_id not in labeled_ids]

        if not to_label:
            logger.info(f"{prov_name}: All {len(examples)} examples already labeled")
            continue

        logger.info(
            f"{prov_name}: Labeling {len(to_label)} examples "
            f"({len(labeled_ids)} already done)"
        )

        # Get provider
        try:
            prov = get_provider(prov_name, model)
        except Exception as e:
            logger.error(f"Failed to initialize {prov_name}: {e}")
            continue

        # Label each example
        success_count = 0
        fail_count = 0
        total_input_tokens = 0
        total_output_tokens = 0

        for i, example in enumerate(to_label, 1):
            try:
                output = prov.label_one(example)
                if output:
                    save_label(output_path, example.example_id, output)
                    success_count += 1
                    total_input_tokens += output.usage.input_tokens
                    total_output_tokens += output.usage.output_tokens

                    if i % 10 == 0:
                        logger.info(
                            f"{prov_name}: {i}/{len(to_label)} done "
                            f"({success_count} success, {fail_count} fail)"
                        )
                else:
                    fail_count += 1
                    logger.warning(f"Failed to label: {example.example_id}")
            except Exception as e:
                fail_count += 1
                logger.error(f"Error labeling {example.example_id}: {e}")

        # Print summary
        print(f"\n✅ {prov_name} complete:")
        print(f"   Success: {success_count}/{len(to_label)}")
        print(f"   Failed:  {fail_count}")
        print(f"   Input tokens:  {total_input_tokens:,}")
        print(f"   Output tokens: {total_output_tokens:,}")
        print(f"   Output: {output_path}")


if __name__ == "__main__":
    app()

