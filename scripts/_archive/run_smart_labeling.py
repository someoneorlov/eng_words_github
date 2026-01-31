#!/usr/bin/env python3
"""Run smart LLM labeling with referee logic.

Strategy:
1. Primary judges: Anthropic + Gemini (always called)
2. If they agree → done
3. If they disagree → call OpenAI as referee
4. 2/3 agree → majority vote
5. All different → trust Anthropic

Usage:
    uv run python scripts/run_smart_labeling.py --help
    uv run python scripts/run_smart_labeling.py --dry-run
    uv run python scripts/run_smart_labeling.py --model-tier top
"""

import json
import logging
from pathlib import Path

import typer
from dotenv import load_dotenv

from eng_words.wsd_gold.models import GoldExample
from eng_words.wsd_gold.smart_aggregate import (
    SmartAggregationResult,
    get_smart_aggregation_stats,
    needs_referee,
    smart_aggregate,
)

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

app = typer.Typer(help="Smart LLM labeling with referee logic")

# Model tiers
MODEL_TIERS = {
    "budget": {
        "anthropic": "claude-haiku-4-5-20251001",
        "gemini": "gemini-2.0-flash",
        "openai": "gpt-5-mini",
    },
    "top": {
        "anthropic": "claude-opus-4-5-20251101",
        "gemini": "gemini-3-pro-preview",
        "openai": "gpt-5.2",
    },
}


def load_examples(path: Path) -> list[GoldExample]:
    """Load examples from JSONL file."""
    examples = []
    with open(path) as f:
        for line in f:
            data = json.loads(line)
            examples.append(GoldExample.from_dict(data))
    return examples


def load_existing_results(path: Path) -> dict[str, SmartAggregationResult]:
    """Load existing results for resume capability."""
    if not path.exists():
        return {}

    results = {}
    with open(path) as f:
        for line in f:
            data = json.loads(line)
            example_id = data["example_id"]
            # Reconstruct result (simplified)
            results[example_id] = data
    return results


def save_result(path: Path, example_id: str, result: SmartAggregationResult) -> None:
    """Save a single result to JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "example_id": example_id,
        "synset_id": result.label.synset_id,
        "agreement_type": result.agreement_type,
        "used_referee": result.used_referee,
        "agreement_ratio": result.label.agreement_ratio,
        "flags": result.label.flags,
        "primary_outputs": {k: v.to_dict() for k, v in result.primary_outputs.items()},
    }

    if result.referee_output:
        data["referee_output"] = result.referee_output.to_dict()

    with open(path, "a") as f:
        f.write(json.dumps(data) + "\n")


def get_provider(name: str, model: str):
    """Get a provider instance with specific model."""
    if name == "anthropic":
        from eng_words.wsd_gold.providers import AnthropicGoldProvider

        return AnthropicGoldProvider(model=model)
    elif name == "gemini":
        from eng_words.wsd_gold.providers import GeminiGoldProvider

        return GeminiGoldProvider(model=model)
    elif name == "openai":
        from eng_words.wsd_gold.providers import OpenAIGoldProvider

        return OpenAIGoldProvider(model=model)
    else:
        raise ValueError(f"Unknown provider: {name}")


@app.command()
def main(
    input_path: Path = typer.Option(
        Path("data/wsd_gold/pilot_examples.jsonl"),
        "--input",
        "-i",
        help="Input JSONL file with examples",
    ),
    output_path: Path = typer.Option(
        Path("data/wsd_gold/smart_labels.jsonl"),
        "--output",
        "-o",
        help="Output JSONL file",
    ),
    model_tier: str = typer.Option(
        "top",
        "--model-tier",
        "-m",
        help="Model tier: 'budget' or 'top'",
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
) -> None:
    """Run smart LLM labeling with referee logic.

    Primary: Anthropic + Gemini (always)
    Referee: OpenAI (only when primary disagree)
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

    # Get models
    if model_tier not in MODEL_TIERS:
        logger.error(f"Unknown model tier: {model_tier}")
        raise typer.Exit(1)

    models = MODEL_TIERS[model_tier]
    logger.info(f"Model tier: {model_tier}")
    logger.info(f"  Anthropic: {models['anthropic']}")
    logger.info(f"  Gemini: {models['gemini']}")
    logger.info(f"  OpenAI (referee): {models['openai']}")

    # Dry run: estimate cost
    if dry_run:
        n = len(examples)
        referee_rate = 0.20  # Estimated from pilot

        # Rough cost estimate (per 1M tokens)
        if model_tier == "top":
            primary_cost = n * 2200 * (2.50 + 1.25) / 1_000_000
            referee_cost = n * referee_rate * 2200 * 1.75 / 1_000_000
        else:
            primary_cost = n * 2200 * (0.50 + 0.10) / 1_000_000
            referee_cost = n * referee_rate * 2200 * 0.25 / 1_000_000

        print("\n" + "=" * 50)
        print("💰 COST ESTIMATION (dry run)")
        print("=" * 50)
        print(f"  Examples: {n}")
        print(f"  Primary (Anthropic + Gemini): ${primary_cost:.2f}")
        print(f"  Referee (~{int(n * referee_rate)} calls): ${referee_cost:.2f}")
        print(f"  TOTAL: ${primary_cost + referee_cost:.2f}")
        print("=" * 50 + "\n")
        return

    # Initialize providers
    anthropic_prov = get_provider("anthropic", models["anthropic"])
    gemini_prov = get_provider("gemini", models["gemini"])
    openai_prov = get_provider("openai", models["openai"])

    # Load existing results for resume
    existing = load_existing_results(output_path)
    to_process = [ex for ex in examples if ex.example_id not in existing]

    if not to_process:
        logger.info("All examples already processed")
        return

    logger.info(f"Processing {len(to_process)} examples ({len(existing)} already done)")

    # Process examples
    results: list[SmartAggregationResult] = []
    total_cost = 0.0

    for i, example in enumerate(to_process, 1):
        try:
            # Primary judges
            anthropic_out = anthropic_prov.label_one(example)
            gemini_out = gemini_prov.label_one(example)

            if anthropic_out is None or gemini_out is None:
                logger.warning(f"Primary judge failed for {example.example_id}")
                continue

            total_cost += anthropic_out.usage.cost_usd + gemini_out.usage.cost_usd

            # Check if referee needed
            if needs_referee(anthropic_out, gemini_out):
                openai_out = openai_prov.label_one(example)
                if openai_out:
                    total_cost += openai_out.usage.cost_usd
                result = smart_aggregate(anthropic_out, gemini_out, openai_out)
            else:
                result = smart_aggregate(anthropic_out, gemini_out)

            results.append(result)
            save_result(output_path, example.example_id, result)

            if i % 10 == 0:
                stats = get_smart_aggregation_stats(results)
                logger.info(
                    f"Progress: {i}/{len(to_process)} | "
                    f"Referee calls: {stats.referee_calls} ({100*stats.referee_rate:.0f}%) | "
                    f"Cost: ${total_cost:.2f}"
                )

        except Exception as e:
            logger.error(f"Error processing {example.example_id}: {e}")

    # Final stats
    stats = get_smart_aggregation_stats(results)

    print("\n" + "=" * 60)
    print("✅ SMART LABELING COMPLETE")
    print("=" * 60)
    print(f"  Total processed: {len(results)}")
    print(f"  Full agreement: {stats.full_agreement} ({100*stats.full_agreement/stats.total:.0f}%)")
    print(f"  Majority vote: {stats.majority_vote}")
    print(f"  Anthropic fallback: {stats.anthropic_fallback}")
    print(f"  Referee calls: {stats.referee_calls} ({100*stats.referee_rate:.0f}%)")
    print(f"  Total cost: ${total_cost:.2f}")
    print(f"  Output: {output_path}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    app()
