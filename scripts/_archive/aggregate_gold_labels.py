#!/usr/bin/env python3
"""Aggregate gold labels from Anthropic and Gemini, with OpenAI referee.

Strategy:
1. Load Anthropic and Gemini labels
2. If they agree → use that label
3. If they disagree → call OpenAI as referee
4. Final decision: majority vote (2/3) or Anthropic fallback

Usage:
    uv run python scripts/aggregate_gold_labels.py --help
    uv run python scripts/aggregate_gold_labels.py --dry-run
    uv run python scripts/aggregate_gold_labels.py
"""

import json
import logging
from pathlib import Path

import typer
from dotenv import load_dotenv

from eng_words.wsd_gold.models import GoldExample, GoldLabel, LLMUsage, ModelOutput
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

app = typer.Typer(help="Aggregate gold labels with smart referee logic")

EXAMPLES_PATH = Path("data/wsd_gold/examples_all.jsonl")
ANTHROPIC_PATH = Path("data/wsd_gold/labels_full/anthropic.jsonl")
GEMINI_PATH = Path("data/wsd_gold/labels_full/gemini.jsonl")
OPENAI_PATH = Path("data/wsd_gold/labels_full/openai_referee.jsonl")
OUTPUT_PATH = Path("data/wsd_gold/gold_labels_final.jsonl")


def load_examples() -> dict[str, GoldExample]:
    """Load examples as a dict by ID."""
    examples = {}
    with open(EXAMPLES_PATH) as f:
        for line in f:
            ex = GoldExample.from_dict(json.loads(line))
            examples[ex.example_id] = ex
    return examples


def load_labels(path: Path) -> dict[str, ModelOutput]:
    """Load labels from JSONL file."""
    labels = {}
    if not path.exists():
        return labels
    
    with open(path) as f:
        for line in f:
            data = json.loads(line)
            example_id = data["example_id"]
            output_data = data["output"]
            
            # Reconstruct ModelOutput
            usage = LLMUsage(
                input_tokens=output_data.get("usage", {}).get("input_tokens", 0),
                output_tokens=output_data.get("usage", {}).get("output_tokens", 0),
                cached_tokens=output_data.get("usage", {}).get("cached_tokens", 0),
                cost_usd=output_data.get("usage", {}).get("cost_usd", 0),
            )
            
            output = ModelOutput(
                chosen_synset_id=output_data.get("chosen_synset_id", ""),
                confidence=output_data.get("confidence", 0),
                flags=output_data.get("flags", []),
                raw_text=output_data.get("raw_text", ""),
                usage=usage,
            )
            labels[example_id] = output
    
    return labels


def save_openai_label(path: Path, example_id: str, output: ModelOutput) -> None:
    """Save OpenAI referee label."""
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, "a") as f:
        data = {"example_id": example_id, "output": output.to_dict()}
        f.write(json.dumps(data) + "\n")


@app.command()
def main(
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Only analyze disagreements, don't call OpenAI",
    ),
    openai_model: str = typer.Option(
        "gpt-5.2",
        "--openai-model",
        help="OpenAI model for referee",
    ),
) -> None:
    """Aggregate labels with smart referee logic."""
    # Load data
    examples = load_examples()
    anthropic_labels = load_labels(ANTHROPIC_PATH)
    gemini_labels = load_labels(GEMINI_PATH)
    openai_labels = load_labels(OPENAI_PATH)
    
    logger.info(f"Examples: {len(examples)}")
    logger.info(f"Anthropic labels: {len(anthropic_labels)}")
    logger.info(f"Gemini labels: {len(gemini_labels)}")
    logger.info(f"OpenAI referee labels: {len(openai_labels)}")
    
    # Find disagreements
    agreements = 0
    disagreements = []
    missing = 0
    
    for example_id in examples:
        a_label = anthropic_labels.get(example_id)
        g_label = gemini_labels.get(example_id)
        
        if not a_label or not g_label:
            missing += 1
            continue
        
        if a_label.chosen_synset_id == g_label.chosen_synset_id:
            agreements += 1
        else:
            disagreements.append(example_id)
    
    total = agreements + len(disagreements)
    agree_pct = 100 * agreements / total if total > 0 else 0
    
    print("\n" + "=" * 60)
    print("📊 АНАЛИЗ РАЗМЕТКИ")
    print("=" * 60)
    print(f"  Всего примеров: {len(examples)}")
    print(f"  Полных пар: {total}")
    print(f"  Согласие: {agreements} ({agree_pct:.1f}%)")
    print(f"  Разногласия: {len(disagreements)} ({100-agree_pct:.1f}%)")
    print(f"  Недостающие: {missing}")
    
    if dry_run:
        print("\n[DRY RUN] OpenAI referee не вызывается")
        
        # Show some disagreement examples
        print("\n📝 Примеры разногласий (первые 10):")
        for i, ex_id in enumerate(disagreements[:10]):
            ex = examples[ex_id]
            a = anthropic_labels[ex_id]
            g = gemini_labels[ex_id]
            print(f"  {i+1}. {ex.target.lemma} ({ex.target.pos})")
            print(f"      Anthropic: {a.chosen_synset_id or 'none'}")
            print(f"      Gemini: {g.chosen_synset_id or 'none'}")
        return
    
    # Call OpenAI for disagreements that don't have referee yet
    to_referee = [ex_id for ex_id in disagreements if ex_id not in openai_labels]
    
    if to_referee:
        logger.info(f"Calling OpenAI referee for {len(to_referee)} disagreements...")
        
        from eng_words.wsd_gold.providers import OpenAIGoldProvider
        
        openai_provider = OpenAIGoldProvider(model=openai_model)
        
        for i, example_id in enumerate(to_referee, 1):
            ex = examples[example_id]
            
            try:
                result = openai_provider.label_one(ex)
                if result:
                    save_openai_label(OPENAI_PATH, example_id, result)
                    openai_labels[example_id] = result
                    
                    if i % 10 == 0:
                        logger.info(f"OpenAI referee: {i}/{len(to_referee)}")
            except Exception as e:
                logger.error(f"Error for {example_id}: {e}")
    
    # Aggregate all labels
    logger.info("Aggregating final labels...")
    
    results: list[SmartAggregationResult] = []
    final_labels = {}
    
    for example_id in examples:
        a_label = anthropic_labels.get(example_id)
        g_label = gemini_labels.get(example_id)
        
        if not a_label or not g_label:
            continue
        
        o_label = openai_labels.get(example_id) if example_id in disagreements else None
        
        result = smart_aggregate(a_label, g_label, o_label)
        results.append(result)
        final_labels[example_id] = result.label
    
    # Save final labels
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    with open(OUTPUT_PATH, "w") as f:
        for example_id, label in final_labels.items():
            data = {
                "example_id": example_id,
                "synset_id": label.synset_id,
                "confidence": label.confidence,
                "agreement_ratio": label.agreement_ratio,
                "flags": label.flags,
                "needs_referee": label.needs_referee,
            }
            f.write(json.dumps(data) + "\n")
    
    # Print stats
    stats = get_smart_aggregation_stats(results)
    
    print("\n" + "=" * 60)
    print("✅ АГРЕГАЦИЯ ЗАВЕРШЕНА")
    print("=" * 60)
    print(f"  Всего примеров: {stats.total}")
    print(f"  Полное согласие (2/2): {stats.full_agreement} ({100*stats.full_agreement/stats.total:.1f}%)")
    print(f"  Majority vote (2/3): {stats.majority_vote}")
    print(f"  Anthropic fallback: {stats.anthropic_fallback}")
    print(f"  Referee вызовов: {stats.referee_calls}")
    print(f"\n  Сохранено: {OUTPUT_PATH}")


if __name__ == "__main__":
    app()

