#!/usr/bin/env python3
"""Compare different LLM models for card generation quality.

This script tests multiple LLM models on a sample of examples
and compares their quality and cost.

Usage:
    uv run python scripts/compare_card_gen_models.py --models all
    uv run python scripts/compare_card_gen_models.py --models gpt-5-mini,claude-haiku-4.5
    uv run python scripts/compare_card_gen_models.py --limit 10  # quick test
"""

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import centralized pricing
from eng_words.constants.llm_pricing import (
    estimate_cost,
)

# Model configurations - use centralized pricing
MODELS = {
    "gpt-5-mini": {
        "provider": "openai",
        "model": "gpt-5-mini",
    },
    "gpt-4o-mini": {
        "provider": "openai",
        "model": "gpt-4o-mini",
    },
    "claude-haiku-4.5": {
        "provider": "anthropic",
        "model": "claude-haiku-4-5-20251001",
    },
    "claude-sonnet-4.5": {
        "provider": "anthropic",
        "model": "claude-sonnet-4-20250514",
    },
    "gemini-3-flash": {
        "provider": "gemini",
        "model": "gemini-3-flash-preview",
    },
}


PROMPT_TEMPLATE = """You are helping create Anki flashcards for an English language learner (B1-B2 level).

## Word Information
- Word: "{lemma}" ({pos})
- Semantic category: {supersense}
- WordNet definition: {wn_definition}

## Example sentences from the book "{book_name}"
{contexts_numbered}

## Your Task
1. **Select examples**: Choose 1-2 sentences that BEST illustrate this meaning of the word
2. **Detect errors**: Mark any sentences where the word seems to have a DIFFERENT meaning than the WordNet definition
3. **Simple definition**: Write a clear, simple definition (avoid jargon, suitable for B1-B2 learner)
4. **Translation**: Provide Russian translation for THIS meaning of the word
5. **General example**: Create one simple, memorable example sentence

## Response Format
Return ONLY valid JSON (no markdown, no explanation):
{{
  "selected_indices": [1, 3],
  "excluded_indices": [],
  "simple_definition": "your simple definition here",
  "translation_ru": "перевод на русский",
  "generated_example": "A simple example sentence using the word."
}}
"""


@dataclass
class ModelResult:
    """Result from a single model on a single example."""

    example_id: str
    model_name: str
    success: bool
    selected_indices: list[int]
    excluded_indices: list[int]
    simple_definition: str
    translation_ru: str
    generated_example: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_ms: int
    error: str | None = None


def build_prompt(sample: dict) -> str:
    """Build prompt for a sample."""
    contexts_numbered = "\n".join(f'{i}. "{ctx}"' for i, ctx in enumerate(sample["contexts"], 1))

    return PROMPT_TEMPLATE.format(
        lemma=sample["lemma"],
        pos=sample["pos"],
        supersense=sample["supersense"],
        wn_definition=sample["wn_definition"],
        book_name=sample["book_name"],
        contexts_numbered=contexts_numbered,
    )


def call_openai(prompt: str, model: str) -> tuple[dict, int, int]:
    """Call OpenAI API."""
    from openai import OpenAI

    client = OpenAI()

    start = time.time()

    # Only original gpt-5/gpt-5-mini/gpt-5-nano don't support temperature=0
    # GPT-5.1 and GPT-5.2 support temperature=0
    is_original_gpt5 = model in ("gpt-5", "gpt-5-mini", "gpt-5-nano")

    kwargs = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "response_format": {"type": "json_object"},
    }

    # GPT-5+ models use max_completion_tokens
    # GPT-5-mini needs more tokens due to internal "thinking"
    if model.startswith("gpt-5"):
        kwargs["max_completion_tokens"] = 1024
    else:
        kwargs["max_tokens"] = 512

    # Only original GPT-5 doesn't support temperature=0
    if not is_original_gpt5:
        kwargs["temperature"] = 0

    response = client.chat.completions.create(**kwargs)
    latency_ms = int((time.time() - start) * 1000)

    content = response.choices[0].message.content or "{}"
    input_tokens = response.usage.prompt_tokens
    output_tokens = response.usage.completion_tokens

    result = json.loads(content)
    return result, input_tokens, output_tokens, latency_ms


def call_anthropic(prompt: str, model: str) -> tuple[dict, int, int]:
    """Call Anthropic API."""
    import anthropic

    client = anthropic.Anthropic()

    start = time.time()
    response = client.messages.create(
        model=model,
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}],
    )
    latency_ms = int((time.time() - start) * 1000)

    content = response.content[0].text
    input_tokens = response.usage.input_tokens
    output_tokens = response.usage.output_tokens

    # Parse JSON from response
    # Handle markdown code blocks if present
    if content.strip().startswith("```"):
        lines = content.strip().split("\n")
        content = "\n".join(lines[1:-1])

    result = json.loads(content)
    return result, input_tokens, output_tokens, latency_ms


def call_gemini(prompt: str, model: str) -> tuple[dict, int, int]:
    """Call Gemini API."""
    from google import genai
    from google.genai import types

    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY or GEMINI_API_KEY not set")

    client = genai.Client(api_key=api_key)

    start = time.time()

    # Configure thinking level for gemini-3-flash models
    config_kwargs = {
        "temperature": 0,
        "max_output_tokens": 512,
        "response_mime_type": "application/json",
    }

    # Add thinking level for flash models to reduce token usage
    if "gemini-3-flash" in model:
        config_kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=0)

    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(**config_kwargs),
    )
    latency_ms = int((time.time() - start) * 1000)

    content = response.text or "{}"
    # Estimate tokens (Gemini doesn't always return usage)
    input_tokens = len(prompt) // 4
    output_tokens = len(content) // 4

    if hasattr(response, "usage_metadata") and response.usage_metadata:
        input_tokens = response.usage_metadata.prompt_token_count or input_tokens
        output_tokens = response.usage_metadata.candidates_token_count or output_tokens

    result = json.loads(content)
    return result, input_tokens, output_tokens, latency_ms


def process_sample(sample: dict, model_name: str, model_config: dict) -> ModelResult:
    """Process a single sample with a model."""
    prompt = build_prompt(sample)

    try:
        provider = model_config["provider"]
        model = model_config["model"]

        if provider == "openai":
            result, in_tok, out_tok, latency = call_openai(prompt, model)
        elif provider == "anthropic":
            result, in_tok, out_tok, latency = call_anthropic(prompt, model)
        elif provider == "gemini":
            result, in_tok, out_tok, latency = call_gemini(prompt, model)
        else:
            raise ValueError(f"Unknown provider: {provider}")

        # Calculate cost using centralized pricing
        cost = estimate_cost(
            provider=provider,
            model=model,
            input_tokens=in_tok,
            output_tokens=out_tok,
            use_batch=False,  # Standard API
        )

        return ModelResult(
            example_id=sample["id"],
            model_name=model_name,
            success=True,
            selected_indices=result.get("selected_indices", []),
            excluded_indices=result.get("excluded_indices", []),
            simple_definition=result.get("simple_definition", ""),
            translation_ru=result.get("translation_ru", ""),
            generated_example=result.get("generated_example", ""),
            input_tokens=in_tok,
            output_tokens=out_tok,
            cost_usd=cost,
            latency_ms=latency,
        )

    except Exception as e:
        return ModelResult(
            example_id=sample["id"],
            model_name=model_name,
            success=False,
            selected_indices=[],
            excluded_indices=[],
            simple_definition="",
            translation_ru="",
            generated_example="",
            input_tokens=0,
            output_tokens=0,
            cost_usd=0,
            latency_ms=0,
            error=str(e),
        )


def compare_models(
    samples: list[dict],
    model_names: list[str],
    output_dir: Path,
) -> dict:
    """Compare models on samples."""
    results = {}

    for model_name in model_names:
        if model_name not in MODELS:
            print(f"⚠️  Unknown model: {model_name}, skipping")
            continue

        model_config = MODELS[model_name]
        print(f"\n{'=' * 60}")
        print(f"Testing: {model_name}")
        print(f"{'=' * 60}")

        # Check API key
        provider = model_config["provider"]
        if provider == "openai" and not os.environ.get("OPENAI_API_KEY"):
            print("  ❌ OPENAI_API_KEY not set, skipping")
            continue
        elif provider == "anthropic" and not os.environ.get("ANTHROPIC_API_KEY"):
            print("  ❌ ANTHROPIC_API_KEY not set, skipping")
            continue
        elif provider == "gemini" and not os.environ.get("GOOGLE_API_KEY"):
            print("  ❌ GOOGLE_API_KEY not set, skipping")
            continue

        model_results = []
        total_cost = 0
        total_latency = 0
        success_count = 0

        for i, sample in enumerate(samples, 1):
            print(f"  [{i}/{len(samples)}] {sample['id']}: {sample['lemma']}...", end=" ")

            result = process_sample(sample, model_name, model_config)
            model_results.append(result)

            if result.success:
                success_count += 1
                total_cost += result.cost_usd
                total_latency += result.latency_ms
                print(f"✓ ({result.latency_ms}ms, ${result.cost_usd:.4f})")
            else:
                print(f"✗ {result.error[:50]}...")

            # Small delay to avoid rate limiting
            time.sleep(0.1)

        results[model_name] = {
            "model_name": model_name,
            "model_config": model_config,
            "results": [asdict(r) for r in model_results],
            "summary": {
                "total_samples": len(samples),
                "success_count": success_count,
                "success_rate": success_count / len(samples),
                "total_cost_usd": total_cost,
                "avg_cost_per_sample": total_cost / max(success_count, 1),
                "avg_latency_ms": total_latency / max(success_count, 1),
            },
        }

        print(f"\n  Summary: {success_count}/{len(samples)} success, ${total_cost:.4f} total")

    return results


def generate_report(results: dict, samples: list[dict], output_path: Path) -> None:
    """Generate comparison report."""
    report = {
        "timestamp": datetime.now().isoformat(),
        "total_samples": len(samples),
        "samples_by_difficulty": {
            "easy": len([s for s in samples if s["difficulty"] == "easy"]),
            "hard": len([s for s in samples if s["difficulty"] == "hard"]),
        },
        "models": {},
    }

    for model_name, model_data in results.items():
        summary = model_data["summary"]
        report["models"][model_name] = {
            "success_rate": summary["success_rate"],
            "total_cost_usd": summary["total_cost_usd"],
            "avg_cost_per_sample": summary["avg_cost_per_sample"],
            "avg_latency_ms": summary["avg_latency_ms"],
            "projected_cost_6000": summary["avg_cost_per_sample"] * 6000,
        }

    # Print comparison table
    print(f"\n{'=' * 80}")
    print("MODEL COMPARISON SUMMARY")
    print(f"{'=' * 80}")
    print(f"{'Model':<20} {'Success':<10} {'Cost/50':<10} {'Cost/6000':<12} {'Latency':<10}")
    print("-" * 80)

    for model_name, stats in report["models"].items():
        print(
            f"{model_name:<20} "
            f"{stats['success_rate']:.0%}       "
            f"${stats['total_cost_usd']:.4f}    "
            f"${stats['projected_cost_6000']:.2f}       "
            f"{stats['avg_latency_ms']:.0f}ms"
        )

    print("-" * 80)

    # Save full results
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    report_path = output_path.parent / "comparison_summary.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {output_path}")
    print(f"Summary saved to: {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Compare LLM models for card generation")
    parser.add_argument(
        "--models",
        default="all",
        help="Comma-separated list of models or 'all'",
    )
    parser.add_argument(
        "--input",
        default="data/card_gen_test/sample_50.jsonl",
        help="Input sample file",
    )
    parser.add_argument(
        "--output",
        default="reports/card_gen_model_comparison.json",
        help="Output results file",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of samples (for quick testing)",
    )

    args = parser.parse_args()

    # Load samples
    samples = []
    with open(args.input) as f:
        for line in f:
            samples.append(json.loads(line))

    if args.limit:
        samples = samples[: args.limit]

    print(f"Loaded {len(samples)} samples")

    # Determine models to test
    if args.models == "all":
        model_names = list(MODELS.keys())
    else:
        model_names = [m.strip() for m in args.models.split(",")]

    print(f"Models to test: {model_names}")

    # Run comparison
    results = compare_models(samples, model_names, Path(args.output).parent)

    # Generate report
    generate_report(results, samples, Path(args.output))


if __name__ == "__main__":
    main()
