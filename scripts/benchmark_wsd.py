#!/usr/bin/env python3
"""
WSD Benchmark Script.

Runs WSD evaluation on Gold Dataset and saves results for comparison.

Usage:
    # Run benchmark with a tag
    uv run python scripts/benchmark_wsd.py --tag "baseline"

    # Compare two benchmarks
    uv run python scripts/benchmark_wsd.py --compare baseline construction_v1

    # Run with limited examples (for testing)
    uv run python scripts/benchmark_wsd.py --tag "test" --limit 100
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import typer

from eng_words.constants import get_gold_dev_path
from eng_words.wsd.wordnet_backend import WordNetSenseBackend
from eng_words.wsd_gold.eval import evaluate_wsd_on_gold

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = typer.Typer(help="WSD Benchmark Tool")

# Default paths
DEFAULT_GOLD_PATH = get_gold_dev_path()
DEFAULT_REPORTS_DIR = Path("reports")


def run_benchmark(
    gold_path: Path,
    limit: int = 0,
    show_progress: bool = True,
) -> dict[str, Any]:
    """
    Run WSD evaluation and return metrics.

    Args:
        gold_path: Path to gold JSONL file
        limit: Limit examples (0 = all)
        show_progress: Show progress bar

    Returns:
        Dictionary with metrics and detailed results
    """
    backend = WordNetSenseBackend()
    results = evaluate_wsd_on_gold(gold_path, backend, limit=limit, show_progress=show_progress)

    return results


def format_metrics(metrics: dict[str, Any]) -> str:
    """Format metrics for display."""
    lines = [
        "=" * 70,
        "📊 WSD BENCHMARK RESULTS",
        "=" * 70,
        "",
        f"📌 SYNSET ACCURACY:     {metrics['accuracy']:.1%} ({metrics['correct']}/{metrics['total']})",
        f"📌 SUPERSENSE ACCURACY: {metrics['supersense_accuracy']:.1%} ({metrics['supersense_correct']}/{metrics['total']})",
        "",
        "📊 ERROR BREAKDOWN:",
        f"   ✅ Correct:          {metrics['error_breakdown']['none']:4}",
        f"   ⚠️  Near-synonym:    {metrics['error_breakdown']['near_synonym']:4} (same category)",
        f"   ❌ Cross-supersense: {metrics['error_breakdown']['cross_supersense']:4} (wrong category)",
        "",
    ]
    return "\n".join(lines)


def compare_metrics(
    old_metrics: dict[str, Any],
    new_metrics: dict[str, Any],
    old_tag: str,
    new_tag: str,
) -> str:
    """Format comparison between two benchmarks."""

    def delta_str(old_val: float, new_val: float, is_pct: bool = True) -> str:
        diff = new_val - old_val
        sign = "+" if diff >= 0 else ""
        if is_pct:
            return f"{sign}{diff:.1%}" if abs(diff) >= 0.001 else "="
        return f"{sign}{diff}" if diff != 0 else "="

    # Handle both 'accuracy' and 'synset_accuracy' keys
    old_synset = old_metrics.get("synset_accuracy", old_metrics.get("accuracy", 0))
    new_synset = new_metrics.get("synset_accuracy", new_metrics.get("accuracy", 0))
    old_ss = old_metrics.get("supersense_accuracy", 0)
    new_ss = new_metrics.get("supersense_accuracy", 0)

    # Format percentages as strings for proper alignment
    old_synset_s = f"{old_synset:.1%}"
    new_synset_s = f"{new_synset:.1%}"
    old_ss_s = f"{old_ss:.1%}"
    new_ss_s = f"{new_ss:.1%}"

    lines = [
        "=" * 70,
        f"📊 COMPARISON: {old_tag} → {new_tag}",
        "=" * 70,
        "",
        f"{'Metric':<25} {'Old':<12} {'New':<12} {'Delta':<10}",
        "-" * 60,
        f"{'Synset Accuracy':<25} {old_synset_s:<12} {new_synset_s:<12} {delta_str(old_synset, new_synset)}",
        f"{'Supersense Accuracy':<25} {old_ss_s:<12} {new_ss_s:<12} {delta_str(old_ss, new_ss)}",
        "",
        "Error Breakdown:",
        f"  {'Correct':<23} {old_metrics['error_breakdown']['none']:<12} {new_metrics['error_breakdown']['none']:<12} {delta_str(old_metrics['error_breakdown']['none'], new_metrics['error_breakdown']['none'], False)}",
        f"  {'Near-synonym':<23} {old_metrics['error_breakdown']['near_synonym']:<12} {new_metrics['error_breakdown']['near_synonym']:<12} {delta_str(old_metrics['error_breakdown']['near_synonym'], new_metrics['error_breakdown']['near_synonym'], False)}",
        f"  {'Cross-supersense':<23} {old_metrics['error_breakdown']['cross_supersense']:<12} {new_metrics['error_breakdown']['cross_supersense']:<12} {delta_str(old_metrics['error_breakdown']['cross_supersense'], new_metrics['error_breakdown']['cross_supersense'], False)}",
        "",
    ]
    return "\n".join(lines)


def find_changed_examples(
    old_results: list[dict],
    new_results: list[dict],
) -> tuple[list[dict], list[dict]]:
    """
    Find examples that changed between benchmarks.

    Returns:
        (improved, regressed) - lists of example dicts
    """
    # Build lookup by example_id
    old_by_id = {r["example_id"]: r for r in old_results}
    new_by_id = {r["example_id"]: r for r in new_results}

    improved = []
    regressed = []

    for example_id in old_by_id:
        if example_id not in new_by_id:
            continue

        old_r = old_by_id[example_id]
        new_r = new_by_id[example_id]

        # Check if correctness changed
        old_correct = old_r.get("is_correct", False)
        new_correct = new_r.get("is_correct", False)

        if not old_correct and new_correct:
            improved.append(
                {
                    "example_id": example_id,
                    "lemma": new_r.get("lemma", ""),
                    "pos": new_r.get("pos", ""),
                    "old_predicted": old_r.get("predicted", ""),
                    "new_predicted": new_r.get("predicted", ""),
                    "gold": new_r.get("gold", ""),
                    "old_error_type": old_r.get("error_type", ""),
                }
            )
        elif old_correct and not new_correct:
            regressed.append(
                {
                    "example_id": example_id,
                    "lemma": new_r.get("lemma", ""),
                    "pos": new_r.get("pos", ""),
                    "old_predicted": old_r.get("predicted", ""),
                    "new_predicted": new_r.get("predicted", ""),
                    "gold": new_r.get("gold", ""),
                    "new_error_type": new_r.get("error_type", ""),
                }
            )

    return improved, regressed


def format_changed_examples(
    improved: list[dict],
    regressed: list[dict],
    limit: int = 10,
) -> str:
    """Format changed examples for display."""
    lines = []

    if improved:
        lines.append(f"✅ IMPROVED ({len(improved)} total, showing {min(limit, len(improved))}):")
        for ex in improved[:limit]:
            lines.append(
                f"   {ex['lemma']:<12} {ex['pos']:<5} {ex['old_error_type']:<15} → correct"
            )
    else:
        lines.append("✅ IMPROVED: none")

    lines.append("")

    if regressed:
        lines.append(
            f"❌ REGRESSED ({len(regressed)} total, showing {min(limit, len(regressed))}):"
        )
        for ex in regressed[:limit]:
            lines.append(f"   {ex['lemma']:<12} {ex['pos']:<5} correct → {ex['new_error_type']}")
    else:
        lines.append("❌ REGRESSED: none")

    return "\n".join(lines)


def save_benchmark(
    tag: str,
    metrics: dict[str, Any],
    results: list[dict],
    reports_dir: Path,
) -> Path:
    """
    Save benchmark results to JSON file.

    Returns:
        Path to saved file
    """
    reports_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).isoformat()

    data = {
        "tag": tag,
        "timestamp": timestamp,
        "metrics": {
            "synset_accuracy": metrics["accuracy"],
            "supersense_accuracy": metrics["supersense_accuracy"],
            "total": metrics["total"],
            "correct": metrics["correct"],
            "supersense_correct": metrics["supersense_correct"],
            "error_breakdown": metrics["error_breakdown"],
        },
        "results": results,
    }

    # Save with tag name
    file_path = reports_dir / f"benchmark_{tag}.json"
    with open(file_path, "w") as f:
        json.dump(data, f, indent=2)

    return file_path


def load_benchmark(tag: str, reports_dir: Path) -> dict[str, Any] | None:
    """Load benchmark results by tag."""
    file_path = reports_dir / f"benchmark_{tag}.json"

    if not file_path.exists():
        return None

    with open(file_path) as f:
        return json.load(f)


@app.command()
def run(
    tag: str = typer.Option(..., "--tag", "-t", help="Tag for this benchmark run"),
    gold_path: Path = typer.Option(
        DEFAULT_GOLD_PATH,
        "--gold",
        "-g",
        help="Path to gold dataset",
    ),
    reports_dir: Path = typer.Option(
        DEFAULT_REPORTS_DIR,
        "--reports",
        "-r",
        help="Directory for benchmark reports",
    ),
    limit: int = typer.Option(0, "--limit", "-n", help="Limit examples (0 = all)"),
    compare_to: str | None = typer.Option(
        None,
        "--compare-to",
        "-c",
        help="Tag of baseline benchmark to compare against",
    ),
) -> None:
    """
    Run WSD benchmark and save results.
    """
    logger.info(f"Running benchmark with tag: {tag}")
    logger.info(f"Gold dataset: {gold_path}")

    # Run evaluation
    results = run_benchmark(gold_path, limit=limit)

    # Display metrics
    print(format_metrics(results["metrics"]))

    # Save results
    saved_path = save_benchmark(
        tag=tag,
        metrics=results["metrics"],
        results=results["results"],
        reports_dir=reports_dir,
    )
    logger.info(f"Saved benchmark to: {saved_path}")

    # Compare if requested
    if compare_to:
        old_data = load_benchmark(compare_to, reports_dir)
        if old_data:
            print(
                compare_metrics(
                    old_data["metrics"],
                    results["metrics"],
                    compare_to,
                    tag,
                )
            )

            improved, regressed = find_changed_examples(
                old_data["results"],
                results["results"],
            )
            print(format_changed_examples(improved, regressed))
        else:
            logger.warning(f"Could not find benchmark with tag: {compare_to}")


@app.command()
def compare(
    old_tag: str = typer.Argument(..., help="Tag of older benchmark"),
    new_tag: str = typer.Argument(..., help="Tag of newer benchmark"),
    reports_dir: Path = typer.Option(
        DEFAULT_REPORTS_DIR,
        "--reports",
        "-r",
        help="Directory for benchmark reports",
    ),
    show_examples: int = typer.Option(
        10,
        "--examples",
        "-e",
        help="Number of changed examples to show",
    ),
) -> None:
    """
    Compare two benchmark results.
    """
    old_data = load_benchmark(old_tag, reports_dir)
    new_data = load_benchmark(new_tag, reports_dir)

    if not old_data:
        typer.echo(f"Error: Could not find benchmark with tag: {old_tag}")
        raise typer.Exit(1)

    if not new_data:
        typer.echo(f"Error: Could not find benchmark with tag: {new_tag}")
        raise typer.Exit(1)

    print(
        compare_metrics(
            old_data["metrics"],
            new_data["metrics"],
            old_tag,
            new_tag,
        )
    )

    improved, regressed = find_changed_examples(
        old_data["results"],
        new_data["results"],
    )
    print(format_changed_examples(improved, regressed, limit=show_examples))


@app.command()
def list_benchmarks(
    reports_dir: Path = typer.Option(
        DEFAULT_REPORTS_DIR,
        "--reports",
        "-r",
        help="Directory for benchmark reports",
    ),
) -> None:
    """
    List all saved benchmarks.
    """
    if not reports_dir.exists():
        typer.echo("No benchmarks found.")
        return

    files = sorted(reports_dir.glob("benchmark_*.json"))

    if not files:
        typer.echo("No benchmarks found.")
        return

    typer.echo(f"{'Tag':<25} {'Synset Acc':<12} {'SS Acc':<12} {'Timestamp'}")
    typer.echo("-" * 70)

    for f in files:
        with open(f) as fp:
            data = json.load(fp)
        tag = data.get("tag", f.stem)
        metrics = data.get("metrics", {})
        timestamp = data.get("timestamp", "")[:10]

        synset_acc = f"{metrics.get('synset_accuracy', 0):.1%}"
        ss_acc = f"{metrics.get('supersense_accuracy', 0):.1%}"

        typer.echo(f"{tag:<25} {synset_acc:<12} {ss_acc:<12} {timestamp}")


if __name__ == "__main__":
    app()
