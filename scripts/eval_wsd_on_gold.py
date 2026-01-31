#!/usr/bin/env python3
"""Evaluate WSD backend on gold dataset.

This script evaluates the current WSD backend (WordNetSenseBackend)
against the gold-labeled dataset and produces a detailed report.

Usage:
    uv run python scripts/eval_wsd_on_gold.py
    uv run python scripts/eval_wsd_on_gold.py --gold-path data/wsd_gold/gold_dev.jsonl
    uv run python scripts/eval_wsd_on_gold.py --limit 100

Output:
    - Console: Summary metrics
    - File: reports/wsd_eval_report.json (detailed results)
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import typer

from eng_words.constants import get_gold_dev_path
from eng_words.wsd import WordNetSenseBackend
from eng_words.wsd_gold.eval import evaluate_wsd_on_gold

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

app = typer.Typer(help="Evaluate WSD on gold dataset")

DEFAULT_GOLD_PATH = get_gold_dev_path()
REPORTS_DIR = Path("reports")


@app.command()
def main(
    gold_path: Path = typer.Option(
        DEFAULT_GOLD_PATH,
        "--gold-path",
        "-g",
        help="Path to gold JSONL file",
    ),
    limit: int = typer.Option(
        0,
        "--limit",
        "-n",
        help="Limit number of examples (0 = all)",
    ),
    output: Path = typer.Option(
        REPORTS_DIR / "wsd_eval_report.json",
        "--output",
        "-o",
        help="Output path for detailed report",
    ),
    show_errors: bool = typer.Option(
        False,
        "--show-errors",
        help="Show individual error examples",
    ),
    top_errors: int = typer.Option(
        10,
        "--top-errors",
        help="Number of error examples to show",
    ),
) -> None:
    """Evaluate WSD backend on gold dataset."""
    if not gold_path.exists():
        logger.error(f"Gold file not found: {gold_path}")
        raise typer.Exit(1)

    # Initialize WSD backend
    logger.info("Initializing WSD backend...")
    backend = WordNetSenseBackend()

    # Run evaluation
    logger.info(f"Evaluating on {gold_path}...")
    results = evaluate_wsd_on_gold(
        gold_path=gold_path,
        backend=backend,
        limit=limit,
        show_progress=True,
    )

    # Print summary
    metrics = results["metrics"]
    by_pos = results["by_pos"]
    by_difficulty = results["by_difficulty"]
    baseline_accuracy = results["baseline_accuracy"]

    print("\n" + "=" * 60)
    print("📊 WSD EVALUATION RESULTS")
    print("=" * 60)
    print(f"\n  Gold dataset: {gold_path}")
    print(f"  Total examples: {metrics['total']}")
    print(f"\n  Overall Accuracy: {metrics['accuracy']:.1%}")
    print(f"  Baseline Accuracy: {baseline_accuracy:.1%}")
    improvement = (metrics["accuracy"] - baseline_accuracy) * 100
    print(f"  Improvement: {improvement:+.1f} p.p.")

    print("\n📍 By POS:")
    for pos, pos_metrics in sorted(by_pos.items()):
        print(f"    {pos:8s}: {pos_metrics['accuracy']:5.1%} ({pos_metrics['total']:4d} examples)")

    print("\n📈 By Difficulty:")
    for diff, diff_metrics in sorted(by_difficulty.items()):
        print(
            f"    {diff:8s}: {diff_metrics['accuracy']:5.1%} ({diff_metrics['total']:4d} examples)"
        )

    # Show error examples
    if show_errors:
        errors = [r for r in results["results"] if not r["is_correct"]]
        print(f"\n❌ Error Examples (first {top_errors}):")
        for i, err in enumerate(errors[:top_errors], 1):
            print(f"  {i}. {err['lemma']} ({err['pos']})")
            print(f"      Predicted: {err['predicted']}")
            print(f"      Gold:      {err['gold']}")

    # Save detailed report
    output.parent.mkdir(parents=True, exist_ok=True)

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "gold_path": str(gold_path),
        "total_examples": metrics["total"],
        "metrics": {
            "accuracy": metrics["accuracy"],
            "baseline_accuracy": baseline_accuracy,
            "improvement_pp": improvement,
        },
        "by_pos": {k: v["accuracy"] for k, v in by_pos.items()},
        "by_difficulty": {k: v["accuracy"] for k, v in by_difficulty.items()},
        "error_count": metrics["total"] - metrics["correct"],
        "errors": [
            {
                "example_id": r["example_id"],
                "lemma": r["lemma"],
                "pos": r["pos"],
                "predicted": r["predicted"],
                "gold": r["gold"],
            }
            for r in results["results"]
            if not r["is_correct"]
        ],
    }

    with open(output, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n  Report saved: {output}")
    print("=" * 60)


if __name__ == "__main__":
    app()
