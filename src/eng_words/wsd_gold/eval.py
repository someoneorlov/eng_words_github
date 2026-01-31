"""WSD evaluation on gold dataset.

This module provides functions to evaluate WSD predictions against
the gold-labeled dataset.

Supports both fine-grained (synset) and coarse-grained (supersense) evaluation.

Usage:
    from eng_words.wsd_gold.eval import evaluate_wsd_on_gold
    from eng_words.wsd import WordNetSenseBackend

    backend = WordNetSenseBackend()
    results = evaluate_wsd_on_gold("data/wsd_gold/gold_dev.jsonl", backend)
    print(results["metrics"])           # Synset-level
    print(results["supersense_metrics"]) # Supersense-level
"""

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

from nltk.corpus import wordnet as wn

logger = logging.getLogger(__name__)


def get_supersense(synset_id: str) -> str:
    """Get supersense (lexname) from synset ID.

    Args:
        synset_id: WordNet synset ID (e.g., 'bank.n.01')

    Returns:
        Supersense string (e.g., 'noun.artifact') or 'unknown'
    """
    if not synset_id:
        return "unknown"
    try:
        synset = wn.synset(synset_id)
        return synset.lexname()
    except Exception:
        return "unknown"


def load_gold_examples(path: Path | str) -> list[dict[str, Any]]:
    """Load gold examples from JSONL file.

    Args:
        path: Path to JSONL file with gold examples

    Returns:
        List of example dictionaries with gold_synset_id field
    """
    path = Path(path)
    examples = []

    with open(path) as f:
        for line in f:
            data = json.loads(line.strip())
            examples.append(data)

    return examples


def evaluate_single(
    predicted_synset: str | None,
    gold_synset: str,
) -> dict[str, Any]:
    """Evaluate a single WSD prediction.

    Args:
        predicted_synset: Predicted synset ID (or None if no prediction)
        gold_synset: Gold standard synset ID

    Returns:
        Dictionary with evaluation result including:
        - is_correct: Exact synset match
        - is_supersense_correct: Same supersense (coarse-grained)
        - error_type: 'none', 'near_synonym', 'cross_supersense'
    """
    is_correct = predicted_synset == gold_synset

    # Supersense-level evaluation
    pred_supersense = get_supersense(predicted_synset) if predicted_synset else "unknown"
    gold_supersense = get_supersense(gold_synset)
    is_supersense_correct = pred_supersense == gold_supersense

    # Classify error type
    if is_correct:
        error_type = "none"
    elif is_supersense_correct:
        error_type = "near_synonym"  # Same category, wrong specific sense
    else:
        error_type = "cross_supersense"  # Different category entirely

    return {
        "is_correct": is_correct,
        "is_supersense_correct": is_supersense_correct,
        "error_type": error_type,
        "predicted": predicted_synset,
        "gold": gold_synset,
        "predicted_supersense": pred_supersense,
        "gold_supersense": gold_supersense,
    }


def compute_metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute aggregate metrics from evaluation results.

    Args:
        results: List of evaluation results from evaluate_single

    Returns:
        Dictionary with accuracy metrics at both synset and supersense level
    """
    if not results:
        return {
            "accuracy": 0.0,
            "supersense_accuracy": 0.0,
            "total": 0,
            "correct": 0,
            "supersense_correct": 0,
            "error_breakdown": {"none": 0, "near_synonym": 0, "cross_supersense": 0},
        }

    total = len(results)
    correct = sum(1 for r in results if r.get("is_correct", False))
    supersense_correct = sum(1 for r in results if r.get("is_supersense_correct", False))

    # Error type breakdown
    error_breakdown = {
        "none": sum(1 for r in results if r.get("error_type") == "none"),
        "near_synonym": sum(1 for r in results if r.get("error_type") == "near_synonym"),
        "cross_supersense": sum(1 for r in results if r.get("error_type") == "cross_supersense"),
    }

    return {
        "accuracy": correct / total,
        "supersense_accuracy": supersense_correct / total,
        "total": total,
        "correct": correct,
        "supersense_correct": supersense_correct,
        "error_breakdown": error_breakdown,
    }


def compute_metrics_by_segment(
    results: list[dict[str, Any]],
    segment_key: str,
) -> dict[str, dict[str, Any]]:
    """Compute metrics broken down by segment.

    Args:
        results: List of evaluation results with segment info
        segment_key: Key to segment by (e.g., 'pos', 'difficulty')

    Returns:
        Dictionary mapping segment value to metrics
    """
    by_segment: dict[str, list[dict]] = defaultdict(list)

    for result in results:
        segment_value = result.get(segment_key, "unknown")
        by_segment[segment_value].append(result)

    return {
        segment: compute_metrics(segment_results) for segment, segment_results in by_segment.items()
    }


def run_wsd_prediction(
    example: dict[str, Any],
    backend: Any,
) -> dict[str, Any]:
    """Run WSD prediction on a single example.

    Args:
        example: Gold example dictionary
        backend: WSD backend with disambiguate_word method

    Returns:
        Dictionary with predicted_synset and confidence
    """
    context = example["context_window"]
    target = example["target"]
    lemma = target["lemma"]
    pos = target["pos"]

    annotation = backend.disambiguate_word(context, lemma, pos)

    return {
        "predicted_synset": annotation.sense_id,
        "confidence": annotation.confidence,
    }


def classify_example_difficulty(example: dict[str, Any]) -> str:
    """Classify example difficulty based on metadata.

    Args:
        example: Gold example dictionary

    Returns:
        'easy', 'medium', or 'hard'
    """
    metadata = example.get("metadata", {})
    sense_count = metadata.get("wn_sense_count", 1)
    margin = metadata.get("baseline_margin", 1.0)

    if sense_count <= 2 and margin >= 0.3:
        return "easy"
    elif sense_count >= 6 or margin < 0.15:
        return "hard"
    else:
        return "medium"


def evaluate_wsd_on_gold(
    gold_path: Path | str,
    backend: Any,
    limit: int = 0,
    show_progress: bool = True,
) -> dict[str, Any]:
    """Evaluate WSD backend on gold dataset.

    Args:
        gold_path: Path to gold JSONL file
        backend: WSD backend with disambiguate_word method
        limit: Limit number of examples (0 = all)
        show_progress: Show progress bar

    Returns:
        Dictionary with:
        - metrics: Overall accuracy metrics
        - by_pos: Metrics broken down by POS
        - by_difficulty: Metrics broken down by difficulty
        - results: List of individual results
    """
    examples = load_gold_examples(gold_path)

    if limit > 0:
        examples = examples[:limit]

    results = []

    if show_progress:
        try:
            from tqdm import tqdm

            iterator = tqdm(examples, desc="Evaluating WSD")
        except ImportError:
            iterator = examples
    else:
        iterator = examples

    for example in iterator:
        # Run WSD prediction
        prediction = run_wsd_prediction(example, backend)

        # Get gold label
        gold_synset = example.get("gold_synset_id", "")

        # Evaluate
        eval_result = evaluate_single(prediction["predicted_synset"], gold_synset)

        # Add metadata for segmentation
        target = example.get("target", {})
        eval_result["pos"] = target.get("pos", "unknown")
        eval_result["lemma"] = target.get("lemma", "")
        eval_result["difficulty"] = classify_example_difficulty(example)
        eval_result["confidence"] = prediction["confidence"]
        eval_result["example_id"] = example.get("example_id", "")

        # Check if baseline was correct
        metadata = example.get("metadata", {})
        baseline_top1 = metadata.get("baseline_top1", "")
        eval_result["baseline_correct"] = baseline_top1 == gold_synset

        results.append(eval_result)

    # Compute aggregate metrics
    metrics = compute_metrics(results)
    by_pos = compute_metrics_by_segment(results, "pos")
    by_difficulty = compute_metrics_by_segment(results, "difficulty")

    # Compute baseline accuracy (from stored baseline_top1)
    baseline_correct = sum(1 for r in results if r.get("baseline_correct", False))
    baseline_accuracy = baseline_correct / len(results) if results else 0.0

    # Supersense confusion analysis
    supersense_confusion = defaultdict(int)
    for r in results:
        if r.get("error_type") == "cross_supersense":
            key = (r.get("predicted_supersense", ""), r.get("gold_supersense", ""))
            supersense_confusion[key] += 1

    # Sort by frequency
    top_confusion = sorted(supersense_confusion.items(), key=lambda x: -x[1])[:20]

    return {
        "metrics": metrics,
        "baseline_accuracy": baseline_accuracy,
        "by_pos": by_pos,
        "by_difficulty": by_difficulty,
        "supersense_confusion": top_confusion,
        "results": results,
    }
