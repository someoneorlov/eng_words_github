"""Interactive analysis of WSD evaluation results with automatic suspicious case detection.

First identifies suspicious cases automatically (uncertain verdict, low confidence,
lemma not in synset), then allows manual review and classification (TP/FP/TN/FN/LQ).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import typer
from nltk.corpus import wordnet as wn

app = typer.Typer(add_completion=False)


@dataclass
class ErrorRecord:
    """Record of a WSD error found during analysis."""

    index: int
    lemma: str
    sentence_text: str
    assigned_synset: str
    jury_verdict: str
    confidence: float
    error_type: Literal["false_positive", "false_negative", "low_quality"]
    reason: str
    correct_synset: str | None = None
    notes: str = ""
    suspicious_reason: str = ""


def get_synset_info(synset_id: str) -> dict:
    """Get synset information for display."""
    try:
        synset = wn.synset(synset_id)
        return {
            "synset_id": synset_id,
            "definition": synset.definition(),
            "lemmas": [lem.name() for lem in synset.lemmas()],
        }
    except Exception as e:
        return {"synset_id": synset_id, "error": str(e)}


def is_suspicious(sample: dict) -> tuple[bool, str]:
    """Check if a sample is suspicious and return reason."""
    # Uncertain verdict
    if sample["jury_verdict"] == "uncertain":
        return True, "uncertain_verdict"

    # Low confidence
    if sample["confidence"] < 0.7:
        return True, "low_confidence"

    # Check if lemma is in synset lemmas
    try:
        synset = wn.synset(sample["assigned_synset"])
        synset_lemmas = [lem.name().lower() for lem in synset.lemmas()]
        if sample["lemma"].lower() not in synset_lemmas:
            return True, "lemma_not_in_synset"
    except Exception:
        pass

    # Check for common error patterns
    lemma_lower = sample["lemma"].lower()

    # Multi-word lemma but synset is single word
    if "_" in lemma_lower or " " in lemma_lower:
        # This might be a phrasal verb or multi-word expression
        pass

    return False, ""


def display_sample(sample: dict, index: int, total: int, suspicious: bool = False) -> None:
    """Display a single evaluation sample."""
    marker = "⚠️  SUSPICIOUS" if suspicious else ""
    print("\n" + "=" * 80)
    print(f"Sample {index + 1}/{total} {marker}")
    print("=" * 80)
    print(f"Lemma: {sample['lemma']}")
    print("\nSentence:")
    # Highlight lemma in sentence
    sentence = sample["sentence_text"]
    lemma_lower = sample["lemma"].lower()
    sentence_lower = sentence.lower()
    if lemma_lower in sentence_lower:
        pos = sentence_lower.find(lemma_lower)
        highlighted = f"{sentence[:pos]}**{sentence[pos:pos+len(sample['lemma'])]}**{sentence[pos+len(sample['lemma']):]}"
        print(f"  {highlighted}")
    else:
        print(f"  {sentence}")
    print()
    print(f"Assigned synset: {sample['assigned_synset']}")
    synset_info = get_synset_info(sample["assigned_synset"])
    if "error" not in synset_info:
        print(f"  Definition: {synset_info['definition']}")
        print(f"  Lemmas: {', '.join(synset_info['lemmas'][:5])}")
        if len(synset_info["lemmas"]) > 5:
            print(f"  ... and {len(synset_info['lemmas']) - 5} more")
    else:
        print(f"  Error loading synset: {synset_info['error']}")
    print()
    print(f"Jury verdict: {sample['jury_verdict']} (confidence: {sample['confidence']})")
    if "jury_votes" in sample:
        print(f"Jury votes: {sample['jury_votes']}")


@app.command()
def main(
    eval_path: Path = typer.Argument(..., help="Path to evaluation JSON file"),
    output_path: Path = typer.Option(
        None, help="Path to save analysis results (default: eval_path with _analysis suffix)"
    ),
    suspicious_only: bool = typer.Option(False, help="Only show suspicious cases for review"),
):
    """Analyze WSD evaluation results with automatic suspicious case detection."""
    with open(eval_path) as f:
        data = json.load(f)

    results = data["results"]
    total = len(results)

    print(f"Loaded {total} evaluation samples")

    # Find suspicious cases
    suspicious_indices = []
    for i, sample in enumerate(results):
        is_susp, reason = is_suspicious(sample)
        if is_susp:
            suspicious_indices.append((i, reason))

    print(f"\nFound {len(suspicious_indices)} suspicious cases:")
    for idx, reason in suspicious_indices:
        print(
            f"  {idx + 1}. {results[idx]['lemma']} -> {results[idx]['assigned_synset']} ({reason})"
        )

    if suspicious_only and len(suspicious_indices) == 0:
        print("\nNo suspicious cases found. All samples look good!")
        return

    print("\nClassification guide:")
    print("  TP (True Positive): Correct assignment, jury says 'correct'")
    print("  FP (False Positive): Wrong assignment, jury says 'correct'")
    print("  TN (True Negative): Correct rejection, jury says 'incorrect'")
    print("  FN (False Negative): Correct assignment, jury says 'incorrect'")
    print("  LQ (Low Quality): Assignment is technically correct but poor quality")
    print("\nCommands:")
    print("  tp/fp/tn/fn/lq - Classify sample")
    print("  e - Record as error (will ask for details)")
    print("  s - Skip (mark as TP)")
    print("  n - Next suspicious case")
    print("  q - Quit and save")
    print()

    errors = []
    classifications = {"tp": 0, "fp": 0, "tn": 0, "fn": 0, "lq": 0}

    # Review suspicious cases first
    if suspicious_only:
        indices_to_review = [idx for idx, _ in suspicious_indices]
    else:
        indices_to_review = list(range(total))

    i = 0
    while i < len(indices_to_review):
        idx = indices_to_review[i]
        sample = results[idx]
        is_susp, reason = is_suspicious(sample)
        display_sample(sample, idx, total, suspicious=is_susp)
        if is_susp:
            print(f"\n⚠️  Suspicious reason: {reason}")

        cmd = typer.prompt("\nClassification", default="s").strip().lower()

        if cmd == "q":
            break
        elif cmd == "n" and suspicious_only:
            # Skip to next suspicious
            classifications["tp"] += 1
            i += 1
            continue
        elif cmd == "tp":
            classifications["tp"] += 1
            i += 1
        elif cmd == "fp":
            classifications["fp"] += 1
            error = ErrorRecord(
                index=idx,
                lemma=sample["lemma"],
                sentence_text=sample["sentence_text"],
                assigned_synset=sample["assigned_synset"],
                jury_verdict=sample["jury_verdict"],
                confidence=sample["confidence"],
                error_type="false_positive",
                reason=typer.prompt("Why is this a false positive?"),
                correct_synset=typer.prompt("Correct synset (if known)", default="") or None,
                suspicious_reason=reason if is_susp else "",
            )
            errors.append(error)
            i += 1
        elif cmd == "tn":
            classifications["tn"] += 1
            i += 1
        elif cmd == "fn":
            classifications["fn"] += 1
            error = ErrorRecord(
                index=idx,
                lemma=sample["lemma"],
                sentence_text=sample["sentence_text"],
                assigned_synset=sample["assigned_synset"],
                jury_verdict=sample["jury_verdict"],
                confidence=sample["confidence"],
                error_type="false_negative",
                reason=typer.prompt("Why is this a false negative?"),
                correct_synset=sample["assigned_synset"],
                suspicious_reason=reason if is_susp else "",
            )
            errors.append(error)
            i += 1
        elif cmd == "lq":
            classifications["lq"] += 1
            error = ErrorRecord(
                index=idx,
                lemma=sample["lemma"],
                sentence_text=sample["sentence_text"],
                assigned_synset=sample["assigned_synset"],
                jury_verdict=sample["jury_verdict"],
                confidence=sample["confidence"],
                error_type="low_quality",
                reason=typer.prompt("Why is this low quality?"),
                suspicious_reason=reason if is_susp else "",
            )
            errors.append(error)
            i += 1
        elif cmd == "e":
            error_type = typer.prompt("Error type (fp/fn/lq)", default="fp").strip().lower()
            error = ErrorRecord(
                index=idx,
                lemma=sample["lemma"],
                sentence_text=sample["sentence_text"],
                assigned_synset=sample["assigned_synset"],
                jury_verdict=sample["jury_verdict"],
                confidence=sample["confidence"],
                error_type=error_type,
                reason=typer.prompt("Reason"),
                correct_synset=typer.prompt("Correct synset (if known)", default="") or None,
                suspicious_reason=reason if is_susp else "",
            )
            errors.append(error)
            if error_type == "fp":
                classifications["fp"] += 1
            elif error_type == "fn":
                classifications["fn"] += 1
            elif error_type == "lq":
                classifications["lq"] += 1
            i += 1
        elif cmd == "s":
            classifications["tp"] += 1
            i += 1
        else:
            print(f"Unknown command: {cmd}")

    # Save results
    if output_path is None:
        output_path = eval_path.parent / f"{eval_path.stem}_analysis.json"

    output_data = {
        "summary": {
            "total": total,
            "analyzed": sum(classifications.values()),
            "true_positives": classifications["tp"],
            "false_positives": classifications["fp"],
            "true_negatives": classifications["tn"],
            "false_negatives": classifications["fn"],
            "low_quality": classifications["lq"],
            "suspicious_cases": len(suspicious_indices),
        },
        "errors": [asdict(e) for e in errors],
    }

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"\nAnalysis saved to {output_path}")
    print(f"Analyzed: {output_data['summary']['analyzed']}/{total}")
    print(f"Errors found: {len(errors)}")
    print(f"Suspicious cases: {len(suspicious_indices)}")


if __name__ == "__main__":
    app()
