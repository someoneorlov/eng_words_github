"""Automatic analysis of WSD evaluation to find potential errors.

This script performs fully automatic analysis without interactive review.
Useful for quick screening before manual review.
"""

from __future__ import annotations

import json
from pathlib import Path

import typer
from nltk.corpus import wordnet as wn

app = typer.Typer(add_completion=False)


def analyze_sample(sample: dict, index: int) -> dict | None:
    """Analyze a single sample and return error info if found."""
    lemma = sample["lemma"]
    synset_id = sample["assigned_synset"]
    sentence = sample["sentence_text"]
    verdict = sample["jury_verdict"]
    confidence = sample["confidence"]

    issues = []
    error_type = None

    # Check 1: Uncertain verdict
    if verdict == "uncertain":
        issues.append("uncertain_verdict")
        error_type = "suspicious"

    # Check 2: Low confidence
    if confidence < 0.7:
        issues.append(f"low_confidence_{confidence}")
        error_type = "suspicious"

    # Check 3: Lemma not in synset
    try:
        synset = wn.synset(synset_id)
        synset_lemmas = [lem.name().lower() for lem in synset.lemmas()]
        lemma_lower = lemma.lower()

        if lemma_lower not in synset_lemmas:
            # Check if it's a common inflection
            is_inflection = False
            if lemma_lower.endswith("ing") and lemma_lower[:-3] in synset_lemmas:
                is_inflection = True
            elif lemma_lower.endswith("ed") and lemma_lower[:-2] in synset_lemmas:
                is_inflection = True
            elif lemma_lower.endswith("s") and lemma_lower[:-1] in synset_lemmas:
                is_inflection = True

            if not is_inflection:
                issues.append("lemma_not_in_synset")
                error_type = "error"
    except Exception as e:
        issues.append(f"synset_error: {e}")
        error_type = "error"

    # Check 4: Lemma not in sentence (might be OK for inflections)
    sentence_lower = sentence.lower()
    if lemma_lower not in sentence_lower:
        # Check for common forms
        forms_found = []
        for form in [
            lemma_lower + "s",
            lemma_lower + "ed",
            lemma_lower + "ing",
            lemma_lower + "es",
            lemma_lower + "ly",
        ]:
            if form in sentence_lower:
                forms_found.append(form)

        if not forms_found:
            issues.append("lemma_not_in_sentence")
            error_type = "error"

    # Check 5: Semantic mismatch (heuristic)
    # This is harder to detect automatically, but we can check for obvious cases
    try:
        synset = wn.synset(synset_id)
        definition = synset.definition().lower()

        # Check if definition makes sense in context
        # This is a simple heuristic - can be improved
        if "wind" in definition and "breeze" in definition and "air" == lemma_lower:
            # "air" assigned to breeze synset - might be wrong if context suggests manner
            if (
                "manner" in sentence_lower
                or "appearance" in sentence_lower
                or "jocular" in sentence_lower
            ):
                issues.append("semantic_mismatch_wind_vs_manner")
                error_type = "error"
    except Exception:
        pass

    if issues:
        return {
            "index": index,
            "lemma": lemma,
            "synset_id": synset_id,
            "sentence": sentence,
            "verdict": verdict,
            "confidence": confidence,
            "issues": issues,
            "error_type": error_type,
        }

    return None


@app.command()
def main(
    eval_path: Path = typer.Argument(..., help="Path to evaluation JSON file"),
    output_path: Path = typer.Option(
        None, help="Path to save analysis results (default: eval_path with _auto_analysis suffix)"
    ),
):
    """Automatically analyze WSD evaluation to find potential errors.

    This script performs fully automatic analysis without interactive review.
    Useful for quick screening before manual review with analyze_wsd_evaluation_auto.py.
    """
    with open(eval_path) as f:
        data = json.load(f)

    results = data["results"]
    errors = []
    suspicious = []

    print(f"Analyzing {len(results)} samples...\n")

    for i, sample in enumerate(results):
        analysis = analyze_sample(sample, i)
        if analysis:
            if analysis["error_type"] == "error":
                errors.append(analysis)
            else:
                suspicious.append(analysis)

    print(f"Found {len(errors)} potential ERRORS:")
    print("=" * 80)
    for err in errors:
        print(f"\n{err['index'] + 1}. {err['lemma']} -> {err['synset_id']}")
        print(f"   Issues: {', '.join(err['issues'])}")
        print(f"   Sentence: {err['sentence'][:120]}...")
        print(f"   Verdict: {err['verdict']} (confidence: {err['confidence']})")

    print(f"\n\nFound {len(suspicious)} SUSPICIOUS cases:")
    print("=" * 80)
    for sus in suspicious:
        print(f"\n{sus['index'] + 1}. {sus['lemma']} -> {sus['synset_id']}")
        print(f"   Issues: {', '.join(sus['issues'])}")
        print(f"   Sentence: {sus['sentence'][:120]}...")
        print(f"   Verdict: {sus['verdict']} (confidence: {sus['confidence']})")

    # Save to file
    output = {
        "errors": errors,
        "suspicious": suspicious,
        "summary": {
            "total": len(results),
            "errors": len(errors),
            "suspicious": len(suspicious),
        },
    }

    if output_path is None:
        output_path = eval_path.parent / f"{eval_path.stem}_auto_analysis.json"

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n\nAnalysis saved to {output_path}")


if __name__ == "__main__":
    app()
