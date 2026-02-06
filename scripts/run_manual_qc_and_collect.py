#!/usr/bin/env python3
"""Run manual QC pipeline per PIPELINE_B_MANUAL_QC_INSTRUCTIONS and emit a single JSON with run results.

Evaluates QC gate and regression 49 in-process; expects report scripts to have been run (or run them first).
Writes: data/experiment/pipeline_b_run_result.json with PASS/FAIL, paths, and deliverables checklist.

Usage (from project root):
    # First generate reports (optional; script will still evaluate gate + regression):
    #   uv run python scripts/run_quality_investigation.py --gate --cards data/experiment/cards_B_batch_2.json --output data/experiment/qc_gate_report.md
    #   uv run python scripts/run_regression_49.py --cards data/experiment/cards_B_batch_2.json --output data/experiment/regression_49_report.md
    #   uv run python scripts/run_quality_investigation.py --cards data/experiment/cards_B_batch_2.json --output data/experiment/investigation_report.md
    #   uv run python scripts/check_quality_b_batch.py --input data/experiment/cards_B_batch_2.json --sample 49 --output data/experiment/quality_report_B_batch.md
    uv run python scripts/run_manual_qc_and_collect.py
    uv run python scripts/run_manual_qc_and_collect.py --cards data/experiment/cards_B_batch_2.json --output data/experiment/pipeline_b_run_result.json
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "experiment"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main() -> int:
    import argparse
    p = argparse.ArgumentParser(description="Collect Pipeline B manual QC result into JSON")
    p.add_argument("--cards", type=Path, default=DATA_DIR / "cards_B_batch_2.json", help="Cards JSON path")
    p.add_argument("--output", type=Path, default=DATA_DIR / "pipeline_b_run_result.json", help="Output JSON path")
    args = p.parse_args()

    cards_path = args.cards if args.cards.is_absolute() else PROJECT_ROOT / args.cards
    output_path = args.output if args.output.is_absolute() else PROJECT_ROOT / args.output

    if not cards_path.exists():
        print(f"Cards file not found: {cards_path}", file=sys.stderr)
        return 1

    result = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "cards_path": str(cards_path),
        "gate": None,
        "regression_49": None,
        "reports": {
            "qc_gate": str(DATA_DIR / "qc_gate_report.md"),
            "regression_49": str(DATA_DIR / "regression_49_report.md"),
            "investigation": str(DATA_DIR / "investigation_report.md"),
            "quality_sample": str(DATA_DIR / "quality_report_B_batch.md"),
        },
        "deliverables": {
            "code": [
                "contract invariants: src/eng_words/word_family/contract.py",
                "QCPolicy strict/relaxed: src/eng_words/word_family/qc_types.py",
                "normalize_for_matching + contractions: src/eng_words/text_norm.py",
                "headword resolution + validation: src/eng_words/word_family/headword.py",
                "mode routing (word/phrasal/mwe): src/eng_words/word_family/batch_io.py, batch_schemas.py",
            ],
            "reports": [
                "investigation_report.md (auto)",
                "qc_gate_report.md (QC-gate PASS/FAIL)",
                "regression_49_report.md (49/49 table)",
                "quality_report_B_batch.md (sample for manual review)",
            ],
            "docs": [
                "Роли Stage1 vs LLM: docs/PIPELINE_B_CONTRACTS_AND_RUN.md",
                "Mode contracts: docs/PIPELINE_B_CONTRACTS_AND_RUN.md",
                "Candidate extraction policy (precision-first): docs/PIPELINE_B_CONTRACTS_AND_RUN.md",
            ],
            "commands": [
                "word deck: run_pipeline_b_batch.py create / download [--run-gate]",
                "phrasal deck: BatchConfig(mode=phrasal, top_k=..., candidates_path=...) + render_requests / create_batch / download_batch",
            ],
        },
    }

    # QC gate (in-process)
    try:
        from eng_words.word_family.qc_gate import load_result_and_evaluate_gate
        passed, summary, message = load_result_and_evaluate_gate(cards_path)
        result["gate"] = {"passed": passed, "message": message, "summary": summary}
    except Exception as e:
        result["gate"] = {"passed": False, "message": str(e), "summary": {}}

    # Regression 49 (in-process)
    try:
        from eng_words.word_family.regression import load_result_and_evaluate_regression
        r = load_result_and_evaluate_regression(cards_path)
        result["regression_49"] = {
            "passed": r.passed,
            "total_cards": r.total_cards,
            "valid_schema_rate": r.valid_schema_rate,
            "lemma_headword_in_example_rate": r.lemma_headword_in_example_rate,
            "pos_consistency_rate": r.pos_consistency_rate,
            "major_or_invalid_rate": r.major_or_invalid_rate,
            "checklist": r.checklist,
        }
    except Exception as e:
        result["regression_49"] = {"passed": False, "error": str(e)}

    for key in list(result["reports"].keys()):
        result["reports"][key + "_exists"] = Path(result["reports"][key]).exists()

    # Overall PASS/FAIL (per document: gate is the go/no-go)
    result["overall_pass"] = result["gate"] and result["gate"].get("passed", False)
    result["overall_verdict"] = "PASS" if result["overall_pass"] else "FAIL"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"Run result written to {output_path}")
    print(f"Gate: {result['gate']['passed'] if result.get('gate') else '?'} | Regression 49: {result['regression_49']['passed'] if result.get('regression_49') else '?'} | Verdict: {result['overall_verdict']}")
    return 0 if result["overall_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
