#!/usr/bin/env python3
"""Full Pipeline B run: Stage 1 (sample) + Stage 2 (LLM batch) with fixed seed and 50 lemmas / 50 examples.

Produces a single JSON with all results: stage1 summary, stage2 cards path and stats, gate, regression_49, deliverables.
Same seed (42) and limit 50, max_examples 50 for comparable results with the previous test run.

Prerequisites:
- Stage 1 full outputs: data/processed/{book}_tokens.parquet, data/processed/{book}_sentences.parquet
  (create via: uv run python -m eng_words.pipeline --book-path data/raw/BOOK.epub --book-name BOOK)
- GOOGLE_API_KEY for Stage 2 (create + download).

Usage (from project root):
    uv run python scripts/run_full_pipeline_b_and_collect_json.py --book american_tragedy
    uv run python scripts/run_full_pipeline_b_and_collect_json.py --book american_tragedy --seed 42 --limit 50 --max-examples 50 --output-json data/experiment/full_run_result.json
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
EXPERIMENT_DIR = DATA_DIR / "experiment"
BATCH_DIR = EXPERIMENT_DIR / "batch_b"

# Default: same as previous test run (50 lemmas, 50 examples, seed 42)
DEFAULT_SEED = 42
DEFAULT_LIMIT = 50
DEFAULT_MAX_EXAMPLES = 50
DEFAULT_SAMPLE_SIZE = 2000


def _run(cmd: list[str], cwd: Path | None = None, env: dict | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        cwd=cwd or PROJECT_ROOT,
        env={**(env or {})},
        capture_output=True,
        text=True,
        timeout=600,
    )


def main() -> int:
    import argparse
    p = argparse.ArgumentParser(description="Full Pipeline B run (Stage 1 + Stage 2) and collect JSON")
    p.add_argument("--book", type=str, default="american_tragedy", help="Book name for Stage 1 (data/processed/{book}_tokens.parquet)")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed for sentence sampling")
    p.add_argument("--limit", type=int, default=DEFAULT_LIMIT, help="Max lemmas for Pipeline B (50 for comparable run)")
    p.add_argument("--max-examples", type=int, default=DEFAULT_MAX_EXAMPLES, help="Max examples per lemma")
    p.add_argument("--size", type=int, default=DEFAULT_SAMPLE_SIZE, help="Number of sentences to sample in Stage 1")
    p.add_argument("--output-cards", type=Path, default=EXPERIMENT_DIR / "cards_B_batch_full_run.json", help="Stage 2 output cards path")
    p.add_argument("--output-json", type=Path, default=EXPERIMENT_DIR / "full_run_result.json", help="Final JSON with all results")
    p.add_argument("--skip-stage1", action="store_true", help="Skip Stage 1 (use existing tokens_sample.parquet / sentences_sample.parquet)")
    p.add_argument("--skip-stage2", action="store_true", help="Skip Stage 2 (only evaluate existing output-cards and write JSON)")
    args = p.parse_args()

    output_json = args.output_json if args.output_json.is_absolute() else PROJECT_ROOT / args.output_json
    output_cards = args.output_cards if args.output_cards.is_absolute() else PROJECT_ROOT / args.output_cards

    result = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": {
            "book": args.book,
            "seed": args.seed,
            "limit": args.limit,
            "max_examples": args.max_examples,
            "sample_size": args.size,
        },
        "stage1": None,
        "stage2": None,
        "gate": None,
        "regression_49": None,
        "reports": {},
        "deliverables": {
            "code": [
                "contract invariants: src/eng_words/word_family/contract.py",
                "QCPolicy strict/relaxed: src/eng_words/word_family/qc_types.py",
                "normalize_for_matching + contractions: src/eng_words/text_norm.py",
                "headword resolution + validation: src/eng_words/word_family/headword.py",
                "mode routing (word/phrasal/mwe): src/eng_words/word_family/batch_io.py, batch_schemas.py",
            ],
            "reports": [
                "investigation_report.md",
                "qc_gate_report.md",
                "regression_49_report.md",
                "quality_report_B_batch.md",
            ],
            "docs": [
                "Роли Stage1 vs LLM: docs/PIPELINE_B_CONTRACTS_AND_RUN.md",
                "Mode contracts: docs/PIPELINE_B_CONTRACTS_AND_RUN.md",
                "Candidate extraction policy: docs/PIPELINE_B_CONTRACTS_AND_RUN.md",
            ],
            "commands": [
                "word deck: run_pipeline_b_batch.py create / download [--run-gate]",
                "phrasal deck: BatchConfig(mode=phrasal, top_k=..., candidates_path=...) + render_requests / create_batch / download_batch",
            ],
        },
        "overall_verdict": None,
    }

    # --- Stage 1: prepare sample (tokens_sample.parquet, sentences_sample.parquet, sample_stats.json)
    tokens_sample = EXPERIMENT_DIR / "tokens_sample.parquet"
    sentences_sample = EXPERIMENT_DIR / "sentences_sample.parquet"
    sample_stats_path = EXPERIMENT_DIR / "sample_stats.json"

    if not args.skip_stage1:
        r = _run([
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "prepare_pipeline_b_sample.py"),
            "--book", args.book,
            "--seed", str(args.seed),
            "--size", str(args.size),
            "--data-dir", str(DATA_DIR),
        ])
        if r.returncode != 0:
            print(r.stderr or r.stdout, file=sys.stderr)
            print(
                "Stage 1 failed. Ensure Stage 1 full outputs exist:\n"
                "  data/processed/{" + args.book + "}_tokens.parquet\n"
                "  data/processed/{" + args.book + "}_sentences.parquet\n"
                "Create them with: uv run python -m eng_words.pipeline --book-path data/raw/BOOK.epub --book-name " + args.book,
                file=sys.stderr,
            )
            return 1
        if sample_stats_path.exists():
            result["stage1"] = {"stats": json.loads(sample_stats_path.read_text(encoding="utf-8"))}
        result["stage1"] = result["stage1"] or {"done": True}
        result["stage1"]["tokens_sample"] = str(tokens_sample)
        result["stage1"]["sentences_sample"] = str(sentences_sample)
    else:
        if not tokens_sample.exists() or not sentences_sample.exists():
            print("--skip-stage1 set but tokens_sample.parquet or sentences_sample.parquet missing.", file=sys.stderr)
            return 1
        result["stage1"] = {"skipped": True, "tokens_sample": str(tokens_sample), "sentences_sample": str(sentences_sample)}
        if sample_stats_path.exists():
            result["stage1"]["stats"] = json.loads(sample_stats_path.read_text(encoding="utf-8"))

    # --- Stage 2: render_requests (limit, max_examples) -> create_batch -> wait -> download
    # BatchConfig uses TOKENS_PATH, SENTENCES_PATH from batch.py which point to experiment/tokens_sample.parquet
    # We need to call CLI with limit and max_examples; output path we override by writing to output_cards after.
    if not args.skip_stage2:
        # render-requests
        r = _run([
            sys.executable, str(PROJECT_ROOT / "scripts" / "run_pipeline_b_batch.py"),
            "render-requests", "--limit", str(args.limit), "--max-examples", str(args.max_examples),
        ])
        if r.returncode != 0:
            print(r.stderr or r.stdout, file=sys.stderr)
            return 1
        # create (overwrites batch_info; uses batch_dir and default output_cards path)
        r = _run([
            sys.executable, str(PROJECT_ROOT / "scripts" / "run_pipeline_b_batch.py"),
            "create", "--limit", str(args.limit), "--max-examples", str(args.max_examples), "--overwrite",
        ])
        if r.returncode != 0:
            print(r.stderr or r.stdout, file=sys.stderr)
            return 1
        # wait (includes download; writes to data/experiment/cards_B_batch.json)
        r = _run([
            sys.executable, str(PROJECT_ROOT / "scripts" / "run_pipeline_b_batch.py"), "wait",
        ])
        if r.returncode != 0:
            print(r.stderr or r.stdout, file=sys.stderr)
            return 1
        default_cards = EXPERIMENT_DIR / "cards_B_batch.json"
        cards_path_for_eval = default_cards if default_cards.exists() else output_cards
        if default_cards.exists() and output_cards != default_cards:
            import shutil
            shutil.copy2(default_cards, output_cards)
        if cards_path_for_eval.exists():
            data = json.loads(cards_path_for_eval.read_text(encoding="utf-8"))
            result["stage2"] = {
                "cards_path": str(cards_path_for_eval),
                "stats": data.get("stats", {}),
                "cards_generated": data.get("stats", {}).get("cards_generated"),
                "validation_errors_count": len(data.get("validation_errors", [])),
            }
    else:
        for candidate in [output_cards, EXPERIMENT_DIR / "cards_B_batch.json", EXPERIMENT_DIR / "cards_B_batch_2.json"]:
            if candidate.exists():
                cards_path_for_eval = candidate
                break
        else:
            cards_path_for_eval = output_cards
        if not cards_path_for_eval.exists():
            print("--skip-stage2 set but no cards file found.", file=sys.stderr)
            return 1
        data = json.loads(cards_path_for_eval.read_text(encoding="utf-8"))
        result["stage2"] = {
            "cards_path": str(cards_path_for_eval),
            "stats": data.get("stats", {}),
            "cards_generated": data.get("stats", {}).get("cards_generated"),
            "validation_errors_count": len(data.get("validation_errors", [])),
            "skipped": True,
        }

    # --- Gate + regression (in-process)
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    cards_path = Path(result["stage2"]["cards_path"])
    try:
        from eng_words.word_family.qc_gate import load_result_and_evaluate_gate
        passed, summary, message = load_result_and_evaluate_gate(cards_path)
        result["gate"] = {"passed": passed, "message": message, "summary": summary}
    except Exception as e:
        result["gate"] = {"passed": False, "message": str(e), "summary": {}}
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

    result["overall_verdict"] = "PASS" if (result["gate"] and result["gate"].get("passed")) else "FAIL"
    result["reports"] = {
        "qc_gate": str(EXPERIMENT_DIR / "qc_gate_report.md"),
        "regression_49": str(EXPERIMENT_DIR / "regression_49_report.md"),
        "investigation": str(EXPERIMENT_DIR / "investigation_report.md"),
        "quality_sample": str(EXPERIMENT_DIR / "quality_report_B_batch.md"),
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"Full run result written to {output_json}")
    print(f"Stage1: {'ok' if result.get('stage1') else '-'} | Stage2: {'ok' if result.get('stage2') else '-'} | Gate: {result['gate']['passed'] if result.get('gate') else '?'} | Regression: {result['regression_49']['passed'] if result.get('regression_49') else '?'} | Verdict: {result['overall_verdict']}")
    return 0 if result["overall_verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
