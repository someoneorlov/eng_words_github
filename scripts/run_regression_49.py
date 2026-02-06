#!/usr/bin/env python3
"""Stage 7 regression: evaluate 49/49 cards against PASS/FAIL criteria.

Loads cards JSON, optionally lemma_pos_per_example and manual_qc; runs
regression criteria (valid_schema, lemma/headword_in_example, pos_consistency,
major_or_invalid). Exits 0 on PASS, 1 on FAIL; writes report to --output.

Usage (from project root):
    uv run python scripts/run_regression_49.py --cards data/experiment/cards_B_batch_2.json
    uv run python scripts/run_regression_49.py --cards path/to/cards.json --output reports/regression_49.md
    uv run python scripts/run_regression_49.py --cards path/to/cards.json --lemma-pos batch_b/lemma_pos_per_example.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eng_words.word_family.regression import (
    load_result_and_evaluate_regression,
    REGRESSION_LEMMA_IN_EXAMPLE_MIN,
    REGRESSION_MAJOR_OR_INVALID_MAX,
    REGRESSION_POS_CONSISTENCY_MIN,
    REGRESSION_VALID_SCHEMA_MIN,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Pipeline B Stage 7 regression (49/49 PASS/FAIL)")
    parser.add_argument(
        "--cards",
        type=Path,
        default=PROJECT_ROOT / "data" / "experiment" / "cards_B_batch_2.json",
        help="Path to pipeline output cards JSON",
    )
    parser.add_argument(
        "--lemma-pos",
        type=Path,
        default=None,
        help="Path to lemma_pos_per_example.json (optional, for pos_consistency)",
    )
    parser.add_argument(
        "--manual-qc",
        type=Path,
        default=None,
        help="Path to manual QC JSON (card_key -> OK|Minor|Major|Invalid)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write report to this path (default: stdout)",
    )
    args = parser.parse_args()

    cards_path = args.cards if args.cards.is_absolute() else PROJECT_ROOT / args.cards
    lemma_pos_path = None
    if args.lemma_pos:
        lemma_pos_path = args.lemma_pos if args.lemma_pos.is_absolute() else PROJECT_ROOT / "data" / "experiment" / args.lemma_pos
    manual_qc_path = None
    if args.manual_qc:
        manual_qc_path = args.manual_qc if args.manual_qc.is_absolute() else PROJECT_ROOT / args.manual_qc

    r = load_result_and_evaluate_regression(
        cards_path,
        lemma_pos_per_example_path=lemma_pos_path,
        manual_qc_path=manual_qc_path,
    )

    lines = [
        "# Pipeline B — Stage 7 regression report (49/49)",
        "",
        f"**Cards file:** `{cards_path}`",
        f"**Total cards:** {r.total_cards}",
        "",
        "## Criteria (strict)",
        "",
        f"| Criterion | Threshold | Actual | Status |",
        f"|-----------|-----------|--------|--------|",
        f"| valid_schema_rate | == {REGRESSION_VALID_SCHEMA_MIN} | {r.valid_schema_rate:.2%} | {'✓' if r.valid_schema_rate >= REGRESSION_VALID_SCHEMA_MIN else '✗'} |",
    ]
    le_rate = r.lemma_headword_in_example_rate
    if le_rate is not None:
        lines.append(f"| lemma/headword_in_example_rate | >= {REGRESSION_LEMMA_IN_EXAMPLE_MIN} | {le_rate:.2%} | {'✓' if le_rate >= REGRESSION_LEMMA_IN_EXAMPLE_MIN else '✗'} |")
    else:
        lines.append("| lemma/headword_in_example_rate | >= 0.98 | (skipped) | — |")
    pos_rate = r.pos_consistency_rate
    if pos_rate is not None:
        lines.append(f"| pos_consistency_rate | >= {REGRESSION_POS_CONSISTENCY_MIN} | {pos_rate:.2%} | {'✓' if pos_rate >= REGRESSION_POS_CONSISTENCY_MIN else '✗'} |")
    else:
        lines.append("| pos_consistency_rate | >= 0.98 | (N/A, no lemma_pos) | — |")
    lines.append(f"| major_or_invalid_rate | <= {REGRESSION_MAJOR_OR_INVALID_MAX} | {r.major_or_invalid_rate:.2%} | {'✓' if r.major_or_invalid_rate <= REGRESSION_MAJOR_OR_INVALID_MAX else '✗'} |")
    lines.extend(["", "## Result", ""])
    if r.passed:
        lines.append("**PASS** — all criteria met.")
    else:
        lines.append("**FAIL** — see checklist below.")
        lines.append("")
        lines.append("### Checklist (what to fix)")
        for item in r.checklist:
            lines.append(f"- {item}")
        if r.failing_cards:
            lines.append("")
            lines.append("### Sample failing cards")
            lines.append("")
            for fc in r.failing_cards[:15]:
                lines.append(f"- lemma=`{fc.get('lemma')}` reason=`{fc.get('reason')}`")
    lines.append("")

    report = "\n".join(lines)
    if args.output:
        out_path = args.output if args.output.is_absolute() else PROJECT_ROOT / args.output
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(report, encoding="utf-8")
        print(report)
        print(f"Report written to {out_path}", file=sys.stderr)
    else:
        print(report)

    return 0 if r.passed else 1


if __name__ == "__main__":
    sys.exit(main())
