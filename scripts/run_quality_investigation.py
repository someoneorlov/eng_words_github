#!/usr/bin/env python3
"""Run checks from quality_errors_investigation_plan.md.

Performs data checks (A2, A4, B1–B3, B4, E1, F1/F2, G1) and writes
data/experiment/investigation_report.md.

With --gate: evaluates QC gate from validation_errors in cards JSON;
writes PASS/FAIL to report and exits with 1 if gate not passed (Stage 7).

Usage (from project root):
    uv run python scripts/run_quality_investigation.py
    uv run python scripts/run_quality_investigation.py --cards path/to/cards_B_batch.json
    uv run python scripts/run_quality_investigation.py --gate --cards path/to/cards_B_batch.json --output report.md
"""

import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
DATA_DIR = PROJECT_ROOT / "data" / "experiment"
BATCH_DIR = DATA_DIR / "batch_b"
CARDS_PATH = DATA_DIR / "cards_B_batch.json"
RESULTS_PATH = BATCH_DIR / "results.jsonl"
LEMMA_EXAMPLES_PATH = BATCH_DIR / "lemma_examples.json"
REQUESTS_PATH = BATCH_DIR / "requests.jsonl"
TOKENS_PATH = DATA_DIR / "tokens_sample.parquet"
SENTENCES_PATH = DATA_DIR / "sentences_sample.parquet"
OUTPUT_REPORT = DATA_DIR / "investigation_report.md"

# Lemmas from consolidated errors (category A - empty examples)
LEMMAS_A = [
    "xviii", "commonplace", "prodigal", "normal", "stricken",
    "decrepit", "terrify", "soil", "ninth", "integrate", "idle",
    "harrow", "fungoid", "exceed", "induce",
]
# Sample lemmas for B (wrong example)
LEMMAS_B_SAMPLE = ["horsemanship", "path", "identify"]


def _lemma_in_text(lemma: str, text: str) -> bool:
    """Check if lemma or its form (including irregular: went, thought, said, etc.) appears in text."""
    if not text or not lemma:
        return False
    try:
        from eng_words.validation.example_validator import _get_word_forms, _word_in_text
        forms = _get_word_forms(lemma)
        if any(_word_in_text(f, text) for f in forms):
            return True
    except ImportError:
        pass
    # Fallback: lemma as substring or \blemma\w*\b
    low = text.lower()
    lem = lemma.lower()
    if lem in low:
        return True
    return bool(re.search(rf"\b{re.escape(lem)}\w*\b", low, re.I))


def run_b4_lemma_in_example(cards: list) -> list[dict]:
    """B4: Cards where at least one example does not contain the lemma."""
    bad = []
    for c in cards:
        lemma = c.get("lemma", "")
        examples = c.get("examples") or []
        for i, ex in enumerate(examples):
            if not _lemma_in_text(lemma, ex):
                bad.append({
                    "lemma": lemma,
                    "example_index": i + 1,
                    "example_preview": (ex[:120] + "...") if len(ex) > 120 else ex,
                })
                break
    return bad


def run_a2_a4(results_path: Path, lemma_examples_path: Path, lemmas: list[str]) -> list[dict]:
    """A2/A4: For each lemma from A, check raw response and len(lemma_examples)."""
    if not results_path.exists() or not lemma_examples_path.exists():
        return []
    with open(lemma_examples_path, encoding="utf-8") as f:
        lemma_examples = json.load(f)
    key_to_response = {}
    with open(results_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            key_to_response[row.get("key", "")] = row.get("response", {})

    rows = []
    for lemma in lemmas:
        key = f"lemma:{lemma}"
        resp = key_to_response.get(key, {})
        n_examples = len(lemma_examples.get(lemma, []))
        raw_cards = []
        raw_indices = []
        if resp.get("candidates"):
            parts = resp["candidates"][0].get("content", {}).get("parts", [])
            if parts:
                raw_text = parts[0].get("text", "")
                try:
                    data = json.loads(
                        raw_text.strip().removeprefix("```json").removeprefix("```").strip().removesuffix("```").strip()
                    )
                    raw_cards = data.get("cards", [])
                    for card in raw_cards:
                        raw_indices.append(card.get("selected_example_indices", []))
                except Exception:
                    raw_cards = ["(parse failed)"]
        rows.append({
            "lemma": lemma,
            "len_lemma_examples": n_examples,
            "raw_cards_count": len(raw_cards),
            "raw_indices_sample": str(raw_indices[:3]) if raw_indices else "-",
        })
    return rows


def run_b3_tokens_sentences(tokens_path: Path, sentences_path: Path, lemma: str) -> dict | None:
    """B3: For one lemma, get sentence_ids from tokens and check if sentence texts contain lemma."""
    if not tokens_path.exists() or not sentences_path.exists():
        return None
    tok = pd.read_parquet(tokens_path)
    sent = pd.read_parquet(sentences_path)
    subset = tok[tok["lemma"] == lemma]
    if subset.empty:
        return {"lemma": lemma, "tokens_count": 0, "sentence_ids": [], "texts_contain_lemma": []}
    sids = subset["sentence_id"].unique().tolist()
    sent_lookup = sent.set_index("sentence_id")["text"].to_dict()
    texts_contain = []
    for sid in sids[:10]:
        t = sent_lookup.get(sid, "")
        texts_contain.append(lemma.lower() in t.lower() if t else False)
    return {
        "lemma": lemma,
        "tokens_count": len(subset),
        "sentence_ids_sample": sids[:5],
        "first_text_preview": (sent_lookup.get(sids[0], ""))[:150] if sids else "",
        "texts_contain_lemma": texts_contain,
    }


def run_e1_sentence_boundaries(sentences_path: Path) -> list[dict]:
    """E1: Find sentences containing XXIX or Biltz (chapter header in sentence)."""
    if not sentences_path.exists():
        return []
    df = pd.read_parquet(sentences_path)
    out = []
    for _, row in df.iterrows():
        t = row.get("text", "") or ""
        if "XXIX" in t or "Biltz" in t:
            out.append({"sentence_id": row.get("sentence_id"), "preview": t[:200]})
    return out[:10]


def run_f1_f2_882(sentences_path: Path, tokens_path: Path) -> dict:
    """F1/F2: Find 882 in sentences and in tokens."""
    out = {"in_sentences": [], "in_tokens": None}
    if sentences_path.exists():
        sent = pd.read_parquet(sentences_path)
        for _, row in sent.iterrows():
            t = str(row.get("text", ""))
            if "882" in t:
                out["in_sentences"].append({"sentence_id": row.get("sentence_id"), "preview": t[:180]})
        out["in_sentences"] = out["in_sentences"][:5]
    if tokens_path.exists():
        tok = pd.read_parquet(tokens_path)
        # token or lemma column might have 882
        for col in ["lemma", "surface"]:
            if col in tok.columns:
                match = tok[tok[col].astype(str) == "882"]
                if not match.empty:
                    out["in_tokens"] = {"column": col, "count": len(match), "sample": match.head(2).to_dict()}
                    break
        if out["in_tokens"] is None:
            out["in_tokens"] = "no row with 882 in lemma/surface/token"
    return out


def run_g1_edge_lemmas(tokens_path: Path, lemmas: list[str]) -> list[dict]:
    """G1: For xviii, pre, ninth - pos, is_alpha, is_stop in tokens."""
    if not tokens_path.exists():
        return []
    tok = pd.read_parquet(tokens_path)
    out = []
    for lemma in lemmas:
        sub = tok[tok["lemma"] == lemma]
        if sub.empty:
            out.append({"lemma": lemma, "found": False})
            continue
        out.append({
            "lemma": lemma,
            "found": True,
            "count": len(sub),
            "pos": sub["pos"].unique().tolist() if "pos" in sub.columns else [],
            "is_alpha": sub["is_alpha"].unique().tolist() if "is_alpha" in sub.columns else [],
            "is_stop": sub["is_stop"].unique().tolist() if "is_stop" in sub.columns else [],
        })
    return out


def _run_gate_and_write_report(cards_path: Path, output_path: Path) -> bool:
    """Run QC gate on cards JSON; write short report. Returns True if PASS."""
    from eng_words.word_family.qc_gate import load_result_and_evaluate_gate

    passed, summary, message = load_result_and_evaluate_gate(cards_path)
    lines = [
        "# QC Gate Report (Pipeline B)",
        "",
        f"**Result:** {'PASS' if passed else 'FAIL'}",
        "",
        message,
        "",
        "## Summary",
        "",
        f"- Cards generated: {summary.get('cards_generated', '—')}",
        f"- Validation errors (dropped): {summary.get('validation_errors_count', '—')}",
        f"- Total processed: {summary.get('total_processed', '—')}",
        "",
    ]
    if summary.get("counts_by_type"):
        lines.append("Counts by error_type:")
        for k, v in summary["counts_by_type"].items():
            lines.append(f"- {k}: {v}")
        lines.append("")
    if summary.get("rates"):
        lines.append("Rates (errors / total_processed):")
        for k, v in summary["rates"].items():
            lines.append(f"- {k}: {v:.2%}")
        lines.append("")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(message)
    print(f"Report written to {output_path}")
    return passed


def main():
    import argparse
    p = argparse.ArgumentParser(description="Run quality investigation checks")
    p.add_argument("--cards", type=Path, default=CARDS_PATH, help="Path to cards_B_batch.json")
    p.add_argument("--output", type=Path, default=OUTPUT_REPORT, help="Output report path")
    p.add_argument("--gate", action="store_true", help="Run QC gate only: PASS/FAIL from validation_errors, exit 1 on FAIL")
    args = p.parse_args()

    if args.gate:
        passed = _run_gate_and_write_report(args.cards, args.output)
        return 0 if passed else 1

    report = []

    # --- B4: lemma in example
    if args.cards.exists():
        with open(args.cards, encoding="utf-8") as f:
            data = json.load(f)
        cards = data.get("cards", [])
        b4 = run_b4_lemma_in_example(cards)
        report.append(("B4. Карточки, где пример не содержит лемму", b4, len(b4)))
    else:
        report.append(("B4. Карточки (файл отсутствует)", [], 0))

    # --- A2/A4
    a_rows = run_a2_a4(RESULTS_PATH, LEMMA_EXAMPLES_PATH, LEMMAS_A)
    report.append(("A2/A4. Леммы с пустыми примерами (сырой ответ и len(lemma_examples))", a_rows, len(a_rows)))

    # --- B3 sample
    b3_horse = run_b3_tokens_sentences(TOKENS_PATH, SENTENCES_PATH, "horsemanship")
    b3_path = run_b3_tokens_sentences(TOKENS_PATH, SENTENCES_PATH, "path")
    report.append(("B3. Источник предложений (horsemanship)", [b3_horse] if b3_horse else [], 1 if b3_horse else 0))
    report.append(("B3. Источник предложений (path)", [b3_path] if b3_path else [], 1 if b3_path else 0))

    # --- E1
    e1 = run_e1_sentence_boundaries(SENTENCES_PATH)
    report.append(("E1. Предложения с XXIX/Biltz (границы)", e1, len(e1)))

    # --- F1/F2
    f1f2 = run_f1_f2_882(SENTENCES_PATH, TOKENS_PATH)
    report.append(("F1/F2. Поиск 882 в предложениях и токенах", [f1f2], 1))

    # --- G1
    g1 = run_g1_edge_lemmas(TOKENS_PATH, ["xviii", "pre", "ninth"])
    report.append(("G1. Пограничные леммы (pos, is_alpha, is_stop)", g1, len(g1)))

    # --- Write report
    lines = [
        "# Отчёт проверки (quality_errors_investigation_plan)",
        "",
        "Сгенерировано скриптом `scripts/run_quality_investigation.py`.",
        "",
    ]
    for title, data, count in report:
        lines.extend([f"## {title}", "", f"Записей: {count}", ""])
        if isinstance(data, list) and data and isinstance(data[0], dict):
            for i, row in enumerate(data):
                lines.append(f"### {i + 1}")
                for k, v in row.items():
                    lines.append(f"- **{k}:** {v}")
                lines.append("")
        elif data:
            lines.append(json.dumps(data, ensure_ascii=False, indent=2))
            lines.append("")
        lines.append("---")
        lines.append("")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print(f"Report written to {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
