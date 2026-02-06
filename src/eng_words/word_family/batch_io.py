"""Pipeline B batch — IO layer (reading/writing files, no network).

read_tokens, read_sentences: parquet inputs.
write_json, write_jsonl: atomic write (temp + rename) so failures do not leave broken files.
render_requests: build requests.jsonl + lemma_examples.json from tokens/sentences (no network).
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

from eng_words.word_family.batch_core import (
    build_prompt,
    build_retry_prompt,
    filter_valid_cards,
    merge_retry_results,
    parse_one_result,
    RETRY_REASON_ERROR_TYPES,
)
from eng_words.word_family.batch_schemas import (
    BatchConfig,
    BatchPaths,
    ErrorEntry,
    BatchInfo,
    CURRENT_BATCH_INFO_SCHEMA,
    read_batch_info,
    write_batch_info,
)
from eng_words.word_family import batch_api
from eng_words.word_family.clusterer import group_examples_by_lemma
from eng_words.word_family.batch_qc import (
    cards_lemma_not_in_example,
    check_qc_threshold,
    get_cards_failing_duplicate_sense,
    get_cards_failing_headword_invalid_for_mode,
    get_cards_failing_lemma_in_example,
    get_cards_failing_pos_mismatch,
)
from eng_words.word_family.contract import assert_contract_invariants
from eng_words.word_family.qc_gate import DEFAULT_QC_GATE_THRESHOLDS, evaluate_gate

# MWE/phrasal deck (PIPELINE_B_FIXES_PLAN 5)
from eng_words.constants import (
    MWE_COUNT,
    MWE_HEADWORD,
    MWE_SAMPLE_SENTENCE_IDS,
    MWE_TYPE,
    MWE_TYPE_PHRASAL_VERB,
)
from eng_words.text_norm import match_target_in_text


def _default_candidates_path(sentences_path: Path) -> Path:
    """Derive MWE candidates path from sentences path: .../book_sentences.parquet -> .../book_mwe_candidates.parquet."""
    p = Path(sentences_path)
    name = p.stem  # e.g. book_sentences
    if name.endswith("_sentences"):
        base = name[: -len("_sentences")]
        return p.parent / f"{base}_mwe_candidates.parquet"
    return p.parent / "mwe_candidates.parquet"


def load_mwe_candidates_for_deck(
    path: Path,
    mode: str,
    top_k: int,
) -> list[str]:
    """Load MWE candidates parquet, filter by mode, take top_k by count; return list of headwords."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"MWE candidates not found: {path}. Run Stage 1 with extract_mwe_candidates.")
    df = pd.read_parquet(path)
    if df.empty or MWE_HEADWORD not in df.columns:
        return []
    if mode == "phrasal" and MWE_TYPE in df.columns:
        df = df[df[MWE_TYPE] == MWE_TYPE_PHRASAL_VERB]
    if MWE_COUNT in df.columns:
        df = df.sort_values(MWE_COUNT, ascending=False)
    headwords = df[MWE_HEADWORD].astype(str).str.strip().tolist()
    if top_k > 0:
        headwords = headwords[:top_k]
    return headwords


def headword_examples_from_sentences(
    sentences_df: pd.DataFrame,
    headwords: list[str],
    max_examples: int,
) -> tuple[dict[str, list[str]], dict[str, list[dict[str, Any]]]]:
    """Build headword -> examples from sentences (match_target_in_text). Returns (texts, v2 with sentence_id)."""
    empty_v2: dict[str, list[dict[str, Any]]] = {hw: [] for hw in headwords}
    if "text" not in sentences_df.columns or "sentence_id" not in sentences_df.columns:
        return {hw: [] for hw in headwords}, empty_v2
    texts_out: dict[str, list[str]] = {}
    v2_out: dict[str, list[dict[str, Any]]] = {}
    for hw in headwords:
        rows = []
        for _, row in sentences_df.iterrows():
            text = row.get("text") or ""
            if match_target_in_text(hw, text):
                rows.append((int(row["sentence_id"]), text))
        rows.sort(key=lambda x: x[0])
        if max_examples > 0:
            rows = rows[:max_examples]
        texts_out[hw] = [t for _, t in rows]
        v2_out[hw] = [{"sentence_id": sid, "text": t} for sid, t in rows]
    return texts_out, v2_out


def read_tokens(tokens_path: Path) -> pd.DataFrame:
    """Load tokens parquet. Raises FileNotFoundError with clear message if missing."""
    path = Path(tokens_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Tokens file not found: {path}. "
            "Run: uv run python scripts/prepare_pipeline_b_sample.py [--book NAME] [--size N]"
        )
    return pd.read_parquet(path)


def read_sentences(sentences_path: Path) -> pd.DataFrame:
    """Load sentences parquet. Raises FileNotFoundError with clear message if missing."""
    path = Path(sentences_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Sentences file not found: {path}. "
            "Run: uv run python scripts/prepare_pipeline_b_sample.py [--book NAME] [--size N]"
        )
    return pd.read_parquet(path)


def read_lemma_examples(path: Path) -> dict[str, list[str]]:
    """Load lemma_examples.json. Backward compatible: v1 {lemma: [text, ...]} or v2 {lemma: [{sentence_id, text}, ...]}.
    Returns normalized dict[str, list[str]] (texts in order) for use with parse_one_result.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Lemma examples file not found: {path}")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    out: dict[str, list[str]] = {}
    for lemma, raw in data.items():
        if not raw:
            out[lemma] = []
            continue
        if isinstance(raw[0], dict):
            # v2: list of {sentence_id, text}
            out[lemma] = [item["text"] for item in raw if isinstance(item, dict) and "text" in item]
        else:
            # v1: list of strings
            out[lemma] = [str(t) for t in raw]
    return out


def read_lemma_pos_per_example(path: Path) -> dict[str, list[str]]:
    """Load lemma_pos_per_example.json: lemma -> list of POS (one per example, same order as lemma_examples).
    If file is missing (e.g. old batch), returns {}. Used for POS mismatch QC gate.
    """
    path = Path(path)
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        return {}
    return {k: list(v) if isinstance(v, list) else [] for k, v in data.items()}


def load_lemma_groups(config: BatchConfig) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Load tokens + sentences, group by lemma; return DataFrame and lemma -> examples map.
    Uses config.tokens_path, config.sentences_path, config.limit, config.max_examples.
    """
    tokens = read_tokens(config.tokens_path)
    sentences = read_sentences(config.sentences_path)
    lemma_groups = group_examples_by_lemma(tokens, sentences)
    if config.limit and config.limit > 0:
        lemma_groups = lemma_groups.head(config.limit)
    lemma_examples: dict[str, list[str]] = {}
    for _, row in lemma_groups.iterrows():
        ex = list(row["examples"])
        if len(ex) > config.max_examples:
            ex = ex[: config.max_examples]
        lemma_examples[row["lemma"]] = ex
    return lemma_groups, lemma_examples


def _retry_cache_key(lemma: str, mode: str, prompt: str) -> str:
    """Stable cache key: lemma + mode + hash of prompt."""
    h = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]
    return f"{lemma}:{mode}:{h}"


def load_retry_cache(path: Path) -> dict[str, dict]:
    """Load retry_cache.jsonl into dict key -> response. Missing file → {}."""
    path = Path(path)
    if not path.exists():
        return {}
    out: dict[str, dict] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            k = row.get("key")
            r = row.get("response")
            if k is not None and r is not None:
                out[k] = r
    return out


def append_retry_cache_entry(path: Path, key: str, response: dict) -> None:
    """Append one key/response pair to retry_cache.jsonl."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"key": key, "response": response}, ensure_ascii=False) + "\n")


def write_json(path: Path, data: Any, *, indent: int = 2, ensure_ascii: bool = False) -> None:
    """Write JSON file atomically (temp + rename). On exception, target is not created."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=indent, ensure_ascii=ensure_ascii)
        os.replace(tmp, path)
    except Exception:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
        raise


def write_jsonl(path: Path, items: list[dict[str, Any]]) -> None:
    """Write JSONL file atomically (temp + rename). On exception, target is not created."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            for obj in items:
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        os.replace(tmp, path)
    except Exception:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
        raise


def render_requests(config: BatchConfig) -> None:
    """Build requests.jsonl and lemma_examples.json from tokens/sentences (word) or candidates+sentences (phrasal/mwe).

    Determinism: word = lemmas sorted; phrasal/mwe = top_k headwords, examples by sentence_id order.
    """
    paths = BatchPaths.from_dir(config.batch_dir)

    if config.mode in ("phrasal", "mwe"):
        # Phrasal/MWE deck: headwords from candidates, examples from sentences (PIPELINE_B_FIXES_PLAN 5)
        candidates_path = config.candidates_path or _default_candidates_path(config.sentences_path)
        top_k = config.top_k if config.top_k is not None else 200
        headwords = load_mwe_candidates_for_deck(candidates_path, config.mode, top_k)
        if not headwords:
            raise ValueError(
                f"No MWE candidates for mode={config.mode}. "
                f"Check {candidates_path} exists and has phrasal_verb rows for phrasal mode."
            )
        sentences = read_sentences(config.sentences_path)
        lemma_data_texts, lemma_data_v2 = headword_examples_from_sentences(
            sentences, headwords, config.max_examples
        )
        all_lemmas = headwords
        lemma_examples = lemma_data_texts
        lemma_examples_v2 = lemma_data_v2
        lemma_pos_per_example_limited = {hw: [] for hw in headwords}
        lemma_pos_distribution = {}
    else:
        # Word deck: lemma grouping from tokens + sentences
        tokens = read_tokens(config.tokens_path)
        sentences = read_sentences(config.sentences_path)
        lemma_groups = group_examples_by_lemma(tokens, sentences)

        if lemma_groups.empty:
            raise ValueError("No lemmas from group_examples_by_lemma. Check tokens/sentences.")

        # Per-lemma: sort (sentence_id, text, pos) by sentence_id, take first max_examples. Write v2 format.
        lemma_data_v2: dict[str, list[dict[str, Any]]] = {}
        lemma_data_texts: dict[str, list[str]] = {}
        lemma_pos_distribution = {}
        lemma_pos_per_example: dict[str, list[str]] = {}
        for _, row in lemma_groups.iterrows():
            lemma = row["lemma"]
            sids = list(row["sentence_ids"])
            exs = list(row["examples"])
            pos_per = list(row.get("pos_per_example", []))
            if len(pos_per) != len(sids):
                pos_per = []
            if pos_per:
                triples = sorted(zip(sids, exs, pos_per))
                if config.max_examples > 0:
                    triples = triples[: config.max_examples]
                lemma_data_v2[lemma] = [{"sentence_id": t[0], "text": t[1]} for t in triples]
                lemma_data_texts[lemma] = [t[1] for t in triples]
                lemma_pos_per_example[lemma] = [t[2] for t in triples]
            else:
                pairs = sorted(zip(sids, exs))
                if config.max_examples > 0:
                    pairs = pairs[: config.max_examples]
                lemma_data_v2[lemma] = [{"sentence_id": sid, "text": text} for sid, text in pairs]
                lemma_data_texts[lemma] = [text for _, text in pairs]
                lemma_pos_per_example[lemma] = []
            lemma_pos_distribution[lemma] = dict(Counter(row["pos_variants"]))

        # Sort lemmas for deterministic order, then apply limit
        all_lemmas = sorted(lemma_data_v2.keys())
        if config.limit is not None and config.limit > 0:
            all_lemmas = all_lemmas[: config.limit]
        lemma_examples = {k: lemma_data_texts[k] for k in all_lemmas}
        lemma_examples_v2 = {k: lemma_data_v2[k] for k in all_lemmas}
        lemma_pos_per_example_limited = {k: lemma_pos_per_example.get(k, []) for k in all_lemmas}

    key_prefix = "headword:" if config.mode in ("phrasal", "mwe") else "lemma:"
    requests: list[dict[str, Any]] = []
    for lemma in all_lemmas:
        examples = lemma_examples[lemma]
        pos_dist = lemma_pos_distribution.get(lemma) if isinstance(lemma_pos_distribution, dict) else None
        prompt = build_prompt(lemma, examples, pos_distribution=pos_dist)
        requests.append({
            "key": f"{key_prefix}{lemma}",
            "request": {
                "model": f"models/{config.model}",
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "temperature": 0.0,
                    "maxOutputTokens": 16384,
                    "responseMimeType": "application/json",
                },
            },
        })

    paths.batch_info.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(paths.requests, requests)
    write_json(paths.lemma_examples, lemma_examples_v2, ensure_ascii=False)
    write_json(paths.lemma_pos_per_example, lemma_pos_per_example_limited, ensure_ascii=False)


def parse_results(config: BatchConfig, *, skip_validation: bool = False) -> None:
    """Read results.jsonl + lemma_examples, parse and validate, write cards + download_log (no network, no retry).

    Fail-fast: if results.jsonl or lemma_examples.json is missing, raises FileNotFoundError.
    """
    paths = BatchPaths.from_dir(config.batch_dir)
    if not paths.results.exists():
        raise FileNotFoundError(
            f"Results file not found: {paths.results}. "
            "Run batch download from API or copy results.jsonl into batch dir."
        )
    lemma_examples = read_lemma_examples(paths.lemma_examples)
    lemma_pos_per_example = read_lemma_pos_per_example(paths.lemma_pos_per_example)

    all_cards: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    error_entries: list[ErrorEntry] = []
    success_count = 0
    lemmas_with_zero_cards: list[str] = []
    cards_with_empty_examples: list[dict[str, Any]] = []
    cards_with_examples_fallback: list[dict[str, Any]] = []

    with open(paths.results, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            key = row.get("key", "")
            resp = row.get("response", {})

            lemma, cards, err = parse_one_result(key, resp, lemma_examples)
            if err:
                errors.append({"lemma": lemma, "error": err})
            else:
                success_count += 1
                if len(cards) == 0:
                    lemmas_with_zero_cards.append(lemma)
                total_ex = len(lemma_examples.get(lemma, []))
                valid_cards, new_errors = filter_valid_cards(
                    cards, lemma, skip_validation=skip_validation, stage="download"
                )
                error_entries.extend(new_errors)
                for c in valid_cards:
                    if not c.get("examples"):
                        cards_with_empty_examples.append({
                            "lemma": lemma,
                            "definition_en": (c.get("definition_en") or "")[:100],
                            "selected_example_indices": c.get("selected_example_indices", []),
                            "total_lemma_examples": total_ex,
                        })
                    elif c.get("examples_fallback"):
                        cards_with_examples_fallback.append({
                            "lemma": lemma,
                            "definition_en": (c.get("definition_en") or "")[:60],
                            "selected_example_indices": c.get("selected_example_indices", []),
                            "total_lemma_examples": total_ex,
                        })
                    c["total_lemma_examples"] = total_ex
                    all_cards.append(c)

    # Precision-first: in strict mode do not write cards with empty examples
    if config.strict:
        all_cards = [c for c in all_cards if c.get("examples")]

    # Stage 4: in strict mode drop cards where lemma/headword not in every example; record as error
    if config.strict:
        failing = get_cards_failing_lemma_in_example(all_cards)
        failing_ids = {id(c) for c in failing}
        for c in failing:
            error_entries.append(
                ErrorEntry(
                    lemma=c.get("lemma", ""),
                    stage="download",
                    error_type="lemma_not_in_example",
                    message="Target (lemma/headword) not in every example; card dropped (precision-first).",
                )
            )
        all_cards = [c for c in all_cards if id(c) not in failing_ids]

    # Stage 4b: in strict mode drop cards where headword is invalid for mode (e.g. multiword in word mode)
    if config.strict:
        hw_failing = get_cards_failing_headword_invalid_for_mode(all_cards, mode=config.mode)
        hw_failing_ids = {id(c) for c in hw_failing}
        for c in hw_failing:
            error_entries.append(
                ErrorEntry(
                    lemma=c.get("lemma", ""),
                    stage="download",
                    error_type="headword_invalid_for_mode",
                    message="Headword must be single-word in word mode; card dropped (precision-first).",
                )
            )
        all_cards = [c for c in all_cards if id(c) not in hw_failing_ids]

    # Stage 5: in strict mode drop cards where claimed POS not in selected examples; record as error
    if config.strict and lemma_pos_per_example:
        pos_failing = get_cards_failing_pos_mismatch(all_cards, lemma_pos_per_example)
        pos_failing_ids = {id(c) for c in pos_failing}
        for c in pos_failing:
            error_entries.append(
                ErrorEntry(
                    lemma=c.get("lemma", ""),
                    stage="download",
                    error_type="pos_mismatch",
                    message="Card part_of_speech not present in selected examples; card dropped (precision-first).",
                )
            )
        all_cards = [c for c in all_cards if id(c) not in pos_failing_ids]

    # Stage 6: in strict mode drop cards that are duplicate senses (same lemma, very similar definition_en)
    if config.strict:
        dup_failing = get_cards_failing_duplicate_sense(all_cards)
        dup_failing_ids = {id(c) for c in dup_failing}
        for c in dup_failing:
            error_entries.append(
                ErrorEntry(
                    lemma=c.get("lemma", ""),
                    stage="download",
                    error_type="duplicate_sense",
                    message="Definition too similar to another card for same lemma; card dropped (precision-first).",
                )
            )
        all_cards = [c for c in all_cards if id(c) not in dup_failing_ids]

    # Fail-fast: contract invariants before writing output (PIPELINE_B_FIXES_PLAN 1.2)
    assert_contract_invariants(
        all_cards,
        lemma_examples,
        strict=config.strict,
        require_examples_non_empty=config.strict,
        lemma_examples_path=paths.lemma_examples,
    )

    timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")
    result = {
        "pipeline": "B",
        "source": "batch_api",
        "timestamp": timestamp,
        "config": {"lemmas_count": len(lemma_examples)},
        "stats": {
            "lemmas_processed": success_count,
            "cards_generated": len(all_cards),
            "errors": len(errors),
            "lemmas_with_zero_cards": len(lemmas_with_zero_cards),
            "validation_errors_count": len(error_entries) if not skip_validation else None,
        },
        "cards": all_cards,
        "errors": errors,
        "validation_errors": [asdict(e) for e in error_entries],
        "lemmas_with_zero_cards": lemmas_with_zero_cards,
    }

    # QC-gate before write (PIPELINE_B_FIXES_PLAN 6.1): strict → fail if any rate exceeds threshold
    if config.strict:
        passed, _summary, message = evaluate_gate(result, DEFAULT_QC_GATE_THRESHOLDS)
        if not passed:
            raise ValueError(f"QC gate FAIL — refusing to write output: {message}")

    config.output_cards_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(config.output_cards_path, result, ensure_ascii=False)

    log = {
        "timestamp": timestamp,
        "lemmas_processed": success_count,
        "cards_generated": len(all_cards),
        "errors_count": len(errors),
        "validation_errors_count": len(error_entries),
        "validation_errors": [asdict(e) for e in error_entries],
        "lemmas_with_zero_cards": lemmas_with_zero_cards,
        "cards_with_empty_examples": cards_with_empty_examples,
        "cards_with_examples_fallback": cards_with_examples_fallback,
        "errors": errors,
    }
    write_json(paths.download_log, log, ensure_ascii=False)


def list_retry_candidates(results_path: Path, lemma_examples_path: Path) -> tuple[set[str], set[str]]:
    """Parse results.jsonl and return lemmas that would be retried (empty or fallback examples).
    No API calls. Returns (lemmas_with_empty_examples, lemmas_with_examples_fallback).
    """
    lemma_examples = read_lemma_examples(lemma_examples_path)
    lemmas_empty: set[str] = set()
    lemmas_fallback: set[str] = set()
    with open(results_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            key = row.get("key", "")
            resp = row.get("response", {})
            if not key.startswith("lemma:"):
                continue
            lemma, cards, err = parse_one_result(key, resp, lemma_examples)
            if err or not cards:
                continue
            n_ex = len(lemma_examples.get(lemma, []))
            for c in cards:
                if not c.get("examples") and n_ex > 0:
                    lemmas_empty.add(lemma)
                    break
            for c in cards:
                if c.get("examples_fallback"):
                    lemmas_fallback.add(lemma)
                    break
    return lemmas_empty, lemmas_fallback


def get_retry_candidates_and_reasons(
    all_cards: list[dict],
    validation_errors: list[dict],
) -> tuple[set[str], dict[str, str]]:
    """Return (lemmas_to_retry, lemma_reason) for 6.2/6.3. Reasons: empty_or_fallback | pos_mismatch | lemma_not_in_example | validation."""
    cards_with_empty = [c for c in all_cards if not c.get("examples")]
    cards_with_fallback = [c for c in all_cards if c.get("examples_fallback")]
    lemmas_to_retry = {c["lemma"] for c in cards_with_empty} | {c["lemma"] for c in cards_with_fallback}
    for e in validation_errors:
        if isinstance(e, dict) and e.get("error_type") in RETRY_REASON_ERROR_TYPES:
            lemmas_to_retry.add(e.get("lemma", ""))
    lemma_reason: dict[str, str] = {lem: "empty_or_fallback" for lem in lemmas_to_retry}
    for e in validation_errors:
        if isinstance(e, dict) and e.get("error_type") in RETRY_REASON_ERROR_TYPES:
            lem = e.get("lemma", "")
            if lem in lemmas_to_retry:
                lemma_reason[lem] = e.get("error_type", "validation")
    return lemmas_to_retry, lemma_reason


def run_standard_api(config: BatchConfig) -> None:
    """Run Pipeline B via Standard API (no batch job): render requests, call generate_content per lemma, write results.jsonl.
    Use for small test runs (e.g. limit 50); then run download_batch(config, from_file=True) to parse and write cards.
    Does not write batch_info.json (no batch job created).
    """
    paths = BatchPaths.from_dir(config.batch_dir)
    paths.batch_info.parent.mkdir(parents=True, exist_ok=True)
    render_requests(config)
    client = batch_api.get_client()
    results_lines: list[dict[str, Any]] = []
    with open(paths.requests, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            key = row.get("key", "")
            req = row.get("request", {})
            parts = (req.get("contents") or [{}])[0].get("parts") or [{}]
            prompt = (parts[0].get("text") or "").strip()
            if not prompt:
                results_lines.append({"key": key, "response": {}})
                continue
            resp = batch_api.generate_content_for_prompt(client, config.model, prompt)
            results_lines.append({"key": key, "response": resp or {}})
    write_jsonl(paths.results, results_lines)


def create_batch(config: BatchConfig, *, overwrite: bool = False) -> BatchInfo:
    """Render requests, upload to Gemini, create batch job, write batch_info.json.
    If batch_info.json already exists and overwrite is False, raises FileExistsError.
    """
    paths = BatchPaths.from_dir(config.batch_dir)
    if paths.batch_info.exists() and not overwrite:
        raise FileExistsError(
            f"Batch info already exists: {paths.batch_info}. "
            "Use overwrite=True to replace, or remove the file."
        )
    render_requests(config)
    lemma_examples = read_lemma_examples(paths.lemma_examples)
    n = len(lemma_examples)
    client = batch_api.get_client()
    uploaded_name = batch_api.upload_requests_file(client, paths.requests)
    job = batch_api.create_batch_job(
        client,
        config.model,
        uploaded_name,
        display_name=f"pipeline_b_{n}_lemmas",
    )
    info = BatchInfo(
        schema_version=CURRENT_BATCH_INFO_SCHEMA,
        batch_name=job.name,
        model=config.model,
        lemmas_count=n,
        created_at=time.strftime("%Y-%m-%d %H:%M:%S"),
        uploaded_file=uploaded_name,
    )
    write_batch_info(paths.batch_info, info)
    return info


def get_batch_status(batch_dir: Path) -> dict[str, Any]:
    """Read batch_info, call API, return structured status (state, batch_name, model, lemmas_count)."""
    paths = BatchPaths.from_dir(Path(batch_dir))
    info = read_batch_info(paths.batch_info)
    if not info.batch_name:
        return {
            "state": "NO_BATCH",
            "batch_name": None,
            "model": info.model,
            "lemmas_count": info.lemmas_count,
        }
    client = batch_api.get_client()
    job = client.batches.get(name=info.batch_name)
    return {
        "state": job.state.name,
        "batch_name": job.name,
        "model": info.model,
        "lemmas_count": info.lemmas_count,
        "error": getattr(job, "error", None),
    }


def wait_for_batch(
    batch_dir: Path,
    poll_interval_sec: int = 60,
    timeout_sec: int | None = None,
) -> dict[str, Any]:
    """Poll batch status until SUCCEEDED, FAILED, or CANCELLED. Does not call download."""
    paths = BatchPaths.from_dir(Path(batch_dir))
    if not paths.batch_info.exists():
        raise FileNotFoundError(f"Batch info not found: {paths.batch_info}. Run create first.")
    info = read_batch_info(paths.batch_info)
    if not info.batch_name:
        raise ValueError("Batch info has no batch_name.")
    client = batch_api.get_client()
    name = info.batch_name
    start = time.monotonic()
    while True:
        job = client.batches.get(name=name)
        state = job.state.name
        if state == "JOB_STATE_SUCCEEDED":
            return {"state": state, "batch_name": job.name, "model": info.model, "lemmas_count": info.lemmas_count}
        if state in ("JOB_STATE_FAILED", "JOB_STATE_CANCELLED"):
            raise RuntimeError(
                f"Batch ended with {state}"
                + (f": {getattr(job, 'error', '')}" if getattr(job, "error", None) else "")
            )
        if timeout_sec is not None and (time.monotonic() - start) >= timeout_sec:
            raise TimeoutError(f"Batch not complete after {timeout_sec}s. State: {state}")
        time.sleep(poll_interval_sec)


def download_batch(
    config: BatchConfig,
    *,
    from_file: bool = False,
    skip_validation: bool = False,
    retry_empty: bool = True,
    retry_thinking: bool = False,
    thinking_model: str = "gemini-2.5-pro",
) -> None:
    """Download results (if not from_file), parse, optionally run retry. Writes cards + download_log.
    When from_file=True uses existing results.jsonl (no API download); retry may still use network.
    """
    paths = BatchPaths.from_dir(config.batch_dir)
    if not paths.lemma_examples.exists():
        raise FileNotFoundError(
            f"Lemma examples not found: {paths.lemma_examples}. Re-run create or copy lemma_examples.json."
        )
    if not from_file:
        info = read_batch_info(paths.batch_info)
        if not info.batch_name:
            raise ValueError("Batch info has no batch_name.")
        client = batch_api.get_client()
        status = get_batch_status(config.batch_dir)
        if status["state"] != "JOB_STATE_SUCCEEDED":
            raise RuntimeError(f"Batch not complete: {status['state']}. Run status or wait.")
        content = batch_api.download_batch_results(client, info.batch_name)
        paths.results.write_bytes(content)

    parse_results(config, skip_validation=skip_validation)

    if not (retry_empty or retry_thinking):
        return

    # Retry: read current cards, run retry loop (with cache), write again (6.2: once per lemma, super-strict prompt)
    result = json.loads(config.output_cards_path.read_text(encoding="utf-8"))
    all_cards = list(result["cards"])
    lemma_examples = read_lemma_examples(paths.lemma_examples)
    client = batch_api.get_client()
    validation_errors = list(result.get("validation_errors", []))
    retry_cache = load_retry_cache(paths.retry_cache)
    retry_log: list[dict[str, Any]] = []

    lemmas_to_retry, lemma_reason = get_retry_candidates_and_reasons(all_cards, validation_errors)

    def _get_retry_response(lemma: str, mode: str, use_thinking: bool) -> tuple[dict | None, str]:
        examples = lemma_examples.get(lemma, [])
        if not examples:
            return None, "network"
        prompt = build_retry_prompt(lemma, examples)
        key = _retry_cache_key(lemma, mode, prompt)
        if key in retry_cache:
            return retry_cache[key], "cache"
        resp = batch_api.call_standard_retry(
            client, lemma, lemma_examples, config.model,
            thinking_model=thinking_model, use_thinking=use_thinking,
        )
        if resp:
            append_retry_cache_entry(paths.retry_cache, key, resp)
            retry_cache[key] = resp
        return resp, "network"

    if retry_empty and lemmas_to_retry:
        for lemma in sorted(lemmas_to_retry):
            reason = lemma_reason.get(lemma, "empty_or_fallback")
            resp, source = _get_retry_response(lemma, "standard", use_thinking=False)
            if not resp:
                retry_log.append({"lemma": lemma, "reason": reason, "source": source, "outcome": "failed"})
                continue
            _, new_cards, err = parse_one_result(f"lemma:{lemma}", resp, lemma_examples)
            if err or not new_cards:
                retry_log.append({"lemma": lemma, "reason": reason, "source": source, "outcome": "failed"})
                continue
            total_ex = len(lemma_examples.get(lemma, []))
            valid_new, new_errs = filter_valid_cards(new_cards, lemma, skip_validation=skip_validation, stage="retry")
            validation_errors.extend(asdict(e) for e in new_errs)
            for c in valid_new:
                c["total_lemma_examples"] = total_ex
            if valid_new and all(c.get("examples") and not c.get("examples_fallback") for c in valid_new):
                all_cards = merge_retry_results(all_cards, lemma, valid_new)
                retry_log.append({"lemma": lemma, "reason": reason, "source": source, "outcome": "success"})
            else:
                retry_log.append({"lemma": lemma, "reason": reason, "source": source, "outcome": "validation_failed"})

    if retry_thinking:
        lemmas_still_empty = {c["lemma"] for c in all_cards if not c.get("examples")}
        for lemma in sorted(lemmas_still_empty):
            resp, source = _get_retry_response(lemma, "thinking", use_thinking=True)
            if not resp:
                retry_log.append({"lemma": lemma, "reason": "still_empty", "source": source, "outcome": "failed"})
                continue
            _, new_cards, err = parse_one_result(f"lemma:{lemma}", resp, lemma_examples)
            if err or not new_cards:
                retry_log.append({"lemma": lemma, "reason": "still_empty", "source": source, "outcome": "failed"})
                continue
            total_ex = len(lemma_examples.get(lemma, []))
            valid_new, new_errs = filter_valid_cards(new_cards, lemma, skip_validation=skip_validation, stage="retry_thinking")
            validation_errors.extend(asdict(e) for e in new_errs)
            for c in valid_new:
                c["total_lemma_examples"] = total_ex
            if valid_new and all(c.get("examples") for c in valid_new):
                all_cards = merge_retry_results(all_cards, lemma, valid_new)
                retry_log.append({"lemma": lemma, "reason": "still_empty", "source": source, "outcome": "success"})
            else:
                retry_log.append({"lemma": lemma, "reason": "still_empty", "source": source, "outcome": "validation_failed"})

    # Stage 4: in strict mode drop cards where lemma/headword not in every example (also after retry merge)
    if config.strict:
        failing = get_cards_failing_lemma_in_example(all_cards)
        failing_ids = {id(c) for c in failing}
        for c in failing:
            validation_errors.append({
                "lemma": c.get("lemma", ""),
                "stage": "download",
                "error_type": "lemma_not_in_example",
                "message": "Target (lemma/headword) not in every example; card dropped (precision-first).",
            })
        all_cards = [c for c in all_cards if id(c) not in failing_ids]

    # Stage 4b: in strict mode drop cards where headword invalid for mode (e.g. multiword in word mode)
    if config.strict:
        hw_failing = get_cards_failing_headword_invalid_for_mode(all_cards, mode=config.mode)
        hw_failing_ids = {id(c) for c in hw_failing}
        for c in hw_failing:
            validation_errors.append({
                "lemma": c.get("lemma", ""),
                "stage": "download",
                "error_type": "headword_invalid_for_mode",
                "message": "Headword must be single-word in word mode; card dropped (precision-first).",
            })
        all_cards = [c for c in all_cards if id(c) not in hw_failing_ids]

    timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")
    result["cards"] = all_cards
    result["timestamp"] = timestamp
    result["stats"]["cards_generated"] = len(all_cards)
    result["stats"]["validation_errors_count"] = len(validation_errors) if not skip_validation else None
    result["validation_errors"] = validation_errors

    cards_lemma_not_in_example_list = cards_lemma_not_in_example(all_cards)
    if config.strict and (config.max_warning_rate is not None or config.max_warnings_absolute is not None):
        check_qc_threshold(
            len(cards_lemma_not_in_example_list),
            len(all_cards),
            strict=config.strict,
            max_warning_rate=config.max_warning_rate,
            max_warnings_absolute=config.max_warnings_absolute,
            label="lemma_not_in_example",
        )

    result["stats"]["cards_lemma_not_in_example_count"] = len(cards_lemma_not_in_example_list)

    # QC-gate before write (PIPELINE_B_FIXES_PLAN 6.1): strict → fail if any rate exceeds threshold
    if config.strict:
        passed, _summary, message = evaluate_gate(result, DEFAULT_QC_GATE_THRESHOLDS)
        if not passed:
            raise ValueError(f"QC gate FAIL — refusing to write output: {message}")

    write_json(config.output_cards_path, result, ensure_ascii=False)
    log = {
        "timestamp": timestamp,
        "cards_generated": len(all_cards),
        "validation_errors_count": len(validation_errors),
        "validation_errors": validation_errors,
        "errors": result.get("errors", []),
        "retry_log": retry_log,
        "cards_lemma_not_in_example_count": len(cards_lemma_not_in_example_list),
        "cards_lemma_not_in_example": cards_lemma_not_in_example_list,
    }
    write_json(paths.download_log, log, ensure_ascii=False)
