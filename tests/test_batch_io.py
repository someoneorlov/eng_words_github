"""Tests for Pipeline B batch IO (batch_io.py)."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from eng_words.word_family import batch_io
from eng_words.word_family.batch_io import (
    append_retry_cache_entry,
    create_batch,
    download_batch,
    get_batch_status,
    headword_examples_from_sentences,
    list_retry_candidates,
    load_lemma_groups,
    load_mwe_candidates_for_deck,
    load_retry_cache,
    parse_results,
    read_lemma_examples,
    read_lemma_pos_per_example,
    read_sentences,
    read_tokens,
    render_requests,
    wait_for_batch,
    write_json,
    write_jsonl,
)
from eng_words.word_family.batch_schemas import BatchConfig


class TestReadTokens:
    def test_missing_file_raises_file_not_found(self, tmp_path: Path):
        path = tmp_path / "tokens.parquet"
        assert not path.exists()
        with pytest.raises(FileNotFoundError) as exc_info:
            read_tokens(path)
        assert "tokens" in str(exc_info.value).lower() or path.name in str(exc_info.value)


class TestReadSentences:
    def test_missing_file_raises_file_not_found(self, tmp_path: Path):
        path = tmp_path / "sentences.parquet"
        assert not path.exists()
        with pytest.raises(FileNotFoundError) as exc_info:
            read_sentences(path)
        assert "sentence" in str(exc_info.value).lower() or path.name in str(exc_info.value)


class TestWriteJsonAtomic:
    def test_writes_file(self, tmp_path: Path):
        path = tmp_path / "out.json"
        write_json(path, {"a": 1})
        assert path.exists()
        with open(path, encoding="utf-8") as f:
            assert json.load(f) == {"a": 1}

    def test_atomic_write_on_failure_does_not_leave_target(self, tmp_path: Path):
        path = tmp_path / "out.json"
        # Writing non-serializable data will fail inside json.dump
        class Bad:
            pass
        try:
            write_json(path, {"x": Bad()})
        except (TypeError, ValueError):
            pass
        assert not path.exists(), "Target must not exist after failed write"
        # Temp file may or may not exist; target must not be created
        for p in tmp_path.iterdir():
            assert p.name != "out.json"


class TestWriteJsonlAtomic:
    def test_writes_lines(self, tmp_path: Path):
        path = tmp_path / "out.jsonl"
        write_jsonl(path, [{"k": "a"}, {"k": "b"}])
        assert path.exists()
        lines = path.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 2
        assert json.loads(lines[0]) == {"k": "a"}
        assert json.loads(lines[1]) == {"k": "b"}

    def test_atomic_write_on_failure_does_not_leave_target(self, tmp_path: Path):
        path = tmp_path / "out.jsonl"
        class Bad:
            pass
        try:
            write_jsonl(path, [{"a": 1}, {"b": Bad()}])
        except (TypeError, ValueError):
            pass
        assert not path.exists(), "Target must not exist after failed write"


def _make_tokens_sentences(tmp_path: Path) -> tuple[Path, Path]:
    """Create minimal tokens.parquet and sentences.parquet for render_requests tests."""
    tokens = pd.DataFrame({
        "lemma": ["run", "run", "go", "go"],
        "sentence_id": [1, 2, 3, 4],
        "pos": ["VERB", "VERB", "VERB", "VERB"],
        "is_alpha": [True, True, True, True],
        "is_stop": [False, False, False, False],
    })
    sentences = pd.DataFrame({
        "sentence_id": [1, 2, 3, 4],
        "text": ["He runs.", "She runs.", "Go away.", "We go home."],
    })
    tok_path = tmp_path / "tokens.parquet"
    sent_path = tmp_path / "sentences.parquet"
    tokens.to_parquet(tok_path, index=False)
    sentences.to_parquet(sent_path, index=False)
    return tok_path, sent_path


class TestRenderRequests:
    def test_creates_requests_and_lemma_examples_files(self, tmp_path: Path):
        tok_path, sent_path = _make_tokens_sentences(tmp_path)
        batch_dir = tmp_path / "batch_b"
        config = BatchConfig(
            tokens_path=tok_path,
            sentences_path=sent_path,
            batch_dir=batch_dir,
            output_cards_path=tmp_path / "cards.json",
            limit=None,
            max_examples=10,
        )
        render_requests(config)
        assert (batch_dir / "requests.jsonl").exists()
        assert (batch_dir / "lemma_examples.json").exists()
        assert (batch_dir / "lemma_pos_per_example.json").exists()

    def test_lemma_order_stable_and_limit_applied(self, tmp_path: Path):
        tok_path, sent_path = _make_tokens_sentences(tmp_path)
        batch_dir = tmp_path / "batch_b"
        config = BatchConfig(
            tokens_path=tok_path,
            sentences_path=sent_path,
            batch_dir=batch_dir,
            output_cards_path=tmp_path / "cards.json",
            limit=1,
            max_examples=10,
        )
        render_requests(config)
        with open(batch_dir / "requests.jsonl", encoding="utf-8") as f:
            lines = f.readlines()
        keys = [json.loads(line)["key"] for line in lines if line.strip()]
        # Sorted lemmas: go, run. limit=1 → only first alphabetically = "go"
        assert keys == ["lemma:go"]
        le = json.loads((batch_dir / "lemma_examples.json").read_text(encoding="utf-8"))
        assert list(le.keys()) == ["go"]

    def test_max_examples_trims_stably(self, tmp_path: Path):
        tok_path, sent_path = _make_tokens_sentences(tmp_path)
        batch_dir = tmp_path / "batch_b"
        config = BatchConfig(
            tokens_path=tok_path,
            sentences_path=sent_path,
            batch_dir=batch_dir,
            output_cards_path=tmp_path / "cards.json",
            limit=None,
            max_examples=1,
        )
        render_requests(config)
        le = read_lemma_examples(batch_dir / "lemma_examples.json")
        assert le["go"] == ["Go away."]  # sentence_id 3 first when sorted
        assert le["run"] == ["He runs."]  # sentence_id 1 first when sorted

    def test_writes_lemma_examples_v2_format(self, tmp_path: Path):
        tok_path, sent_path = _make_tokens_sentences(tmp_path)
        batch_dir = tmp_path / "batch_b"
        config = BatchConfig(
            tokens_path=tok_path,
            sentences_path=sent_path,
            batch_dir=batch_dir,
            output_cards_path=tmp_path / "cards.json",
            limit=None,
            max_examples=2,
        )
        render_requests(config)
        raw = json.loads((batch_dir / "lemma_examples.json").read_text(encoding="utf-8"))
        assert isinstance(raw["go"][0], dict)
        assert "sentence_id" in raw["go"][0] and "text" in raw["go"][0]
        assert raw["go"][0]["text"] == "Go away."

    def test_render_requests_phrasal_mode_uses_candidates_and_headword_keys(self, tmp_path: Path):
        """Stage 5: phrasal mode builds requests from MWE candidates and sentences; keys are headword:."""
        tok_path, sent_path = _make_tokens_sentences(tmp_path)
        # Overwrite sentences with text containing "look up" for headword matching
        sentences = pd.DataFrame({
            "sentence_id": [1, 2],
            "text": ["I look up the word.", "We look up the address."],
        })
        sentences.to_parquet(sent_path, index=False)
        candidates_path = tmp_path / "book_mwe_candidates.parquet"
        mwe_df = pd.DataFrame([{
            "headword": "look up",
            "type": "phrasal_verb",
            "count": 2,
            "sample_sentence_ids": [1, 2],
            "source": "stage1_detector",
        }])
        mwe_df.to_parquet(candidates_path, index=False)
        batch_dir = tmp_path / "batch_phrasal"
        batch_dir.mkdir()
        config = BatchConfig(
            tokens_path=tok_path,
            sentences_path=sent_path,
            batch_dir=batch_dir,
            output_cards_path=tmp_path / "cards_phrasal.json",
            mode="phrasal",
            top_k=1,
            max_examples=10,
            candidates_path=candidates_path,
        )
        render_requests(config)
        with open(batch_dir / "requests.jsonl", encoding="utf-8") as f:
            lines = [json.loads(l) for l in f if l.strip()]
        assert len(lines) == 1
        assert lines[0]["key"] == "headword:look up"
        le = read_lemma_examples(batch_dir / "lemma_examples.json")
        assert "look up" in le
        assert len(le["look up"]) == 2
        assert "look up" in le["look up"][0]


class TestPhrasalDeckHelpers:
    """Stage 5: helpers for phrasal/MWE deck (candidates + headword examples)."""

    def test_load_mwe_candidates_for_deck_phrasal_filters_type(self, tmp_path: Path):
        mwe_df = pd.DataFrame([
            {"headword": "look up", "type": "phrasal_verb", "count": 5},
            {"headword": "by the way", "type": "fixed_expression", "count": 3},
        ])
        path = tmp_path / "mwe.parquet"
        mwe_df.to_parquet(path, index=False)
        headwords = load_mwe_candidates_for_deck(path, "mwe", top_k=10)
        assert "look up" in headwords and "by the way" in headwords  # mwe mode does not filter type
        headwords_phrasal = load_mwe_candidates_for_deck(path, "phrasal", top_k=10)
        assert headwords_phrasal == ["look up"]  # phrasal filters to type=phrasal_verb only

    def test_load_mwe_candidates_for_deck_top_k(self, tmp_path: Path):
        mwe_df = pd.DataFrame([
            {"headword": "a", "type": "phrasal_verb", "count": 3},
            {"headword": "b", "type": "phrasal_verb", "count": 2},
            {"headword": "c", "type": "phrasal_verb", "count": 1},
        ])
        path = tmp_path / "mwe.parquet"
        mwe_df.to_parquet(path, index=False)
        assert len(load_mwe_candidates_for_deck(path, "phrasal", top_k=2)) == 2
        assert len(load_mwe_candidates_for_deck(path, "phrasal", top_k=10)) == 3

    def test_headword_examples_from_sentences_matches(self):
        sentences = pd.DataFrame({
            "sentence_id": [1, 2, 3],
            "text": ["I look up the word.", "We look up the address.", "No match here."],
        })
        texts, v2 = headword_examples_from_sentences(sentences, ["look up"], max_examples=10)
        assert texts["look up"] == ["I look up the word.", "We look up the address."]
        assert len(v2["look up"]) == 2
        assert v2["look up"][0]["sentence_id"] == 1 and v2["look up"][0]["text"] == "I look up the word."


class TestReadLemmaExamples:
    def test_v1_format_returns_texts(self, tmp_path: Path):
        path = tmp_path / "lemma_examples.json"
        v1 = {"run": ["He runs.", "She runs."], "go": ["Go away."]}
        path.write_text(json.dumps(v1, ensure_ascii=False), encoding="utf-8")
        out = read_lemma_examples(path)
        assert out == v1

    def test_v2_format_returns_texts_in_order(self, tmp_path: Path):
        path = tmp_path / "lemma_examples.json"
        v2 = {"run": [{"sentence_id": 1, "text": "He runs."}, {"sentence_id": 2, "text": "She runs."}]}
        path.write_text(json.dumps(v2, ensure_ascii=False), encoding="utf-8")
        out = read_lemma_examples(path)
        assert out == {"run": ["He runs.", "She runs."]}

    def test_empty_lemma_list_returns_empty(self, tmp_path: Path):
        path = tmp_path / "lemma_examples.json"
        path.write_text(json.dumps({"run": []}, ensure_ascii=False), encoding="utf-8")
        out = read_lemma_examples(path)
        assert out == {"run": []}

    def test_missing_file_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            read_lemma_examples(tmp_path / "missing.json")


class TestReadLemmaPosPerExample:
    """lemma_pos_per_example.json: optional; used for POS mismatch QC gate."""

    def test_missing_file_returns_empty(self, tmp_path: Path):
        out = read_lemma_pos_per_example(tmp_path / "missing.json")
        assert out == {}

    def test_returns_dict_of_lists(self, tmp_path: Path):
        path = tmp_path / "lemma_pos_per_example.json"
        path.write_text(json.dumps({"run": ["VERB", "NOUN"], "go": ["VERB"]}, ensure_ascii=False), encoding="utf-8")
        out = read_lemma_pos_per_example(path)
        assert out == {"run": ["VERB", "NOUN"], "go": ["VERB"]}


class TestParseResults:
    def test_missing_results_raises(self, tmp_path: Path):
        batch_dir = tmp_path / "batch_b"
        batch_dir.mkdir()
        (batch_dir / "lemma_examples.json").write_text("{}", encoding="utf-8")
        config = BatchConfig(
            tokens_path=tmp_path / "t.parquet",
            sentences_path=tmp_path / "s.parquet",
            batch_dir=batch_dir,
            output_cards_path=tmp_path / "cards.json",
        )
        with pytest.raises(FileNotFoundError) as exc_info:
            parse_results(config)
        assert "results" in str(exc_info.value).lower() or "results.jsonl" in str(exc_info.value)

    def test_parses_and_writes_cards_and_log(self, tmp_path: Path):
        batch_dir = tmp_path / "batch_b"
        batch_dir.mkdir()
        lemma_ex = {"run": ["He runs.", "She runs."]}
        write_json(batch_dir / "lemma_examples.json", lemma_ex, ensure_ascii=False)
        # One valid result line
        raw = {
            "cards": [
                {
                    "meaning_id": 1,
                    "definition_en": "move fast",
                    "definition_ru": "бежать",
                    "part_of_speech": "verb",
                    "selected_example_indices": [1, 2],
                    "generated_example": "He runs.",
                }
            ],
            "ignored_indices": [],
            "ignore_reasons": {},
        }
        with open(batch_dir / "results.jsonl", "w", encoding="utf-8") as f:
            f.write(json.dumps({"key": "lemma:run", "response": {"candidates": [{"content": {"parts": [{"text": json.dumps(raw)}]}}]}}, ensure_ascii=False) + "\n")

        config = BatchConfig(
            tokens_path=tmp_path / "t.parquet",
            sentences_path=tmp_path / "s.parquet",
            batch_dir=batch_dir,
            output_cards_path=tmp_path / "cards.json",
        )
        parse_results(config)
        assert (tmp_path / "cards.json").exists()
        assert (batch_dir / "download_log.json").exists()
        cards_data = json.loads((tmp_path / "cards.json").read_text(encoding="utf-8"))
        assert "cards" in cards_data
        assert len(cards_data["cards"]) == 1
        assert cards_data["cards"][0]["lemma"] == "run"

    def test_log_includes_empty_and_fallback_lists(self, tmp_path: Path):
        """One result with out-of-range indices so card has examples_fallback."""
        batch_dir = tmp_path / "batch_b"
        batch_dir.mkdir()
        lemma_ex = {"run": ["He runs.", "She runs."]}
        write_json(batch_dir / "lemma_examples.json", lemma_ex, ensure_ascii=False)
        raw = {
            "cards": [
                {
                    "meaning_id": 1,
                    "definition_en": "move",
                    "definition_ru": "бежать",
                    "part_of_speech": "verb",
                    "selected_example_indices": [5, 6],
                    "generated_example": "He runs.",
                }
            ],
            "ignored_indices": [],
            "ignore_reasons": {},
        }
        with open(batch_dir / "results.jsonl", "w", encoding="utf-8") as f:
            f.write(json.dumps({"key": "lemma:run", "response": {"candidates": [{"content": {"parts": [{"text": json.dumps(raw)}]}}]}}, ensure_ascii=False) + "\n")
        config = BatchConfig(
            tokens_path=tmp_path / "t.parquet",
            sentences_path=tmp_path / "s.parquet",
            batch_dir=batch_dir,
            output_cards_path=tmp_path / "cards.json",
        )
        parse_results(config)
        log = json.loads((batch_dir / "download_log.json").read_text(encoding="utf-8"))
        assert "cards_with_empty_examples" in log or "cards_with_examples_fallback" in log

    def test_lemmas_with_zero_cards_in_log(self, tmp_path: Path):
        """Result line that yields zero cards (empty cards list) is recorded in log."""
        batch_dir = tmp_path / "batch_b"
        batch_dir.mkdir()
        lemma_ex = {"run": ["He runs."], "go": ["Go away."]}
        write_json(batch_dir / "lemma_examples.json", lemma_ex, ensure_ascii=False)
        raw_empty = {"cards": [], "ignored_indices": [], "ignore_reasons": {}}
        with open(batch_dir / "results.jsonl", "w", encoding="utf-8") as f:
            f.write(json.dumps({"key": "lemma:run", "response": {"candidates": [{"content": {"parts": [{"text": json.dumps(raw_empty)}]}}]}}, ensure_ascii=False) + "\n")
        config = BatchConfig(
            tokens_path=tmp_path / "t.parquet",
            sentences_path=tmp_path / "s.parquet",
            batch_dir=batch_dir,
            output_cards_path=tmp_path / "cards.json",
        )
        parse_results(config)
        log = json.loads((batch_dir / "download_log.json").read_text(encoding="utf-8"))
        assert "lemmas_with_zero_cards" in log
        assert "run" in log["lemmas_with_zero_cards"]

    def test_strict_drops_card_lemma_not_in_example(self, tmp_path: Path):
        """Stage 4: strict mode drops card when lemma not in every example; QC-gate then fails (no write)."""
        batch_dir = tmp_path / "batch_b"
        batch_dir.mkdir()
        lemma_ex = {"run": ["The weather is nice."]}
        write_json(batch_dir / "lemma_examples.json", lemma_ex, ensure_ascii=False)
        raw = {
            "cards": [
                {
                    "meaning_id": 1,
                    "definition_en": "move fast",
                    "definition_ru": "бежать",
                    "part_of_speech": "verb",
                    "selected_example_indices": [1],
                    "generated_example": "He runs.",
                }
            ],
            "ignored_indices": [],
            "ignore_reasons": {},
        }
        with open(batch_dir / "results.jsonl", "w", encoding="utf-8") as f:
            f.write(json.dumps({"key": "lemma:run", "response": {"candidates": [{"content": {"parts": [{"text": json.dumps(raw)}]}}]}}, ensure_ascii=False) + "\n")
        config = BatchConfig(
            tokens_path=tmp_path / "t.parquet",
            sentences_path=tmp_path / "s.parquet",
            batch_dir=batch_dir,
            output_cards_path=tmp_path / "cards.json",
            strict=True,
        )
        with pytest.raises(ValueError, match="QC gate FAIL"):
            parse_results(config)
        assert not (tmp_path / "cards.json").exists()

    def test_strict_drops_card_pos_mismatch(self, tmp_path: Path):
        """Stage 5: card claiming POS not in selected examples is dropped; QC-gate then fails (no write)."""
        batch_dir = tmp_path / "batch_d"
        batch_dir.mkdir()
        write_json(batch_dir / "lemma_examples.json", {"run": [{"sentence_id": 1, "text": "He runs."}, {"sentence_id": 2, "text": "A morning run."}]}, ensure_ascii=False)
        write_json(batch_dir / "lemma_pos_per_example.json", {"run": ["NOUN", "NOUN"]}, ensure_ascii=False)
        raw = {
            "cards": [
                {
                    "meaning_id": 1,
                    "definition_en": "move fast",
                    "definition_ru": "бежать",
                    "part_of_speech": "verb",
                    "selected_example_indices": [1, 2],
                    "generated_example": "He runs.",
                }
            ],
            "ignored_indices": [],
            "ignore_reasons": {},
        }
        with open(batch_dir / "results.jsonl", "w", encoding="utf-8") as f:
            f.write(json.dumps({"key": "lemma:run", "response": {"candidates": [{"content": {"parts": [{"text": json.dumps(raw)}]}}]}}, ensure_ascii=False) + "\n")
        config = BatchConfig(
            tokens_path=tmp_path / "t.parquet",
            sentences_path=tmp_path / "s.parquet",
            batch_dir=batch_dir,
            output_cards_path=tmp_path / "cards.json",
            strict=True,
        )
        with pytest.raises(ValueError, match="QC gate FAIL"):
            parse_results(config)
        assert not (tmp_path / "cards.json").exists()

    def test_strict_drops_card_duplicate_sense(self, tmp_path: Path):
        """Stage 6: duplicate sense dropped; QC-gate fails (rate > 0) so no write."""
        batch_dir = tmp_path / "batch_dup"
        batch_dir.mkdir()
        write_json(
            batch_dir / "lemma_examples.json",
            {"run": [{"sentence_id": 1, "text": "He runs."}, {"sentence_id": 2, "text": "A morning run."}]},
            ensure_ascii=False,
        )
        raw = {
            "cards": [
                {
                    "meaning_id": 1,
                    "definition_en": "to move quickly on foot",
                    "definition_ru": "бежать",
                    "part_of_speech": "verb",
                    "selected_example_indices": [1],
                },
                {
                    "meaning_id": 2,
                    "definition_en": "to move quickly on foot.",
                    "definition_ru": "бежать",
                    "part_of_speech": "verb",
                    "selected_example_indices": [2],
                },
            ],
            "ignored_indices": [],
            "ignore_reasons": {},
        }
        with open(batch_dir / "results.jsonl", "w", encoding="utf-8") as f:
            f.write(json.dumps({"key": "lemma:run", "response": {"candidates": [{"content": {"parts": [{"text": json.dumps(raw)}]}}]}}, ensure_ascii=False) + "\n")
        config = BatchConfig(
            tokens_path=tmp_path / "t.parquet",
            sentences_path=tmp_path / "s.parquet",
            batch_dir=batch_dir,
            output_cards_path=tmp_path / "cards.json",
            strict=True,
        )
        with pytest.raises(ValueError, match="QC gate FAIL"):
            parse_results(config)
        assert not (tmp_path / "cards.json").exists()

    def test_strict_drops_card_headword_invalid_for_mode(self, tmp_path: Path):
        """Stage 4b: word mode drops card when headword is multiword; QC-gate then fails (no write)."""
        batch_dir = tmp_path / "batch_hw"
        batch_dir.mkdir()
        write_json(
            batch_dir / "lemma_examples.json",
            {"look": [{"sentence_id": 1, "text": "I look up the word."}]},
            ensure_ascii=False,
        )
        write_json(batch_dir / "lemma_pos_per_example.json", {"look": ["VERB"]}, ensure_ascii=False)
        raw = {
            "cards": [
                {
                    "meaning_id": 1,
                    "lemma": "look",
                    "headword": "look up",
                    "definition_en": "search for",
                    "definition_ru": "искать",
                    "part_of_speech": "verb",
                    "selected_example_indices": [1],
                }
            ],
            "ignored_indices": [],
            "ignore_reasons": {},
        }
        with open(batch_dir / "results.jsonl", "w", encoding="utf-8") as f:
            f.write(json.dumps({"key": "lemma:look", "response": {"candidates": [{"content": {"parts": [{"text": json.dumps(raw)}]}}]}}, ensure_ascii=False) + "\n")
        config = BatchConfig(
            tokens_path=tmp_path / "t.parquet",
            sentences_path=tmp_path / "s.parquet",
            batch_dir=batch_dir,
            output_cards_path=tmp_path / "cards.json",
            strict=True,
            mode="word",
        )
        with pytest.raises(ValueError, match="QC gate FAIL"):
            parse_results(config)
        assert not (tmp_path / "cards.json").exists()


class TestCreateBatch:
    def test_existing_batch_info_raises_without_overwrite(self, tmp_path: Path):
        tok_path, sent_path = _make_tokens_sentences(tmp_path)
        batch_dir = tmp_path / "batch_b"
        batch_dir.mkdir()
        (batch_dir / "batch_info.json").write_text('{"batch_name":"x","lemmas_count":1}', encoding="utf-8")
        config = BatchConfig(
            tokens_path=tok_path,
            sentences_path=sent_path,
            batch_dir=batch_dir,
            output_cards_path=tmp_path / "cards.json",
            limit=1,
        )
        with pytest.raises(FileExistsError, match="already exists"):
            create_batch(config, overwrite=False)

    def test_overwrite_true_allows_create(self, tmp_path: Path):
        tok_path, sent_path = _make_tokens_sentences(tmp_path)
        batch_dir = tmp_path / "batch_b"
        config = BatchConfig(
            tokens_path=tok_path,
            sentences_path=sent_path,
            batch_dir=batch_dir,
            output_cards_path=tmp_path / "cards.json",
            limit=1,
        )
        with patch.object(batch_io.batch_api, "get_client"):
            with patch.object(batch_io.batch_api, "upload_requests_file", return_value="files/abc"):
                with patch.object(batch_io.batch_api, "create_batch_job") as m_create:
                    mock_job = MagicMock()
                    mock_job.name = "batches/xyz"
                    m_create.return_value = mock_job
                    create_batch(config, overwrite=True)
        assert (batch_dir / "batch_info.json").exists()
        info = json.loads((batch_dir / "batch_info.json").read_text(encoding="utf-8"))
        assert info["batch_name"] == "batches/xyz"
        assert info["lemmas_count"] == 1


class TestGetBatchStatus:
    def test_missing_batch_info_raises(self, tmp_path: Path):
        batch_dir = tmp_path / "batch_b"
        batch_dir.mkdir()
        with pytest.raises(FileNotFoundError):
            get_batch_status(batch_dir)

    def test_no_batch_name_returns_no_batch_without_api_call(self, tmp_path: Path):
        batch_dir = tmp_path / "batch_b"
        batch_dir.mkdir()
        (batch_dir / "batch_info.json").write_text(
            '{"batch_name": null, "model": "m", "lemmas_count": 0}', encoding="utf-8"
        )
        status = get_batch_status(batch_dir)
        assert status["state"] == "NO_BATCH"
        assert status["batch_name"] is None
        assert status["lemmas_count"] == 0

    def test_returns_status_from_api(self, tmp_path: Path):
        batch_dir = tmp_path / "batch_b"
        batch_dir.mkdir()
        (batch_dir / "batch_info.json").write_text(
            '{"batch_name":"batches/xyz","model":"m","lemmas_count":5}', encoding="utf-8"
        )
        mock_client = MagicMock()
        mock_job = MagicMock()
        mock_job.state.name = "JOB_STATE_SUCCEEDED"
        mock_job.name = "batches/xyz"
        mock_client.batches.get.return_value = mock_job
        with patch.object(batch_io.batch_api, "get_client", return_value=mock_client):
            status = get_batch_status(batch_dir)
        assert status["state"] == "JOB_STATE_SUCCEEDED"
        assert status["lemmas_count"] == 5


class TestLoadLemmaGroups:
    """load_lemma_groups(config) returns (lemma_groups DataFrame, lemma_examples dict)."""

    def test_returns_groups_and_examples(self, tmp_path: Path):
        tok_path, sent_path = _make_tokens_sentences(tmp_path)
        config = BatchConfig(
            tokens_path=tok_path,
            sentences_path=sent_path,
            batch_dir=tmp_path / "batch",
            output_cards_path=tmp_path / "out.json",
            limit=None,
            max_examples=10,
        )
        groups, examples = load_lemma_groups(config)
        assert not groups.empty
        assert "lemma" in groups.columns and "examples" in groups.columns
        assert set(examples.keys()) == {"go", "run"}
        assert len(examples["run"]) == 2
        assert len(examples["go"]) == 2

    def test_limit_applied(self, tmp_path: Path):
        tok_path, sent_path = _make_tokens_sentences(tmp_path)
        config = BatchConfig(
            tokens_path=tok_path,
            sentences_path=sent_path,
            batch_dir=tmp_path / "batch",
            output_cards_path=tmp_path / "out.json",
            limit=1,
            max_examples=10,
        )
        groups, examples = load_lemma_groups(config)
        assert len(groups) == 1
        assert len(examples) == 1
        assert list(examples.keys())[0] in {"go", "run"}

    def test_max_examples_trims_per_lemma(self, tmp_path: Path):
        tok_path, sent_path = _make_tokens_sentences(tmp_path)
        config = BatchConfig(
            tokens_path=tok_path,
            sentences_path=sent_path,
            batch_dir=tmp_path / "batch",
            output_cards_path=tmp_path / "out.json",
            limit=None,
            max_examples=1,
        )
        _, examples = load_lemma_groups(config)
        assert all(len(ex) == 1 for ex in examples.values())


class TestListRetryCandidates:
    """list_retry_candidates(results_path, lemma_examples_path) returns (lemmas_empty, lemmas_fallback)."""

    def test_returns_empty_and_fallback_sets(self, tmp_path: Path):
        lemma_ex_path = tmp_path / "lemma_examples.json"
        write_json(lemma_ex_path, {"run": ["He runs."], "go": ["Go away."]}, ensure_ascii=False)
        # One result: run with empty examples (indices out of range or empty)
        raw_empty = {
            "cards": [
                {
                    "meaning_id": 1,
                    "definition_en": "move",
                    "definition_ru": "бежать",
                    "part_of_speech": "verb",
                    "selected_example_indices": [],
                    "generated_example": "",
                }
            ],
            "ignored_indices": [],
            "ignore_reasons": {},
        }
        results_path = tmp_path / "results.jsonl"
        with open(results_path, "w", encoding="utf-8") as f:
            f.write(json.dumps({"key": "lemma:run", "response": {"candidates": [{"content": {"parts": [{"text": json.dumps(raw_empty)}]}}]}}, ensure_ascii=False) + "\n")
        lemmas_empty, lemmas_fallback = list_retry_candidates(results_path, lemma_ex_path)
        assert "run" in lemmas_empty or "run" in lemmas_fallback
        assert "go" not in lemmas_empty and "go" not in lemmas_fallback


class TestWaitForBatch:
    def test_missing_batch_info_raises(self, tmp_path: Path):
        batch_dir = tmp_path / "batch_b"
        batch_dir.mkdir()
        with pytest.raises(FileNotFoundError, match="Batch info not found"):
            wait_for_batch(batch_dir)

    def test_no_batch_name_raises(self, tmp_path: Path):
        batch_dir = tmp_path / "batch_b"
        batch_dir.mkdir()
        (batch_dir / "batch_info.json").write_text(
            '{"batch_name": null, "model": "m", "lemmas_count": 0, "created_at": ""}', encoding="utf-8"
        )
        with pytest.raises(ValueError, match="no batch_name"):
            wait_for_batch(batch_dir)

    def test_returns_on_succeeded(self, tmp_path: Path):
        batch_dir = tmp_path / "batch_b"
        batch_dir.mkdir()
        (batch_dir / "batch_info.json").write_text(
            '{"batch_name": "batches/xyz", "model": "m", "lemmas_count": 2, "created_at": ""}', encoding="utf-8"
        )
        mock_job = MagicMock()
        mock_job.state.name = "JOB_STATE_SUCCEEDED"
        mock_job.name = "batches/xyz"
        mock_client = MagicMock()
        mock_client.batches.get.return_value = mock_job
        with patch.object(batch_io.batch_api, "get_client", return_value=mock_client):
            out = wait_for_batch(batch_dir)
        assert out["state"] == "JOB_STATE_SUCCEEDED"
        assert out["lemmas_count"] == 2


class TestDownloadBatch:
    """download_batch with from_file=True and no retry only parses and writes."""

    def test_from_file_no_retry_parses_and_writes(self, tmp_path: Path):
        batch_dir = tmp_path / "batch_b"
        batch_dir.mkdir()
        lemma_ex = {"run": ["He runs.", "She runs."]}
        write_json(batch_dir / "lemma_examples.json", lemma_ex, ensure_ascii=False)
        raw = {
            "cards": [
                {
                    "meaning_id": 1,
                    "definition_en": "move fast",
                    "definition_ru": "бежать",
                    "part_of_speech": "verb",
                    "selected_example_indices": [1, 2],
                    "generated_example": "He runs.",
                }
            ],
            "ignored_indices": [],
            "ignore_reasons": {},
        }
        with open(batch_dir / "results.jsonl", "w", encoding="utf-8") as f:
            f.write(json.dumps({"key": "lemma:run", "response": {"candidates": [{"content": {"parts": [{"text": json.dumps(raw)}]}}]}}, ensure_ascii=False) + "\n")
        output_cards = tmp_path / "cards.json"
        config = BatchConfig(
            tokens_path=tmp_path / "t.parquet",
            sentences_path=tmp_path / "s.parquet",
            batch_dir=batch_dir,
            output_cards_path=output_cards,
        )
        download_batch(config, from_file=True, retry_empty=False, retry_thinking=False)
        assert output_cards.exists()
        data = json.loads(output_cards.read_text(encoding="utf-8"))
        assert len(data["cards"]) == 1
        assert data["cards"][0]["lemma"] == "run"
        assert (batch_dir / "download_log.json").exists()

    def test_missing_lemma_examples_raises(self, tmp_path: Path):
        batch_dir = tmp_path / "batch_b"
        batch_dir.mkdir()
        config = BatchConfig(
            tokens_path=tmp_path / "t.parquet",
            sentences_path=tmp_path / "s.parquet",
            batch_dir=batch_dir,
            output_cards_path=tmp_path / "cards.json",
        )
        with pytest.raises(FileNotFoundError, match="Lemma examples"):
            download_batch(config, from_file=True)

    def test_from_file_false_batch_info_no_batch_name_raises(self, tmp_path: Path):
        batch_dir = tmp_path / "batch_b"
        batch_dir.mkdir()
        write_json(batch_dir / "lemma_examples.json", {"run": ["He runs."]}, ensure_ascii=False)
        (batch_dir / "batch_info.json").write_text(
            '{"batch_name": null, "model": "m", "lemmas_count": 1, "created_at": ""}', encoding="utf-8"
        )
        config = BatchConfig(
            tokens_path=tmp_path / "t.parquet",
            sentences_path=tmp_path / "s.parquet",
            batch_dir=batch_dir,
            output_cards_path=tmp_path / "cards.json",
        )
        with pytest.raises(ValueError, match="no batch_name"):
            download_batch(config, from_file=False)

    def test_from_file_false_batch_not_succeeded_raises(self, tmp_path: Path):
        batch_dir = tmp_path / "batch_b"
        batch_dir.mkdir()
        write_json(batch_dir / "lemma_examples.json", {"run": ["He runs."]}, ensure_ascii=False)
        (batch_dir / "batch_info.json").write_text(
            '{"batch_name": "batches/xyz", "model": "m", "lemmas_count": 1, "created_at": ""}', encoding="utf-8"
        )
        config = BatchConfig(
            tokens_path=tmp_path / "t.parquet",
            sentences_path=tmp_path / "s.parquet",
            batch_dir=batch_dir,
            output_cards_path=tmp_path / "cards.json",
        )
        with patch.object(batch_io.batch_api, "get_client"):
            with patch.object(batch_io, "get_batch_status", return_value={"state": "JOB_STATE_RUNNING", "lemmas_count": 1}):
                with pytest.raises(RuntimeError, match="Batch not complete"):
                    download_batch(config, from_file=False)

    def test_from_file_false_downloads_then_parses(self, tmp_path: Path):
        """With mocks: batch_info exists, status SUCCEEDED, download returns results content."""
        batch_dir = tmp_path / "batch_b"
        batch_dir.mkdir()
        lemma_ex = {"run": ["He runs.", "She runs."]}
        write_json(batch_dir / "lemma_examples.json", lemma_ex, ensure_ascii=False)
        raw = {
            "cards": [
                {
                    "meaning_id": 1,
                    "definition_en": "move fast",
                    "definition_ru": "бежать",
                    "part_of_speech": "verb",
                    "selected_example_indices": [1, 2],
                    "generated_example": "He runs.",
                }
            ],
            "ignored_indices": [],
            "ignore_reasons": {},
        }
        results_content = (json.dumps({"key": "lemma:run", "response": {"candidates": [{"content": {"parts": [{"text": json.dumps(raw)}]}}]}}, ensure_ascii=False) + "\n").encode("utf-8")
        (batch_dir / "batch_info.json").write_text(
            '{"batch_name": "batches/xyz", "model": "m", "lemmas_count": 1, "created_at": ""}', encoding="utf-8"
        )
        output_cards = tmp_path / "cards.json"
        config = BatchConfig(
            tokens_path=tmp_path / "t.parquet",
            sentences_path=tmp_path / "s.parquet",
            batch_dir=batch_dir,
            output_cards_path=output_cards,
        )
        mock_client = MagicMock()
        with patch.object(batch_io.batch_api, "get_client", return_value=mock_client):
            with patch.object(batch_io.batch_api, "download_batch_results", return_value=results_content):
                status = {"state": "JOB_STATE_SUCCEEDED", "batch_name": "batches/xyz", "lemmas_count": 1}
                with patch.object(batch_io, "get_batch_status", return_value=status):
                    download_batch(config, from_file=False, retry_empty=False, retry_thinking=False)
        assert (batch_dir / "results.jsonl").exists()
        assert output_cards.exists()
        data = json.loads(output_cards.read_text(encoding="utf-8"))
        assert len(data["cards"]) == 1
        assert data["cards"][0]["lemma"] == "run"

    def test_retry_empty_uses_cache_or_api_and_merges(self, tmp_path: Path):
        """from_file=True, one card with empty examples; retry via cache or mock API merges new cards.
        Uses strict=False so the card with empty examples is kept (for retry merge test); strict=True would drop it."""
        batch_dir = tmp_path / "batch_b"
        batch_dir.mkdir()
        lemma_ex = {"run": ["He runs.", "She runs."]}
        write_json(batch_dir / "lemma_examples.json", lemma_ex, ensure_ascii=False)
        # Result that produces one card with no examples (empty selected_example_indices)
        raw_empty = {
            "cards": [
                {
                    "meaning_id": 1,
                    "definition_en": "move fast",
                    "definition_ru": "бежать",
                    "part_of_speech": "verb",
                    "selected_example_indices": [],
                    "generated_example": "He runs.",
                }
            ],
            "ignored_indices": [],
            "ignore_reasons": {},
        }
        with open(batch_dir / "results.jsonl", "w", encoding="utf-8") as f:
            f.write(json.dumps({"key": "lemma:run", "response": {"candidates": [{"content": {"parts": [{"text": json.dumps(raw_empty)}]}}]}}, ensure_ascii=False) + "\n")
        output_cards = tmp_path / "cards.json"
        config = BatchConfig(
            tokens_path=tmp_path / "t.parquet",
            sentences_path=tmp_path / "s.parquet",
            batch_dir=batch_dir,
            output_cards_path=output_cards,
            strict=False,
        )
        download_batch(config, from_file=True, retry_empty=False, retry_thinking=False)
        assert len(json.loads(output_cards.read_text(encoding="utf-8"))["cards"]) == 1
        # Now retry with mock: call_standard_retry returns a response that has valid cards with examples
        raw_fixed = {
            "cards": [
                {
                    "meaning_id": 1,
                    "definition_en": "move fast",
                    "definition_ru": "бежать",
                    "part_of_speech": "verb",
                    "selected_example_indices": [1, 2],
                    "generated_example": "He runs.",
                }
            ],
            "ignored_indices": [],
            "ignore_reasons": {},
        }
        mock_resp = {"candidates": [{"content": {"parts": [{"text": json.dumps(raw_fixed)}]}}]}
        with patch.object(batch_io.batch_api, "get_client"):
            with patch.object(batch_io.batch_api, "call_standard_retry", return_value=mock_resp):
                download_batch(config, from_file=True, retry_empty=True, retry_thinking=False)
        data = json.loads(output_cards.read_text(encoding="utf-8"))
        assert len(data["cards"]) == 1
        assert data["cards"][0].get("examples")  # retry replaced with card that has examples
        log = json.loads((batch_dir / "download_log.json").read_text(encoding="utf-8"))
        assert "retry_log" in log
        assert any(e.get("outcome") == "success" for e in log["retry_log"])
        # 6.3: structured retry reasons (lemma, reason, outcome)
        for e in log["retry_log"]:
            assert "lemma" in e and "reason" in e and "outcome" in e

    def test_download_batch_calls_qc_threshold_when_strict_and_thresholds_set(self, tmp_path: Path):
        """When config has strict=True and max_warning_rate/max_warnings_absolute, check_qc_threshold is used."""
        batch_dir = tmp_path / "batch_b"
        batch_dir.mkdir()
        lemma_ex = {"run": ["He runs.", "She runs."]}
        write_json(batch_dir / "lemma_examples.json", lemma_ex, ensure_ascii=False)
        raw = {
            "cards": [
                {
                    "meaning_id": 1,
                    "definition_en": "move fast",
                    "definition_ru": "бежать",
                    "part_of_speech": "verb",
                    "selected_example_indices": [1, 2],
                    "generated_example": "He runs.",
                }
            ],
            "ignored_indices": [],
            "ignore_reasons": {},
        }
        with open(batch_dir / "results.jsonl", "w", encoding="utf-8") as f:
            f.write(json.dumps({"key": "lemma:run", "response": {"candidates": [{"content": {"parts": [{"text": json.dumps(raw)}]}}]}}, ensure_ascii=False) + "\n")
        config = BatchConfig(
            tokens_path=tmp_path / "t.parquet",
            sentences_path=tmp_path / "s.parquet",
            batch_dir=batch_dir,
            output_cards_path=tmp_path / "cards.json",
            strict=True,
            max_warning_rate=0.5,
            max_warnings_absolute=10,
        )
        with patch.object(batch_io.batch_api, "get_client"):
            download_batch(config, from_file=True, retry_empty=True, retry_thinking=False)
        assert (tmp_path / "cards.json").exists()
        log = json.loads((batch_dir / "download_log.json").read_text(encoding="utf-8"))
        assert "cards_lemma_not_in_example_count" in log
        assert "retry_log" in log

    def test_strict_download_batch_qc_gate_fail_raises_no_write(self, tmp_path: Path):
        """6.1: When strict and validation_errors present, download_batch raises before writing cards."""
        batch_dir = tmp_path / "batch_gate"
        batch_dir.mkdir()
        write_json(
            batch_dir / "lemma_examples.json",
            {"look": [{"sentence_id": 1, "text": "I look up the word."}]},
            ensure_ascii=False,
        )
        write_json(batch_dir / "lemma_pos_per_example.json", {"look": ["VERB"]}, ensure_ascii=False)
        raw = {
            "cards": [
                {
                    "meaning_id": 1,
                    "lemma": "look",
                    "headword": "look up",
                    "definition_en": "search for",
                    "definition_ru": "искать",
                    "part_of_speech": "verb",
                    "selected_example_indices": [1],
                }
            ],
            "ignored_indices": [],
            "ignore_reasons": {},
        }
        with open(batch_dir / "results.jsonl", "w", encoding="utf-8") as f:
            f.write(json.dumps({"key": "lemma:look", "response": {"candidates": [{"content": {"parts": [{"text": json.dumps(raw)}]}}]}}, ensure_ascii=False) + "\n")
        (batch_dir / "batch_info.json").write_text(
            '{"batch_name": "batches/x", "model": "m", "lemmas_count": 1, "created_at": ""}', encoding="utf-8"
        )
        output_cards = tmp_path / "cards.json"
        config = BatchConfig(
            tokens_path=tmp_path / "t.parquet",
            sentences_path=tmp_path / "s.parquet",
            batch_dir=batch_dir,
            output_cards_path=output_cards,
            strict=True,
            mode="word",
        )
        with pytest.raises(ValueError, match="QC gate FAIL"):
            download_batch(config, from_file=True, retry_empty=False, retry_thinking=False)
        assert not output_cards.exists()

    def test_retry_includes_lemmas_from_validation_errors_with_structured_reason(self):
        """6.2/6.3: get_retry_candidates_and_reasons includes lemmas from pos_mismatch/lemma_not_in_example/validation."""
        from eng_words.word_family.batch_io import get_retry_candidates_and_reasons

        all_cards: list[dict] = []
        validation_errors = [
            {"lemma": "run", "stage": "download", "error_type": "pos_mismatch", "message": "dropped"},
        ]
        lemmas_to_retry, lemma_reason = get_retry_candidates_and_reasons(all_cards, validation_errors)
        assert "run" in lemmas_to_retry
        assert lemma_reason.get("run") == "pos_mismatch"

        validation_errors.append({"lemma": "go", "stage": "download", "error_type": "lemma_not_in_example", "message": "dropped"})
        lemmas_to_retry2, lemma_reason2 = get_retry_candidates_and_reasons(all_cards, validation_errors)
        assert "run" in lemmas_to_retry2 and "go" in lemmas_to_retry2
        assert lemma_reason2["run"] == "pos_mismatch" and lemma_reason2["go"] == "lemma_not_in_example"

        # duplicate_sense / headword_invalid_for_mode are not retry reasons
        all_cards3 = [{"lemma": "x", "examples": []}]
        validation_errors3 = [{"lemma": "x", "error_type": "duplicate_sense", "message": "drop"}]
        lemmas3, reasons3 = get_retry_candidates_and_reasons(all_cards3, validation_errors3)
        assert "x" in lemmas3  # from empty examples
        assert reasons3["x"] == "empty_or_fallback"


class TestRetryCache:
    """Retry cache: key = lemma + mode + prompt_hash; cache hit avoids network."""

    def test_load_missing_returns_empty(self, tmp_path: Path):
        assert load_retry_cache(tmp_path / "missing.jsonl") == {}

    def test_append_and_load_roundtrip(self, tmp_path: Path):
        path = tmp_path / "retry_cache.jsonl"
        append_retry_cache_entry(path, "run:standard:abc", {"candidates": [{"content": {"parts": [{"text": "{}"}]}}]})
        append_retry_cache_entry(path, "go:standard:def", {"candidates": []})
        cache = load_retry_cache(path)
        assert "run:standard:abc" in cache
        assert "go:standard:def" in cache
        assert cache["run:standard:abc"]["candidates"][0]["content"]["parts"][0]["text"] == "{}"
