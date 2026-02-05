"""Pipeline B (Word Family) Batch API — facade.

Подмодули:
- **batch_core**: чистая логика (build_prompt, parse_one_result, validate_card, filter_valid_cards, merge_retry_results).
- **batch_io**: файлы и оркестрация (render_requests, parse_results, load_lemma_groups, create_batch, download_batch, list_retry_candidates, wait_for_batch).
- **batch_api**: Gemini (get_client, upload, create batch job, download results, call_standard_retry).
- **batch_qc**: проверки качества (cards_lemma_not_in_example, check_qc_threshold).
- **batch_schemas**: контракты (BatchConfig, BatchPaths, BatchInfo, ErrorEntry).

Артефакты (по умолчанию в data/experiment/ и data/experiment/batch_b/):
- requests.jsonl, lemma_examples.json — вход для Batch API (создаются render_requests).
- batch_info.json — имя job, модель, lemmas_count (создаётся create_batch).
- results.jsonl — ответы Batch API (пишется при download).
- retry_cache.jsonl — кэш ответов Standard API для retry (переиспользуется при повторных download).
- download_log.json — лог download (retry_log, cards_lemma_not_in_example, errors).
- cards_B_batch.json — итоговые карточки (pipeline, stats, cards, validation_errors).

State machine:
  1. render_requests(config) → requests.jsonl + lemma_examples.json
  2. create_batch(config) → upload → batch job → batch_info.json
  3. status / wait_for_batch → опрос до JOB_STATE_SUCCEEDED | FAILED | CANCELLED
  4. download_batch(config) → (опционально скачать results) → parse_results → опционально retry (empty/fallback) → QC → cards + download_log

Offline: render_requests + parse_results (при наличии results.jsonl) не требуют сети; list_retry_candidates — только чтение файлов.

Strict / relaxed (BatchConfig):
- **strict=True** (по умолчанию): контрактные ошибки и большинство QC → исключение. При заданных порогах (max_warning_rate, max_warnings_absolute) проверка «lemma not in example» вызывает check_qc_threshold: превышение → ValueError.
- **relaxed**: часть QC записывается как предупреждения, но пороги по-прежнему применяются; превышение порога → ошибка.

Пороги QC:
- max_warning_rate (доля, например 0.01 = 1%): issue_count / total_count не должна превышать.
- max_warnings_absolute (целое, например 50): issue_count не должен превышать.
- Метрика «lemma not in example»: карточки, у которых хотя бы один пример не содержит лемму (или допустимую словоформу); при strict и заданных порогах проверка выполняется в download_batch после merge retry.
"""

from __future__ import annotations

from pathlib import Path

from eng_words.word_family import batch_api
from eng_words.word_family.batch_core import (
    build_prompt,
    build_retry_prompt,
    filter_valid_cards,
    parse_one_result as _parse_one_result,
    validate_card as _validate_card,
)
from eng_words.word_family.batch_io import load_lemma_groups
from eng_words.word_family.batch_qc import cards_lemma_not_in_example
from eng_words.word_family.batch_schemas import BatchPaths, BatchConfig

# Default paths (project root = src/eng_words/word_family/../../../..)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DATA_DIR = _PROJECT_ROOT / "data" / "experiment"
BATCH_DIR = DATA_DIR / "batch_b"

_paths = BatchPaths.from_dir(BATCH_DIR)
BATCH_INFO_PATH = _paths.batch_info
REQUESTS_PATH = _paths.requests
RESULTS_PATH = _paths.results
LEMMA_EXAMPLES_PATH = _paths.lemma_examples
TOKENS_PATH = DATA_DIR / "tokens_sample.parquet"
SENTENCES_PATH = DATA_DIR / "sentences_sample.parquet"
OUTPUT_CARDS_PATH = DATA_DIR / "cards_B_batch.json"

# Re-export for tests and CLI
get_client = batch_api.get_client

__all__ = [
    "BATCH_DIR",
    "BATCH_INFO_PATH",
    "DATA_DIR",
    "LEMMA_EXAMPLES_PATH",
    "OUTPUT_CARDS_PATH",
    "REQUESTS_PATH",
    "RESULTS_PATH",
    "SENTENCES_PATH",
    "TOKENS_PATH",
    "BatchConfig",
    "BatchPaths",
    "build_prompt",
    "build_retry_prompt",
    "cards_lemma_not_in_example",
    "filter_valid_cards",
    "get_client",
    "load_lemma_groups",
    "_parse_one_result",
    "_validate_card",
]
