# Pipeline B — Quality gate (sample runs)

После каждого sample-прогона Pipeline B рекомендуется запускать проверку качества и фиксировать результат.

**Пороги и правила QC:** см. `docs/PIPELINE_B_QUALITY_DECISIONS.md`.

## Команды

Из корня репозитория:

```bash
# Прогон Pipeline B на sample (например limit 50)
uv run python scripts/run_pipeline_b_batch.py create --limit 50
# ... wait + download ...

# QC gate (PASS/FAIL по validation_errors): exit 1 при FAIL
uv run python scripts/run_quality_investigation.py --gate --cards data/experiment/cards_B_batch.json --output data/experiment/qc_gate_report.md
# Или после download: uv run python scripts/run_pipeline_b_batch.py download --run-gate

# Quality investigation: проверки A2, A4, B1–B4, E1, F1/F2, G1 (без --gate)
uv run python scripts/run_quality_investigation.py --cards data/experiment/cards_B_batch.json
# Отчёт: data/experiment/investigation_report.md
```

Опционально: `scripts/check_quality_b_batch.py` для дополнительных проверок.

## Что фиксировать в PR/логах

- Количество карточек, ошибок парсинга, retry, `cards_lemma_not_in_example` (из `download_log.json`).
- Результат `run_quality_investigation.py`: стало лучше/хуже по сравнению с baseline (см. `docs/pipeline_b_refactor_baseline.md`).

## Пороги QC

- **Gate (после прогона):** `qc_gate.QCGateThresholds` — max_lemma_not_in_example_rate, max_pos_mismatch_rate, max_duplicate_sense_rate и др.; по умолчанию 0.0. Скрипт `--gate` считает доли по `validation_errors` и выходит с кодом 1 при превышении.
- **В strict при download:** невалидные карточки (lemma/headword not in example, pos_mismatch, duplicate_sense) не попадают в output и пишутся в `validation_errors`; при `BatchConfig.max_warning_rate` / `max_warnings_absolute` пайплайн может падать при превышении.
