# Pipeline B — Quality gate (sample runs)

После каждого sample-прогона Pipeline B рекомендуется запускать проверку качества и фиксировать результат.

## Команды

Из корня репозитория:

```bash
# Прогон Pipeline B на sample (например limit 50)
uv run python scripts/run_pipeline_b_batch.py create --limit 50
# ... wait + download ...

# Quality investigation: проверки A2, A4, B1–B4, E1, F1/F2, G1
uv run python scripts/run_quality_investigation.py
# Отчёт: data/experiment/investigation_report.md

# Или с указанием пути к карточкам
uv run python scripts/run_quality_investigation.py --cards data/experiment/cards_B_batch.json
```

Опционально: `scripts/check_quality_b_batch.py` для дополнительных проверок.

## Что фиксировать в PR/логах

- Количество карточек, ошибок парсинга, retry, `cards_lemma_not_in_example` (из `download_log.json`).
- Результат `run_quality_investigation.py`: стало лучше/хуже по сравнению с baseline (см. `docs/pipeline_b_refactor_baseline.md`).

## Пороги QC (lemma_not_in_example)

В strict-mode при заданных порогах в `BatchConfig` (`max_warning_rate`, `max_warnings_absolute`) пайплайн падает при превышении. Без заданных порогов проверка только пишется в `download_log.json` (`cards_lemma_not_in_example`).
