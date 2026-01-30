# План безопасной уборки и рефакторинга

> **Статус:** ✅ ЗАВЕРШЁН (2026-01-21)  
> **Цель:** навести порядок в репозитории, не потерять артефакты и не сломать pipeline.

---

## Итоги рефакторинга

### Что было сделано

| Компонент | Было | Стало | Изменение |
|-----------|------|-------|-----------|
| Python файлов (без archive) | 150 | 110 | -40 |
| Скриптов | 51 | 10 | -41 |
| Документов | 60+ | 12 | -48 |
| Data | ~200 MB | 156 MB | -44 MB |

### Коммиты (8 штук)

```
190fa5f chore: archive 41 more scripts, cleanup temp data
60684fb add cursor rules
fbdf51c chore: update .gitignore with temp files
7e5b06a test: add tests for new modules
453f3c5 feat: add smart card generation pipeline with Stage 2.5 filtering
93e5171 chore: cleanup tracked files, update modules
063e840 docs: archive 55 outdated documents, add new plans
e263b21 refactor: archive deprecated modules and scripts
```

---

## Финальная структура проекта

### Скрипты (10 штук)

| Скрипт | Назначение |
|--------|------------|
| `run_synset_card_generation.py` | Основной pipeline генерации карточек |
| `compare_cards.py` | Сравнение результатов (регресс-тесты) |
| `run_full_generation.sh` | Запуск полной генерации |
| `check_test_progress.sh` | Мониторинг прогресса |
| `monitor_generation.sh` | Мониторинг генерации |
| `run_gold_labeling.py` | Labeling для Golden Dataset |
| `eval_wsd_on_gold.py` | Оценка WSD на Golden Dataset |
| `freeze_gold_dataset.py` | Фиксация Golden Dataset |
| `verify_gold_checksum.py` | Проверка checksum |
| `benchmark_wsd.py` | Бенчмарк WSD |

### Документация (12 штук)

| Документ | Назначение |
|----------|------------|
| `DEVELOPMENT_HISTORY.md` | История разработки |
| `QUALITY_FILTERING_PLAN.md` | План улучшения качества карточек |
| `REFACTOR_AND_BACKUP_PLAN.md` | Этот план (завершён) |
| `BACKLOG_IDEAS.md` | Идеи на будущее |
| `GENERATION_INSTRUCTIONS.md` | Инструкции по генерации |
| `WSD_GOLD_DATASET_USAGE.md` | Использование Golden Dataset |
| `CREDENTIALS_EXPLANATION.md` | Объяснение credentials |
| `GOOGLE_SHEETS_SETUP.md` | Setup Google Sheets |
| `LLM_API_KEYS_SETUP.md` | Setup API ключей |
| `claude_pricing.md` | Справочник цен Claude |
| `google_pricing.md` | Справочник цен Gemini |
| `openai_pricing.txt` | Справочник цен OpenAI |

### Архивы

- `src/eng_words/_archive/` — устаревшие модули (cache.py, card_generator.py, evaluator.py, prompts.py, fallback.py)
- `scripts/_archive/` — 53 одноразовых скрипта
- `tests/_archive/` — тесты для архивированных модулей
- `docs/archive/2026-01-21/` — 55 завершённых документов

---

## Регресс-тесты

### Как запускать

**Replay (без LLM-вызовов, с кэшем):**
```bash
rm -f data/synset_cards/synset_smart_cards_*.json
uv run python scripts/run_synset_card_generation.py 100
uv run python scripts/compare_cards.py \
  --expected backups/2026-01-19/benchmark_100/ \
  --actual data/synset_cards/
```

**Эталон:** `backups/2026-01-19/benchmark_100/` — 54 карточки

### Результаты последней проверки

- ✅ Unit-тесты: 697 passed
- ✅ Replay 100: 54 карточки идентичны эталону
- ✅ Live smoke 10: 5 карточек, 100% с примерами

---

## Следующие шаги

См. `QUALITY_FILTERING_PLAN.md` — осталось:
1. Запустить на полном датасете (7,872 карточки)
2. Анализ результатов
3. Финальная документация
