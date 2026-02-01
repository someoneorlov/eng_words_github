# История: Pipeline A — что было и почему удалён

Этот документ описывает **Pipeline A** (WSD + Synset + SmartCardGenerator), который был полностью удалён из репозитория. Вся логика и код A сохранены в **истории git**: коммит перед удалением содержит полное состояние. Здесь — краткое резюме для понимания контекста и поиска в истории.

---

## Что такое Pipeline A

**Идея:** Сначала размечаем смыслы слов (WSD), агрегируем по synset, потом для каждой пары (lemma, synset) генерируем одну умную карточку через LLM (SmartCardGenerator).

**Цепочка:**
1. Stage 1 с `--enable-wsd` → токены + разметка смыслов (supersense)
2. Synset Aggregation → группировка по (lemma, synset), список карточек для генерации
3. Card Generation → один вызов LLM на карточку (definition, examples, quality)

**Модули (удалены):**
- `eng_words.llm.smart_card_generator` — генерация карточек по (lemma, synset)
- `eng_words.aggregation` — synset_aggregator, llm_aggregator
- `eng_words.validation.synset_validator` — валидация примеров для synset-групп
- `eng_words.wsd.llm_wsd` — LLM WSD и redistribute_empty_cards (карточки без примеров)

**Скрипты (удалены):** всё, что лежало в `legacy/pipeline_a/scripts/` (run_full_synset_aggregation.py, run_synset_card_generation.py, эксперименты run_pipeline_a/b/c и др.).

---

## Эксперимент A vs B vs C (2026-01)

Сравнивали три подхода:

| ID | Подход | Описание |
|----|--------|----------|
| A | Текущий Pipeline | WSD → Synset Aggregation → Card Generation |
| B | Word Family | Группировка по лемме → LLM кластеризация (1–3 карточки на лемму) |
| C | Word Family + hints | То же, что B + WordNet definitions как подсказки |

### Результаты (выборка ~2000 предложений)

| Pipeline | Cards | Lemmas | Cost | Time |
|----------|-------|--------|------|------|
| **A** | 902 | 645 | $1.77 | 6.7h |
| **B** | 4,515 | 3,229 | $2.89 | ~1.5h |
| **C** | 148 | 145 | $0.005 | 7m (rate limit) |

**Coverage:** Pipeline A — 19.7%, Pipeline B — **98.8%**.

**Quality (20 лемм из пересечения):** A лучше 50%, B лучше 35%, Tie 15%.

### Проблемы Pipeline A

- Низкое покрытие: строгие фильтры WSD отсекают много content words.
- Пропуски значений (realization, stone, story и т.д.).
- Баги: дубликаты карточек (pause), попадание function words (still, however).
- «Only A» (45 лемм) — в основном function words, не нужные для карточек.

### Вердикт

**Pipeline B выиграл:** покрытие в разы выше, приемлемое качество, меньше багов, быстрее и дешевле на лемму. Решение: использовать **Pipeline B** (Word Family, Batch API) как единственный путь генерации карточек; Pipeline A удалён.

Подробное описание эксперимента и метрик — в `docs/WORD_FAMILY_PIPELINE.md` (разделы «История: Эксперимент A vs B» и «Вердикт: Pipeline B WINS»).

---

## Что именно удалено (для поиска в git)

- **Папка:** `legacy/pipeline_a/` (скрипты и README)
- **Модули в `src/eng_words/`:**
  - `llm/smart_card_generator.py`
  - `aggregation/` (llm_aggregator.py, synset_aggregator.py, __init__.py)
  - `validation/synset_validator.py`
  - `wsd/llm_wsd.py`
- **Тесты:**  
  `test_smart_card_generator.py`, `test_run_synset_card_generation_integration.py`,  
  `test_llm_aggregator.py`, `test_synset_aggregator.py`, `test_synset_validator.py`,  
  `test_llm_wsd.py`, `scripts/_archive/test_validation_on_gold.py`
- **Функции/опции:**  
  В `pipeline.py` удалены `--smart-cards`, `generate_smart_cards()`.  
  Позже из Stage 1 удалён и **WSD**: параметры `enable_wsd`, `wsd_checkpoint_interval`, `min_sense_freq`, `max_senses`, CLI `--enable-wsd` и т.д., блок с `WordNetSenseBackend`, выходы `sense_tokens`/`supersense_stats`. WSD использовался только Pipeline A; пакет `eng_words.wsd` оставлен (word_family/clusterer, eval/benchmark).  
  В `validation/example_validator.py` удалены `validate_card_examples`, `fix_invalid_cards` (использовались только A).  
  В `constants/paths.py` удалены пути для synset aggregation и card generation (DATA_SYNSET_*, get_aggregated_cards_path, get_smart_cards_*, и т.д.).

---

## Как найти код Pipeline A в истории

В коммите, где выполнено удаление, в сообщении явно указано что-то вроде:  
**«Remove Pipeline A (WSD + Synset + SmartCardGenerator); see docs/HISTORY_PIPELINE_A.md»**.

- Чтобы увидеть **состояние репозитория до удаления** (когда Pipeline A ещё был в коде):  
  `git show <commit>^:src/eng_words/llm/smart_card_generator.py`  
  или проверка родительского коммита:  
  `git checkout <commit>^`
- Чтобы посмотреть **список удалённых файлов** в том коммите:  
  `git diff --name-status <commit>^ <commit>`

Так можно в любой момент вернуться к полной версии кода и скриптов Pipeline A.

---

## Сообщение коммита при удалении

Рекомендуемое сообщение коммита, чтобы легко найти его в истории:

```
Remove Pipeline A (WSD + Synset + SmartCardGenerator); see docs/HISTORY_PIPELINE_A.md

- Delete legacy/pipeline_a/, smart_card_generator, aggregation/, synset_validator, llm_wsd
- Remove --smart-cards and generate_smart_cards from pipeline.py
- Remove validate_card_examples, fix_invalid_cards from example_validator (A-only)
- Remove synset/card paths from constants; update validation/__init__, wsd/__init__
- Remove one-off script create_mini_sample_b.py
- Add docs/HISTORY_PIPELINE_A.md (what was removed, why, how to find in git)
- Update PIPELINES_OVERVIEW, GENERATION_INSTRUCTIONS
```
