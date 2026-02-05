# Pipeline B — Decision points (качество и QC)

Документ фиксирует решения, чтобы агент и разработчик не дрейфовали. Соответствует этапу 8 плана `PIPELINE_B_FIXES_PLAN.MD`.

---

## 1. Правило “each example must contain target”

**Что проверяем:** в **каждом** примере карточки должен присутствовать “target” (лемма, словоформа или headword).

- **Без headword:** target = лемма или её допустимая форма (получаем через `example_validator._get_word_forms(lemma)`); проверка через `word_in_text_for_matching(form, example)` (нормализация + контракции только для matching).
- **С headword:** target = headword; проверка через `headword_in_examples(card)` — headword должен встречаться в **каждом** примере (нормализованная подстрока).
- **Режимы:** в **strict** карточки, не прошедшие проверку, **не попадают в output** и попадают в `validation_errors` с `error_type=lemma_not_in_example`. В relaxed поведение задаётся порогами `max_warning_rate` / `max_warnings_absolute`.

**Где в коде:** `batch_qc.cards_lemma_not_in_example`, `get_cards_failing_lemma_in_example`; в `parse_results` / `download_batch` при `config.strict` эти карточки удаляются и пишутся в `validation_errors`.

---

## 2. Contractions strategy (только для matching)

**Решение:** контракции и нормализация используются **только для сопоставления** (matching), а не для изменения исходных текстов или данных Stage 1.

- **Не делаем:** не меняем текст примеров в карточках, не меняем данные в Stage 1 (tokens/sentences), не подставляем “do not” вместо “don’t” в output.
- **Делаем:** при проверке “lemma/headword в примере” используем слой нормализации и раскрытия контракций только внутри функции сравнения: `eng_words.text_norm` — `normalize_for_matching()`, `expand_contractions_for_matching()`, `word_in_text_for_matching()` (и при необходимости `match_target_in_text()`). Исходный текст карточки остаётся как пришёл от LLM.

**Где в коде:** `text_norm.py`; в QC вызывается только из `batch_qc` при проверке “target in example”.

---

## 3. Headword: контракт и источник правды

- **Источник правды:** headword приходит **от LLM** в поле `headword` карточки. Stage 1 (tokens/sentences) headword не задаёт.
- **Контракт:**
  - Headword **опционален**. Если его нет, QC проверяет лемму/формы в примерах.
  - Если headword есть, он должен встречаться в **каждом** выбранном примере (нормализованная подстрока). Иначе в strict карточка отбрасывается, в `validation_errors` — `lemma_not_in_example` (или явная запись про headword).
  - В `parse_one_result`: если LLM вернул headword, но он не встречается в примерах, поле **headword удаляется** из карточки (не пишем в output), чтобы не сохранять “придуманный” headword.
- **Назначение:** phrasal verbs (“look up”), fixed expressions, каноническая поверхностная форма для MWE.

**Где в коде:** `headword.py` — `infer_headword(card)`, `headword_in_examples(card)`; `batch_core.parse_one_result` (удаление невалидного headword); `batch_qc` при наличии headword проверяет его в примерах.

---

## 4. QC thresholds и strict/relaxed semantics

- **Единое место порогов (gate):** `eng_words.word_family.qc_gate.QCGateThresholds` и `DEFAULT_QC_GATE_THRESHOLDS`. Пороги задают максимально допустимые **доли** (rates) по типам ошибок: `max_lemma_not_in_example_rate`, `max_pos_mismatch_rate`, `max_duplicate_sense_rate`, `max_validation_rate`, `max_other_rate`. По умолчанию все **0.0** (строгий режим: никаких QC-дропов не допускаем).
- **Strict (по умолчанию):** при прогоне пайплайна невалидные карточки (lemma/headword not in example, pos_mismatch, duplicate_sense, validation) **не попадают в output**, записываются в `validation_errors`. Gate (`evaluate_gate`) считает rate = `validation_errors_count / (cards_generated + validation_errors_count)` по каждому `error_type`; если какой-то rate выше порога → **FAIL**, exit code 1 при `--gate` или `--run-gate`.
- **Relaxed:** разрешён только явно; часть сбоев может учитываться как warnings, но с порогами `max_warning_rate` и `max_warnings_absolute` в `BatchConfig`; превышение порога → **error** даже в relaxed.
- **Запуск gate:** `scripts/run_quality_investigation.py --gate --cards path --output report.md` или `run_pipeline_b_batch.py download --run-gate`.

**Где в коде:** `qc_gate.py` (thresholds, `evaluate_gate`, `load_result_and_evaluate_gate`); `batch_qc.check_qc_threshold`; `batch_io.parse_results` (stages 4–6 drop + error_entries).

---

## Связь с другими документами

- План исправлений: `docs/PIPELINE_B_FIXES_PLAN.MD`
- Ручной QC (диагностика): `docs/PIPELINE_B_MANUAL_QC_INSTRUCTIONS.md`
- Автоматический gate (команды): `docs/PIPELINE_B_QUALITY_GATE.md`
- Рефакторинг Pipeline B: `docs/PIPELINE_B_REFACTORING_PLAN.md`
