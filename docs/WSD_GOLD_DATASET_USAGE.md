# Использование WSD Gold Dataset

## Что это

WSD Gold Dataset — эталонный датасет для оценки алгоритмов снятия неоднозначности смысла слова (Word Sense Disambiguation).

**Содержимое:**
- 3000 примеров из 4 книг
- Разметка от LLM-судей (Claude, Gemini, GPT)
- Разбиение на dev (1500) и test_locked (1500)

---

## Быстрый старт

### Оценка текущего WSD

```bash
# Полная оценка на dev (~2 минуты)
make eval-wsd-gold

# Быстрая оценка (100 примеров)
make eval-wsd-gold-quick
```

### Проверка целостности тестового набора

```bash
make verify-gold
# ✅ Checksum verified: 8e642428413d582c...
```

---

## Правила использования

### ✅ РАЗРЕШЕНО

1. **Использовать dev для разработки**
   ```python
   from eng_words.wsd_gold.eval import load_gold_examples

   dev_examples = load_gold_examples("data/wsd_gold/gold_dev.jsonl")
   for ex in dev_examples:
       print(f"Слово: {ex['target']['lemma']}")
       print(f"Контекст: {ex['context_window']}")
       print(f"Эталон: {ex['gold_synset_id']}")
   ```

2. **Смотреть ошибки на dev**
   ```bash
   uv run python scripts/eval_wsd_on_gold.py --show-errors --top-errors 20
   ```

3. **Подбирать параметры на dev**
   - Порог уверенности
   - Веса по частям речи
   - Правила для конструкций

### ❌ ЗАПРЕЩЕНО

1. **Не смотреть в `gold_test_locked.jsonl`!**
   - Только для финального сравнения
   - Просмотр искажает результаты

2. **Не подбирать параметры по тестовому набору**
   - Это утечка данных
   - Результаты будут нечестными

3. **Не редактировать тестовый набор**
   - CI проверяет контрольную сумму
   - Любое изменение сломает проверку

---

## Структура данных

### Пример из датасета

```json
{
  "example_id": "book:american_tragedy_wsd|sent:12008|tok:1",
  "source_id": "american_tragedy_wsd",
  "context_window": "I went to the bank to deposit money.",
  "target": {
    "surface": "bank",
    "lemma": "bank",
    "pos": "NOUN",
    "char_span": [15, 19]
  },
  "candidates": [
    {
      "synset_id": "bank.n.01",
      "gloss": "financial institution",
      "examples": ["The bank raised interest rates."]
    },
    {
      "synset_id": "bank.n.02",
      "gloss": "sloping land beside water",
      "examples": ["They sat on the river bank."]
    }
  ],
  "metadata": {
    "wn_sense_count": 10,
    "baseline_top1": "bank.n.01",
    "baseline_margin": 0.25,
    "is_multiword": false
  },
  "gold_synset_id": "bank.n.01",
  "gold_confidence": 1.0,
  "gold_agreement": 1.0,
  "gold_flags": []
}
```

### Основные поля

| Поле | Описание |
|------|----------|
| `context_window` | Предложение с целевым словом |
| `target.lemma` | Лемма целевого слова |
| `target.pos` | Часть речи (NOUN, VERB, ADJ, ADV) |
| `candidates` | Варианты смыслов из WordNet |
| `gold_synset_id` | Правильный ответ |
| `gold_confidence` | Уверенность (1.0 = все LLM согласны) |
| `gold_agreement` | Доля согласных LLM (0.67 = 2 из 3) |
| `metadata.baseline_top1` | Ответ базового WSD |
| `metadata.baseline_margin` | Отступ до второго кандидата |

---

## Типичный рабочий процесс

### 1. Зафиксировать базовый уровень

```bash
make eval-wsd-gold
# Overall Accuracy: 47.5%
```

### 2. Менять код WSD

```python
# src/eng_words/wsd/wordnet_backend.py
# ... ваши изменения ...
```

### 3. Оценивать на dev

```bash
make eval-wsd-gold
# Overall Accuracy: 52.3%  # Улучшение!
```

### 4. Повторять шаги 2–3 до достижения цели

### 5. Финальное сравнение на тестовом наборе

```bash
uv run python scripts/eval_wsd_on_gold.py \
  --gold-path data/wsd_gold/gold_test_locked.jsonl
```

⚠️ **Делать это только ОДИН раз в конце!**

---

## Метрики

### По частям речи

```
ADJ  (прилагательные):  56.8%
ADV  (наречия):        53.0%
NOUN (существительные): 50.6%
VERB (глаголы):         34.0%  ← Слабое место!
```

### По сложности

```
Easy   (≤2 смысла, margin ≥0.3):  80.8%
Medium (3–5 смыслов):            46.4%
Hard   (≥6 смыслов, margin <0.15): 25.7%
```

**Вывод:** Имеет смысл фокусироваться на глаголах и сложных словах.

---

## API

### Загрузка примеров

```python
from eng_words.wsd_gold import load_gold_examples

examples = load_gold_examples("data/wsd_gold/gold_dev.jsonl")
print(f"Загружено {len(examples)} примеров")
```

### Оценка одного примера

```python
from eng_words.wsd_gold import evaluate_single

result = evaluate_single(
    predicted_synset="bank.n.01",
    gold_synset="bank.n.01"
)
print(result["is_correct"])  # True
```

### Полная оценка

```python
from eng_words.wsd import WordNetSenseBackend
from eng_words.wsd_gold import evaluate_wsd_on_gold

backend = WordNetSenseBackend()
results = evaluate_wsd_on_gold(
    gold_path="data/wsd_gold/gold_dev.jsonl",
    backend=backend
)

print(f"Accuracy: {results['metrics']['accuracy']:.1%}")
print(f"По POS: {results['by_pos']}")
print(f"По сложности: {results['by_difficulty']}")
```

### Использование кэша LLM

```python
from eng_words.wsd_gold import LLMCache

cache = LLMCache(cache_dir="data/wsd_gold/cache")

# Проверка кэша
cached = cache.get("example_id", "gpt-5.2")

# Сохранение в кэш
cache.set("example_id", "gpt-5.2", model_output)

# Статистика
print(cache.stats)  # {"hits": 100, "misses": 5, "hit_rate": 0.95}
```

---

## FAQ

### Почему разбиение по книгам, а не по 375 из каждой?

Чтобы избежать **утечки данных**:
- Модель может запомнить стиль автора
- Частотность слов в книге
- Контекст персонажей

Разбиение по книгам — более строгий тест.

### Можно ли добавлять новые примеры?

Да, но:
1. Не трогать `gold_test_locked.jsonl`
2. Добавлять только в dev
3. Пересчитать контрольную сумму, если меняли тестовый набор

### Что если LLM ошибся при разметке?

Около 2–5% примеров могут иметь шумные эталонные метки. Это нормально:
- На большом датасете шум усредняется
- Важнее общий тренд, а не отдельные примеры
- Можно вести «карантинный» список пограничных случаев
