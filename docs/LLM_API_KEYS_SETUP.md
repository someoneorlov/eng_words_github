# Настройка API-ключей провайдеров LLM

Пайплайн WSD Gold Dataset и генерация карточек используют ключи трёх провайдеров:
- OpenAI (GPT)
- Anthropic (Claude)
- Google (Gemini)

## 1. OpenAI API Key

### Получение ключа
1. Перейти на https://platform.openai.com/
2. Войти или зарегистрироваться
3. **Settings → API Keys** (https://platform.openai.com/api-keys)
4. **Create new secret key**
5. Скопировать ключ (начинается с `sk-...`)

### Пополнение
1. **Settings → Billing**, добавить платёжный метод
2. Рекомендуемый стартовый баланс: **$5–10**

### Цены (янв. 2026)
| Модель | Вход (1M токенов) | Выход (1M токенов) |
|--------|-------------------|---------------------|
| gpt-5-nano | $0.06 | $0.48 |
| gpt-5-mini (по умолчанию) | $0.30 | $2.40 |
| gpt-5 | $1.25 | $10.00 |
| gpt-5.2 | $2.00 | $16.00 |

**Оценка для Gold Dataset:** ~$0.25 на 500 примеров (gpt-5-mini)

---

## 2. Anthropic API Key

### Получение ключа
1. Перейти на https://console.anthropic.com/
2. Войти или зарегистрироваться (возможна очередь)
3. **Settings → API Keys** (https://console.anthropic.com/settings/keys)
4. **Create Key**
5. Скопировать ключ (начинается с `sk-ant-...`)

### Пополнение
1. **Plans & Billing**, добавить платёжный метод
2. Рекомендуемый стартовый баланс: **$5–10**

### Цены (янв. 2026)
| Модель | Вход (1M токенов) | Выход (1M токенов) |
|--------|-------------------|---------------------|
| claude-haiku-4-5 (по умолчанию) | $1.00 | $5.00 |
| claude-sonnet-4-5 | $3.00 | $15.00 |
| claude-opus-4-5 | $5.00 | $25.00 |

**Оценка для Gold Dataset:** ~$0.50 на 500 примеров (claude-haiku-4-5)

---

## 3. Google Gemini API Key

### Получение ключа
1. Перейти на https://aistudio.google.com/
2. Войти с Google-аккаунтом
3. В меню: **Get API Key**
4. Выбрать проект или создать новый → **Create API key**
5. Скопировать ключ (начинается с `AIza...`)

### Альтернатива через Google Cloud
1. https://console.cloud.google.com/ → создать проект
2. Включить **Generative Language API**
3. Создать API key в **APIs & Services → Credentials**

### Цены (янв. 2026)
| Модель | Вход (1M токенов) | Выход (1M токенов) |
|--------|-------------------|---------------------|
| gemini-2.0-flash (по умолчанию) | $0.10 | $0.40 |
| gemini-2.5-flash | $0.08 | $0.32 |
| gemini-2.5-pro | $1.25 | $5.00 |
| gemini-3-pro-preview | $1.50 | $6.00 |

**Оценка для Gold Dataset:** ~$0.05 на 500 примеров (gemini-2.0-flash)

⚠️ **Бесплатный уровень:** у Gemini лимит 15 RPM; для тестов обычно достаточно.

---

## 4. Файл .env

В корне проекта создать `.env`:

```bash
# OpenAI
OPENAI_API_KEY=sk-...

# Anthropic
ANTHROPIC_API_KEY=sk-ant-...

# Google Gemini
GOOGLE_API_KEY=AIza...
```

### Проверка ключей

```bash
python scripts/test_gold_pipeline_real_data.py --with-llm
```

Ожидаемый вывод по каждому провайдеру: сообщение вида «✅ … provider works!».

---

## 5. Зависимости

Установить пакеты с поддержкой LLM:

```bash
pip install -e ".[llm]"

# Или вручную:
pip install openai>=2.11.0 anthropic>=0.75.0 google-genai>=1.0.0
```

---

## 6. Устранение неполадок

- **OpenAI: «Incorrect API key provided»** — проверить, что ключ начинается с `sk-`, баланс > 0.
- **Anthropic: «Invalid API Key»** — проверить формат `sk-ant-...`, активацию аккаунта.
- **Gemini: «API key not valid»** — проверить включение Generative Language API, при необходимости создать ключ заново.
- **Gemini: RESOURCE_EXHAUSTED / 429** — исчерпан лимит (например, бесплатный 15 RPM); подождать или перейти на платный план.
- **Rate limits:** у OpenAI у новых аккаунтов низкий лимит; у Anthropic batch API обходит лимиты; у Gemini на бесплатном уровне 15 RPM.

---

## 7. Оценка стоимости Gold Dataset

Для 500 примеров × 3 провайдера:

| Провайдер | Модель | Стоимость |
|-----------|--------|-----------|
| OpenAI | gpt-5-mini | ~$0.25 |
| Anthropic | claude-haiku-4-5 | ~$0.50 |
| Gemini | gemini-2.0-flash | ~$0.05 |
| **Итого** | | **~$0.80** |

Для полного датасета (1000+ примеров): **~$1–2**
