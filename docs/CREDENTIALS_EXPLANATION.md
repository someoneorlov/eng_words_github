# Как работает GOOGLE_APPLICATION_CREDENTIALS

## Обзор

`GOOGLE_APPLICATION_CREDENTIALS` - это переменная окружения, которая указывает путь к JSON файлу с credentials для Google Service Account.

## Как это работает в нашем коде

### 1. Приоритет поиска credentials

Когда создается `GoogleSheetsBackend`, система ищет credentials в следующем порядке:

1. **Параметр `credentials_path`** (если передан при создании бэкенда)
   ```python
   backend = GoogleSheetsBackend(
       "spreadsheet_id",
       "Sheet1",
       credentials_path=Path("/path/to/credentials.json")
   )
   ```

2. **Переменная окружения `GOOGLE_APPLICATION_CREDENTIALS_JSON`**
   - Содержит JSON строку напрямую
   - Используется редко, обычно для CI/CD

3. **Переменная окружения `GOOGLE_APPLICATION_CREDENTIALS`**
   - Содержит путь к JSON файлу
   - **Рекомендуемый способ** для локальной разработки

4. **Ошибка**, если ничего не найдено

### 2. Где устанавливать GOOGLE_APPLICATION_CREDENTIALS

#### Вариант A: Системная переменная окружения (рекомендуется)

Установите в `~/.zshrc` (или `~/.bashrc`):

```bash
export GOOGLE_APPLICATION_CREDENTIALS="$HOME/.config/eng_words/google-credentials.json"
```

**Преимущества:**
- Работает для всех проектов
- Не нужно дублировать в каждом проекте
- Безопаснее (не в репозитории)

#### Вариант B: В .env файле проекта (альтернатива)

Можно добавить в `.env`:

```bash
GOOGLE_APPLICATION_CREDENTIALS=$HOME/.config/eng_words/google-credentials.json
```

**Но:** Это не обязательно, так как `python-dotenv` автоматически загружает `.env`, но переменные окружения из системы имеют приоритет.

### 3. Почему в .env.example это закомментировано?

В `.env.example` строка закомментирована, потому что:

1. **Обычно credentials устанавливаются в системных переменных** (`~/.zshrc`)
2. **Не нужно дублировать** в каждом проекте
3. **Безопаснее** - credentials не попадут в репозиторий даже случайно

### 4. Как проверить, что credentials работают?

```bash
# Проверить переменную окружения
echo $GOOGLE_APPLICATION_CREDENTIALS

# Проверить, что файл существует
ls -la $GOOGLE_APPLICATION_CREDENTIALS

# Запустить тест
python scripts/test_gsheets.py
```

### 5. Полный пример настройки

```bash
# 1. Сохранить credentials
mkdir -p ~/.config/eng_words
mv ~/Downloads/your-project-xxxxx.json ~/.config/eng_words/google-credentials.json
chmod 600 ~/.config/eng_words/google-credentials.json

# 2. Установить переменную окружения (в ~/.zshrc)
echo 'export GOOGLE_APPLICATION_CREDENTIALS="$HOME/.config/eng_words/google-credentials.json"' >> ~/.zshrc
source ~/.zshrc

# 3. Проверить
echo $GOOGLE_APPLICATION_CREDENTIALS

# 4. Использовать в проекте
# В .env файле указываем только GOOGLE_SHEETS_URL:
# GOOGLE_SHEETS_URL=gsheets://YOUR_SPREADSHEET_ID/Sheet1
```

## Итого

- **GOOGLE_APPLICATION_CREDENTIALS** - путь к JSON файлу с credentials
- Устанавливается в **системных переменных окружения** (`~/.zshrc`)
- **Не нужно** указывать в `.env` файле (но можно, если хотите)
- Код автоматически найдет credentials через переменную окружения

