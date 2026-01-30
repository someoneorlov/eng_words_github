# Настройка Google Sheets для Known Words

Пошаговая инструкция по настройке интеграции с Google Sheets для хранения списка изученных слов.

## Шаг 1: Создание Google Cloud Project и Service Account

1. **Перейдите в Google Cloud Console:**
   - Откройте https://console.cloud.google.com/
   - Войдите в свой Google аккаунт

2. **Создайте новый проект (или выберите существующий):**
   - Нажмите на выпадающий список проектов вверху
   - Нажмите "New Project"
   - Введите название: `eng-words` (или любое другое)
   - Нажмите "Create"

3. **Включите необходимые API:**
   - В меню слева выберите "APIs & Services" → "Library"
   - Найдите и включите:
     - **Google Sheets API**
     - **Google Drive API**

4. **Создайте Service Account:**
   - Перейдите в "APIs & Services" → "Credentials"
   - Нажмите "Create Credentials" → "Service Account"
   - Заполните:
     - **Service account name**: `eng-words-service`
     - **Service account ID**: автоматически заполнится
     - **Description**: `Service account for English Words learning tool`
   - Нажмите "Create and Continue"
   - Роль можно пропустить (нажмите "Continue")
   - Нажмите "Done"

5. **Создайте ключ для Service Account:**
   - Найдите созданный Service Account в списке
   - Нажмите на него
   - Перейдите на вкладку "Keys"
   - Нажмите "Add Key" → "Create new key"
   - Выберите формат: **JSON**
   - Нажмите "Create"
   - **Файл автоматически скачается** - сохраните его в безопасном месте!

6. **Скопируйте email Service Account:**
   - В разделе "Details" Service Account найдите поле **Email**
   - Скопируйте этот email (выглядит как `eng-words-service@your-project.iam.gserviceaccount.com`)
   - Он понадобится на следующем шаге

## Шаг 2: Создание Google Sheets таблицы

1. **Создайте новую Google Sheets таблицу:**
   - Откройте https://sheets.google.com/
   - Нажмите "Blank" для создания новой таблицы
   - Назовите её, например: `English Words - Known Words`

2. **Настройте заголовки:**
   - В первой строке введите заголовки:
     ```
     lemma | status | item_type | tags
     ```
   - Пример заполнения:
     ```
     lemma      | status  | item_type    | tags
     run        | known   | word         | A2 basic_verbs
     give up    | learning| phrasal_verb | B1 phrasal
     the        | ignore  | word         | stopword
     ```

3. **Предоставьте доступ Service Account:**
   - Нажмите кнопку "Share" (Поделиться) в правом верхнем углу
   - В поле "Add people and groups" вставьте **email Service Account** (из шага 1.6)
   - Выберите роль: **Editor** (Редактор)
   - **Снимите галочку** "Notify people" (чтобы не отправлять уведомление)
   - Нажмите "Share"

4. **Скопируйте Spreadsheet ID:**
   - Посмотрите на URL таблицы в браузере
   - URL выглядит так: `https://docs.google.com/spreadsheets/d/SPREADSHEET_ID/edit`
   - Скопируйте `SPREADSHEET_ID` (длинная строка между `/d/` и `/edit`)
   - Пример: `1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms`

## Шаг 3: Сохранение credentials в проекте

1. **Создайте папку для credentials:**
   ```bash
   mkdir -p ~/.config/eng_words
   ```

2. **Переместите скачанный JSON файл:**
   ```bash
   # Переместите скачанный файл (например, your-project-xxxxx.json)
   mv ~/Downloads/your-project-xxxxx.json ~/.config/eng_words/google-credentials.json
   ```

3. **Установите права доступа (важно для безопасности):**
   ```bash
   chmod 600 ~/.config/eng_words/google-credentials.json
   ```

4. **Добавьте в .gitignore (если еще не добавлено):**
   ```bash
   echo "*.json" >> .gitignore
   echo ".config/" >> .gitignore
   ```

## Шаг 4: Настройка переменных окружения

1. **Установите переменную окружения:**
   
   Для текущей сессии:
   ```bash
   export GOOGLE_APPLICATION_CREDENTIALS="$HOME/.config/eng_words/google-credentials.json"
   ```

   Для постоянного использования добавьте в `~/.zshrc` (или `~/.bashrc`):
   ```bash
   echo 'export GOOGLE_APPLICATION_CREDENTIALS="$HOME/.config/eng_words/google-credentials.json"' >> ~/.zshrc
   source ~/.zshrc
   ```

## Шаг 5: Тестирование интеграции

Создайте тестовый скрипт для проверки:

```python
# test_gsheets.py
from pathlib import Path
from eng_words.storage import load_known_words, save_known_words
import pandas as pd

# Замените на ваш SPREADSHEET_ID и имя листа
SPREADSHEET_ID = "YOUR_SPREADSHEET_ID_HERE"
WORKSHEET_NAME = "Sheet1"  # или другое имя листа

# URL формат
gsheets_url = f"gsheets://{SPREADSHEET_ID}/{WORKSHEET_NAME}"

print("Testing Google Sheets integration...")
print(f"URL: {gsheets_url}")

# Тест 1: Загрузка данных
try:
    df = load_known_words(gsheets_url)
    print(f"✅ Load successful! Found {len(df)} rows")
    if not df.empty:
        print("\nFirst few rows:")
        print(df.head())
except Exception as e:
    print(f"❌ Load failed: {e}")

# Тест 2: Сохранение данных
try:
    test_df = pd.DataFrame({
        "lemma": ["test_word", "test_phrase"],
        "status": ["learning", "known"],
        "item_type": ["word", "phrasal_verb"],
        "tags": ["test", "test"]
    })
    save_known_words(test_df, gsheets_url)
    print("\n✅ Save successful!")
except Exception as e:
    print(f"❌ Save failed: {e}")

# Тест 3: Повторная загрузка для проверки
try:
    df = load_known_words(gsheets_url)
    print(f"\n✅ Reload successful! Found {len(df)} rows")
    if "test_word" in df["lemma"].values:
        print("✅ Test data found in sheet!")
except Exception as e:
    print(f"❌ Reload failed: {e}")
```

Запустите:
```bash
python test_gsheets.py
```

## Шаг 6: Использование в CLI

После успешного тестирования используйте в пайплайне:

```bash
python -m eng_words.pipeline \
  --book-path data/raw/your-book.epub \
  --book-name book_name \
  --output-dir data/processed \
  --known-words gsheets://YOUR_SPREADSHEET_ID/Sheet1 \
  --min-book-freq 3 \
  --min-zipf 2.0 \
  --max-zipf 5.3 \
  --top-n 150
```

## Формат данных

Таблица должна содержать следующие колонки (обязательные):
- `lemma` - лемма слова (lowercase)
- `status` - статус: `known`, `learning`, `ignore`, `maybe`
- `item_type` - тип: `word`, `phrasal_verb`, `ngram`
- `tags` - теги (опционально, может быть пустым)

Дополнительные колонки (опциональные, будут проигнорированы):
- `created_at`
- `last_seen_in_book`
- `examples_count`
- `notes`

## Устранение проблем

### Ошибка: "Failed to access Google Sheets"
- Проверьте, что Service Account имеет доступ к таблице (роль Editor)
- Проверьте, что Spreadsheet ID правильный
- Проверьте, что имя листа (worksheet) правильное

### Ошибка: "Google credentials not found"
- Проверьте путь к файлу credentials: `echo $GOOGLE_APPLICATION_CREDENTIALS`
- Проверьте, что файл существует: `ls -la $GOOGLE_APPLICATION_CREDENTIALS`
- Проверьте права доступа: `chmod 600 $GOOGLE_APPLICATION_CREDENTIALS`

### Ошибка: "Permission denied"
- Убедитесь, что Service Account имеет роль **Editor** (не Viewer!)
- Проверьте, что email Service Account правильный

### Таблица не создается автоматически
- Убедитесь, что Service Account имеет доступ к таблице
- Проверьте, что имя листа правильное (регистр важен!)

## Безопасность

⚠️ **Важно:**
- Никогда не коммитьте файл credentials в git
- Храните credentials в безопасном месте
- Используйте права доступа 600 для файла credentials
- Не делитесь credentials с другими

