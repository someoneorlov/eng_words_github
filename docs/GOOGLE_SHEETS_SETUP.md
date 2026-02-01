# Настройка Google Sheets для списка известных слов

Пошаговая инструкция по подключению Google Sheets для хранения списка известных слов.

## Быстрый старт

Если вы уже работали с Google Cloud:

1. **Создать Service Account:**
   - Перейти на https://console.cloud.google.com/
   - Создать или выбрать проект
   - Включить Google Sheets API и Google Drive API
   - Создать Service Account → Keys → Create new key (JSON)
   - Скопировать email сервисного аккаунта

2. **Создать таблицу:**
   - Создать новую таблицу на https://sheets.google.com/
   - Заголовки: `lemma | status | item_type | tags`
   - Открыть доступ по email сервисного аккаунта (роль: Редактор)
   - Скопировать ID таблицы из URL

3. **Сохранить учётные данные:**
   ```bash
   mkdir -p ~/.config/eng_words
   mv ~/Downloads/your-project-xxxxx.json ~/.config/eng_words/google-credentials.json
   chmod 600 ~/.config/eng_words/google-credentials.json
   echo 'export GOOGLE_APPLICATION_CREDENTIALS="$HOME/.config/eng_words/google-credentials.json"' >> ~/.zshrc
   source ~/.zshrc
   ```

4. **Проверка:**
   - Вписать свой Spreadsheet ID в `scripts/test_gsheets.py`
   - Запустить: `python scripts/test_gsheets.py`

5. **Настройка (опционально):**
   - Создать `.env`: `cp .env.example .env`
   - Указать: `GOOGLE_SHEETS_URL=gsheets://YOUR_SPREADSHEET_ID/Sheet1`

6. **Использование в пайплайне:**
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

**Подробная инструкция — ниже.**

---

## Подробная настройка

### Шаг 1: Проект и Service Account в Google Cloud

1. **Открыть Google Cloud Console:** https://console.cloud.google.com/
2. **Создать проект** (или выбрать существующий), включить **Google Sheets API** и **Google Drive API**.
3. **Создать Service Account** (APIs & Services → Credentials → Create Credentials → Service Account), при необходимости пропустить роль.
4. **Создать ключ** (Keys → Add Key → Create new key → JSON), сохранить файл.
5. **Скопировать email** сервисного аккаунта (нужен для шага «Поделиться»).

### Шаг 2: Таблица в Google Sheets

1. Создать новую таблицу на https://sheets.google.com/.
2. В первой строке задать заголовки: `lemma | status | item_type | tags` (примеры строк см. в оригинальной документации).
3. Нажать «Настройки доступа» и добавить **email сервисного аккаунта** с ролью **Редактор** (уведомление можно отключить).
4. **Скопировать ID таблицы** из URL: `https://docs.google.com/spreadsheets/d/SPREADSHEET_ID/edit`.

### Шаг 3: Сохранение учётных данных

- Создать каталог `~/.config/eng_words`, перенести туда скачанный JSON, переименовать в `google-credentials.json`.
- Выставить права: `chmod 600 ~/.config/eng_words/google-credentials.json`.
- При необходимости добавить в `.gitignore`: `*.json`, `.config/`.

### Шаг 4: Переменные окружения

- Для текущей сессии: `export GOOGLE_APPLICATION_CREDENTIALS="$HOME/.config/eng_words/google-credentials.json"`.
- Постоянно: добавить эту строку в `~/.zshrc` (или `~/.bashrc`) и выполнить `source ~/.zshrc`.

### Шаг 5: Проверка интеграции

Использовать тестовый скрипт (заменить `YOUR_SPREADSHEET_ID_HERE` на свой ID и при необходимости имя листа):

- Загрузка: `load_known_words(gsheets_url)`
- Сохранение тестовых данных и повторная загрузка для проверки

Запуск: `python scripts/test_gsheets.py` (после подстановки ID в скрипт).

### Шаг 6: .env (опционально)

- Скопировать `.env.example` в `.env`.
- Указать: `GOOGLE_SHEETS_URL=gsheets://YOUR_SPREADSHEET_ID/Sheet1`.

Тогда пайплайн сможет брать URL таблицы из `.env`, если не передавать `--known-words`.

**Примечание:** Переменную `GOOGLE_APPLICATION_CREDENTIALS` обычно задают в оболочке (`~/.zshrc`), а не в `.env`. Подробнее — в разделе «Учётные данные» ниже.

### Шаг 7: Использование в CLI

После успешной проверки запускать пайплайн с `--known-words gsheets://SPREADSHEET_ID/Sheet1` или без него, если задан `GOOGLE_SHEETS_URL` в `.env`.

## Формат данных

В таблице должны быть колонки:
- `lemma` — лемма слова (строчными)
- `status` — одно из: `known`, `learning`, `ignore`, `maybe`
- `item_type` — одно из: `word`, `phrasal_verb`, `ngram`
- `tags` — опционально

Дополнительные колонки (созданы при необходимости) игнорируются.

## Учётные данные (Credentials)

### Как работает GOOGLE_APPLICATION_CREDENTIALS

Переменная окружения `GOOGLE_APPLICATION_CREDENTIALS` указывает путь к JSON-файлу с ключом сервисного аккаунта Google.

### Порядок поиска учётных данных

При создании бэкенда Google Sheets код ищет учётные данные в таком порядке:

1. Параметр `credentials_path` при создании бэкенда
2. Переменная окружения `GOOGLE_APPLICATION_CREDENTIALS_JSON` (строка JSON)
3. Переменная окружения `GOOGLE_APPLICATION_CREDENTIALS` (путь к файлу) — **рекомендуется для локальной разработки**
4. Если ничего не найдено — ошибка

### Где задавать GOOGLE_APPLICATION_CREDENTIALS

- **Рекомендуется:** в `~/.zshrc` или `~/.bashrc` (работает для всех проектов, не попадает в репозиторий).
- **Альтернатива:** в `.env` проекта (если загружаете его через `python-dotenv`).

В `.env.example` эта переменная закомментирована, чтобы не дублировать настройки и не рисковать случайно закоммитить путь к секрету.

### Проверка

```bash
echo $GOOGLE_APPLICATION_CREDENTIALS
ls -la $GOOGLE_APPLICATION_CREDENTIALS
python scripts/test_gsheets.py
```

## Устранение неполадок

- **«Failed to access Google Sheets»** — проверить доступ сервисного аккаунта (роль Редактор), правильность ID таблицы и имени листа.
- **«Google credentials not found»** — проверить путь в `GOOGLE_APPLICATION_CREDENTIALS`, существование файла и права (например, `chmod 600`).
- **«Permission denied»** — убедиться, что у сервисного аккаунта роль **Редактор**, а не только Просмотр.
- **Credentials not loading** — помнить, что переменные оболочки имеют приоритет над `.env`; после правок в `~/.zshrc` перезапустить терминал.

## Безопасность

- Не коммитить файл с учётными данными в git.
- Хранить JSON в надёжном месте, права на файл — 600.
- Предпочтительно задавать путь к учётным данным через переменные окружения, а не через `.env` в репозитории.
