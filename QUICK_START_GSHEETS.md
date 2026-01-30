# Quick Google Sheets setup

## 1. Create a Service Account in Google Cloud

1. Open https://console.cloud.google.com/
2. Create a project (or select an existing one)
3. Enable APIs:
   - Google Sheets API
   - Google Drive API
4. Create a Service Account:
   - APIs & Services → Credentials → Create Credentials → Service Account
   - Name: `eng-words-service`
5. Create a key:
   - In Service Account → Keys → Add Key → Create new key → JSON
   - **Save the downloaded file!**
6. Copy the Service Account email (from Details)

## 2. Create a Google Sheet

1. Create a new sheet: https://sheets.google.com/
2. Name it: `English Words - Known Words`
3. Add headers in the first row:
   ```
   lemma | status | item_type | tags
   ```
4. Share the sheet:
   - Click "Share" → paste the Service Account email → role **Editor**
5. Copy the Spreadsheet ID from the URL:
   - URL: `https://docs.google.com/spreadsheets/d/SPREADSHEET_ID/edit`
   - Copy `SPREADSHEET_ID`

## 3. Save credentials

```bash
# Create directory
mkdir -p ~/.config/eng_words

# Move the downloaded JSON file
mv ~/Downloads/your-project-xxxxx.json ~/.config/eng_words/google-credentials.json

# Set permissions
chmod 600 ~/.config/eng_words/google-credentials.json

# Add to environment
echo 'export GOOGLE_APPLICATION_CREDENTIALS="$HOME/.config/eng_words/google-credentials.json"' >> ~/.zshrc
source ~/.zshrc
```

## 4. Test

Edit `scripts/test_gsheets.py`:
- Replace `YOUR_SPREADSHEET_ID_HERE` with your Spreadsheet ID
- Change `WORKSHEET_NAME` if needed (default is "Sheet1")

Run:
```bash
python scripts/test_gsheets.py
```

## 5. Configure .env (optional)

Create a `.env` file in the project root:

```bash
# Copy example
cp .env.example .env

# Edit .env and set your Spreadsheet ID
# GOOGLE_SHEETS_URL=gsheets://YOUR_SPREADSHEET_ID/Sheet1
```

Then you don't need to pass `--known-words` on the command line — it will be read from `.env`.

## 6. Use in the pipeline

**With .env:**
```bash
python -m eng_words.pipeline \
  --book-path data/raw/your-book.epub \
  --book-name book_name \
  --output-dir data/processed \
  --min-book-freq 3 \
  --min-zipf 2.0 \
  --max-zipf 5.3 \
  --top-n 150
```

**Or without .env (specify directly):**
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

**Full instructions:** see `docs/GOOGLE_SHEETS_SETUP.md`
