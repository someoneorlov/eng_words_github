# Google Sheets setup for Known Words

Step-by-step guide to set up Google Sheets integration for storing your list of known words.

## Quick Start

If you're familiar with Google Cloud, here's the condensed version:

1. **Create Service Account:**
   - Go to https://console.cloud.google.com/
   - Create/select a project
   - Enable Google Sheets API and Google Drive API
   - Create Service Account → Keys → Create new key (JSON)
   - Copy the Service Account email

2. **Create Google Sheet:**
   - Create a new sheet at https://sheets.google.com/
   - Add headers: `lemma | status | item_type | tags`
   - Share with Service Account email (role: Editor)
   - Copy Spreadsheet ID from URL

3. **Save credentials:**
   ```bash
   mkdir -p ~/.config/eng_words
   mv ~/Downloads/your-project-xxxxx.json ~/.config/eng_words/google-credentials.json
   chmod 600 ~/.config/eng_words/google-credentials.json
   echo 'export GOOGLE_APPLICATION_CREDENTIALS="$HOME/.config/eng_words/google-credentials.json"' >> ~/.zshrc
   source ~/.zshrc
   ```

4. **Test:**
   - Edit `scripts/test_gsheets.py` with your Spreadsheet ID
   - Run: `python scripts/test_gsheets.py`

5. **Configure (optional):**
   - Create `.env` file: `cp .env.example .env`
   - Set: `GOOGLE_SHEETS_URL=gsheets://YOUR_SPREADSHEET_ID/Sheet1`

6. **Use in pipeline:**
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

**For detailed instructions, continue reading below.**

---

## Detailed Setup

### Step 1: Create Google Cloud project and Service Account

1. **Open Google Cloud Console:**
   - Go to https://console.cloud.google.com/
   - Sign in with your Google account

2. **Create a new project (or select an existing one):**
   - Click the project dropdown at the top
   - Click "New Project"
   - Enter a name: `eng-words` (or any other)
   - Click "Create"

3. **Enable required APIs:**
   - In the left menu select "APIs & Services" → "Library"
   - Find and enable:
     - **Google Sheets API**
     - **Google Drive API**

4. **Create a Service Account:**
   - Go to "APIs & Services" → "Credentials"
   - Click "Create Credentials" → "Service Account"
   - Fill in:
     - **Service account name**: `eng-words-service`
     - **Service account ID**: auto-filled
     - **Description**: `Service account for English Words learning tool`
   - Click "Create and Continue"
   - You can skip the role (click "Continue")
   - Click "Done"

5. **Create a key for the Service Account:**
   - Find the created Service Account in the list
   - Click on it
   - Go to the "Keys" tab
   - Click "Add Key" → "Create new key"
   - Choose format: **JSON**
   - Click "Create"
   - **The file will download automatically** — save it in a safe place!

6. **Copy the Service Account email:**
   - In the Service Account "Details" section find the **Email** field
   - Copy this email (looks like `eng-words-service@your-project.iam.gserviceaccount.com`)
   - You will need it in the next step

### Step 2: Create the Google Sheet

1. **Create a new Google Sheet:**
   - Go to https://sheets.google.com/
   - Click "Blank" to create a new sheet
   - Name it e.g.: `English Words - Known Words`

2. **Set up headers:**
   - In the first row enter:
     ```
     lemma | status | item_type | tags
     ```
   - Example rows:
     ```
     lemma      | status  | item_type    | tags
     run        | known   | word         | A2 basic_verbs
     give up    | learning| phrasal_verb | B1 phrasal
     the        | ignore  | word         | stopword
     ```

3. **Share with the Service Account:**
   - Click the "Share" button (top right)
   - In "Add people and groups" paste the **Service Account email** (from step 1.6)
   - Set role: **Editor**
   - **Uncheck** "Notify people" (to avoid sending an email)
   - Click "Share"

4. **Copy the Spreadsheet ID:**
   - Look at the sheet URL in your browser
   - URL looks like: `https://docs.google.com/spreadsheets/d/SPREADSHEET_ID/edit`
   - Copy `SPREADSHEET_ID` (the long string between `/d/` and `/edit`)
   - Example: `1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms`

### Step 3: Save credentials

1. **Create a directory for credentials:**
   ```bash
   mkdir -p ~/.config/eng_words
   ```

2. **Move the downloaded JSON file:**
   ```bash
   mv ~/Downloads/your-project-xxxxx.json ~/.config/eng_words/google-credentials.json
   ```

3. **Set file permissions (important for security):**
   ```bash
   chmod 600 ~/.config/eng_words/google-credentials.json
   ```

4. **Add to .gitignore (if not already):**
   ```bash
   echo "*.json" >> .gitignore
   echo ".config/" >> .gitignore
   ```

### Step 4: Set environment variables

1. **Set the environment variable:**
   
   For the current session:
   ```bash
   export GOOGLE_APPLICATION_CREDENTIALS="$HOME/.config/eng_words/google-credentials.json"
   ```

   For permanent use add to `~/.zshrc` (or `~/.bashrc`):
   ```bash
   echo 'export GOOGLE_APPLICATION_CREDENTIALS="$HOME/.config/eng_words/google-credentials.json"' >> ~/.zshrc
   source ~/.zshrc
   ```

### Step 5: Test the integration

Create a test script:

```python
# test_gsheets.py
from pathlib import Path
from eng_words.storage import load_known_words, save_known_words
import pandas as pd

# Replace with your SPREADSHEET_ID and worksheet name
SPREADSHEET_ID = "YOUR_SPREADSHEET_ID_HERE"
WORKSHEET_NAME = "Sheet1"  # or another worksheet name

gsheets_url = f"gsheets://{SPREADSHEET_ID}/{WORKSHEET_NAME}"

print("Testing Google Sheets integration...")
print(f"URL: {gsheets_url}")

# Test 1: Load data
try:
    df = load_known_words(gsheets_url)
    print(f"✅ Load successful! Found {len(df)} rows")
    if not df.empty:
        print("\nFirst few rows:")
        print(df.head())
except Exception as e:
    print(f"❌ Load failed: {e}")

# Test 2: Save data
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

# Test 3: Reload to verify
try:
    df = load_known_words(gsheets_url)
    print(f"\n✅ Reload successful! Found {len(df)} rows")
    if "test_word" in df["lemma"].values:
        print("✅ Test data found in sheet!")
except Exception as e:
    print(f"❌ Reload failed: {e}")
```

Run:
```bash
python test_gsheets.py
```

Or use the existing test script:
```bash
# Edit scripts/test_gsheets.py and replace YOUR_SPREADSHEET_ID_HERE with your Spreadsheet ID
python scripts/test_gsheets.py
```

### Step 6: Configure .env (optional)

You can configure the Google Sheets URL in `.env` to avoid passing `--known-words` on every command:

1. **Create `.env` file:**
   ```bash
   cp .env.example .env
   ```

2. **Edit `.env` and set:**
   ```bash
   GOOGLE_SHEETS_URL=gsheets://YOUR_SPREADSHEET_ID/Sheet1
   ```

Now you can run the pipeline without specifying `--known-words` — it will be read from `.env`.

**Note:** The `GOOGLE_APPLICATION_CREDENTIALS` environment variable is typically set in your system shell (`~/.zshrc` or `~/.bashrc`), not in `.env`. See the [Credentials Explanation](#credentials-explanation) section below for details.

### Step 7: Use in CLI

After a successful test, use in the pipeline:

**With .env configured:**
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

## Data format

The sheet must have these columns (required):
- `lemma` — word lemma (lowercase)
- `status` — one of: `known`, `learning`, `ignore`, `maybe`
- `item_type` — one of: `word`, `phrasal_verb`, `ngram`
- `tags` — optional, can be empty

Additional columns (optional, ignored):
- `created_at`
- `last_seen_in_book`
- `examples_count`
- `notes`

## Credentials Explanation

### How GOOGLE_APPLICATION_CREDENTIALS works

`GOOGLE_APPLICATION_CREDENTIALS` is an environment variable that points to the path of a JSON file containing credentials for a Google Service Account.

### Credential lookup order

When creating `GoogleSheetsBackend`, the system looks for credentials in this order:

1. **`credentials_path` parameter** (if passed when creating the backend)
   ```python
   backend = GoogleSheetsBackend(
       "spreadsheet_id",
       "Sheet1",
       credentials_path=Path("/path/to/credentials.json")
   )
   ```

2. **Environment variable `GOOGLE_APPLICATION_CREDENTIALS_JSON`**
   - Contains the JSON string directly
   - Used rarely, usually for CI/CD

3. **Environment variable `GOOGLE_APPLICATION_CREDENTIALS`**
   - Contains the path to the JSON file
   - **Recommended** for local development

4. **Error** if none found

### Where to set GOOGLE_APPLICATION_CREDENTIALS

#### Option A: System environment variable (recommended)

Add to `~/.zshrc` (or `~/.bashrc`):

```bash
export GOOGLE_APPLICATION_CREDENTIALS="$HOME/.config/eng_words/google-credentials.json"
```

**Advantages:**
- Works for all projects
- No need to duplicate in each project
- Safer (not in the repo)

#### Option B: In project .env file (alternative)

You can add to `.env`:

```bash
GOOGLE_APPLICATION_CREDENTIALS=$HOME/.config/eng_words/google-credentials.json
```

**Note:** Not required, since `python-dotenv` loads `.env` automatically, but system environment variables take precedence.

### Why is it commented out in .env.example?

In `.env.example` the line is commented because:

1. **Credentials are usually set in system variables** (`~/.zshrc`)
2. **No need to duplicate** in each project
3. **Safer** — credentials won't end up in the repo by mistake

### How to verify credentials work?

```bash
# Check environment variable
echo $GOOGLE_APPLICATION_CREDENTIALS

# Check that file exists
ls -la $GOOGLE_APPLICATION_CREDENTIALS

# Run test
python scripts/test_gsheets.py
```

### Full setup example

```bash
# 1. Save credentials
mkdir -p ~/.config/eng_words
mv ~/Downloads/your-project-xxxxx.json ~/.config/eng_words/google-credentials.json
chmod 600 ~/.config/eng_words/google-credentials.json

# 2. Set environment variable (in ~/.zshrc)
echo 'export GOOGLE_APPLICATION_CREDENTIALS="$HOME/.config/eng_words/google-credentials.json"' >> ~/.zshrc
source ~/.zshrc

# 3. Verify
echo $GOOGLE_APPLICATION_CREDENTIALS

# 4. Use in project
# In .env only set GOOGLE_SHEETS_URL:
# GOOGLE_SHEETS_URL=gsheets://YOUR_SPREADSHEET_ID/Sheet1
```

### Summary

- **GOOGLE_APPLICATION_CREDENTIALS** — path to the JSON credentials file
- Set in **system environment** (`~/.zshrc`)
- **No need** to set in `.env` (but you can if you prefer)
- The code finds credentials automatically via the environment variable

## Troubleshooting

### Error: "Failed to access Google Sheets"
- Check that the Service Account has Editor access to the sheet
- Check that the Spreadsheet ID is correct
- Check that the worksheet name is correct

### Error: "Google credentials not found"
- Check credentials path: `echo $GOOGLE_APPLICATION_CREDENTIALS`
- Check that the file exists: `ls -la $GOOGLE_APPLICATION_CREDENTIALS`
- Check permissions: `chmod 600 $GOOGLE_APPLICATION_CREDENTIALS`
- Verify the environment variable is set: `env | grep GOOGLE_APPLICATION_CREDENTIALS`

### Error: "Permission denied"
- Ensure the Service Account has role **Editor** (not Viewer!)
- Check that the Service Account email is correct
- Verify the Service Account email matches the one you shared the sheet with

### Sheet not created automatically
- Ensure the Service Account has access to the sheet
- Check that the worksheet name is correct (case matters!)

### Credentials not loading
- If using `.env`, remember that system environment variables take precedence
- Restart your terminal after adding to `~/.zshrc` or `~/.bashrc`
- Check credential lookup order (see [Credentials Explanation](#credentials-explanation))

## Security

⚠️ **Important:**
- Never commit the credentials file to git
- Store credentials in a safe place
- Use permissions 600 for the credentials file
- Do not share credentials with others
- Prefer system environment variables over `.env` for credentials
