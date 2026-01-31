# Google Sheets setup for Known Words

Step-by-step guide to set up Google Sheets integration for storing your list of known words.

## Step 1: Create Google Cloud project and Service Account

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

## Step 2: Create the Google Sheet

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

## Step 3: Save credentials

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

## Step 4: Set environment variables

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

## Step 5: Test the integration

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

## Step 6: Use in CLI

After a successful test, use in the pipeline:

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

## Troubleshooting

### Error: "Failed to access Google Sheets"
- Check that the Service Account has Editor access to the sheet
- Check that the Spreadsheet ID is correct
- Check that the worksheet name is correct

### Error: "Google credentials not found"
- Check credentials path: `echo $GOOGLE_APPLICATION_CREDENTIALS`
- Check that the file exists: `ls -la $GOOGLE_APPLICATION_CREDENTIALS`
- Check permissions: `chmod 600 $GOOGLE_APPLICATION_CREDENTIALS`

### Error: "Permission denied"
- Ensure the Service Account has role **Editor** (not Viewer!)
- Check that the Service Account email is correct

### Sheet not created automatically
- Ensure the Service Account has access to the sheet
- Check that the worksheet name is correct (case matters!)

## Security

⚠️ **Important:**
- Never commit the credentials file to git
- Store credentials in a safe place
- Use permissions 600 for the credentials file
- Do not share credentials with others
