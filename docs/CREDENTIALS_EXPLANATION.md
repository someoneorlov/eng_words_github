# How GOOGLE_APPLICATION_CREDENTIALS works

## Overview

`GOOGLE_APPLICATION_CREDENTIALS` is an environment variable that points to the path of a JSON file containing credentials for a Google Service Account.

## How it works in our code

### 1. Credential lookup order

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

### 2. Where to set GOOGLE_APPLICATION_CREDENTIALS

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

### 3. Why is it commented out in .env.example?

In `.env.example` the line is commented because:

1. **Credentials are usually set in system variables** (`~/.zshrc`)
2. **No need to duplicate** in each project
3. **Safer** — credentials won't end up in the repo by mistake

### 4. How to verify credentials work?

```bash
# Check environment variable
echo $GOOGLE_APPLICATION_CREDENTIALS

# Check that file exists
ls -la $GOOGLE_APPLICATION_CREDENTIALS

# Run test
python scripts/test_gsheets.py
```

### 5. Full setup example

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

## Summary

- **GOOGLE_APPLICATION_CREDENTIALS** — path to the JSON credentials file
- Set in **system environment** (`~/.zshrc`)
- **No need** to set in `.env` (but you can if you prefer)
- The code finds credentials automatically via the environment variable
