# LLM provider API keys setup

The WSD Gold Dataset pipeline requires API keys from three providers:
- OpenAI (GPT)
- Anthropic (Claude)
- Google (Gemini)

## 1. OpenAI API Key

### Getting a key:
1. Go to https://platform.openai.com/
2. Sign in or create an account
3. Go to **Settings → API Keys** (https://platform.openai.com/api-keys)
4. Click **"Create new secret key"**
5. Copy the key (starts with `sk-...`)

### Adding credits:
1. Go to **Settings → Billing** (https://platform.openai.com/settings/organization/billing/overview)
2. Add a payment method
3. Recommended initial balance: **$5–10**

### Pricing (Jan 2026):
| Model | Input (1M tokens) | Output (1M tokens) |
|--------|-------------------|-------------------|
| gpt-5-nano | $0.06 | $0.48 |
| gpt-5-mini (default) | $0.30 | $2.40 |
| gpt-5 | $1.25 | $10.00 |
| gpt-5.2 | $2.00 | $16.00 |

**Estimate for Gold Dataset**: ~$0.25 for 500 examples (gpt-5-mini)

---

## 2. Anthropic API Key

### Getting a key:
1. Go to https://console.anthropic.com/
2. Sign in or create an account (waitlist may apply)
3. Go to **Settings → API Keys** (https://console.anthropic.com/settings/keys)
4. Click **"Create Key"**
5. Copy the key (starts with `sk-ant-...`)

### Adding credits:
1. Go to **Plans & Billing** (https://console.anthropic.com/settings/billing)
2. Add a payment method
3. Recommended initial balance: **$5–10**

### Pricing (Jan 2026):
| Model | Input (1M tokens) | Output (1M tokens) |
|--------|-------------------|-------------------|
| claude-haiku-4-5-20251001 (default) | $1.00 | $5.00 |
| claude-sonnet-4-5-20250929 | $3.00 | $15.00 |
| claude-opus-4-5-20251101 | $5.00 | $25.00 |

**Estimate for Gold Dataset**: ~$0.50 for 500 examples (claude-haiku-4-5)

---

## 3. Google Gemini API Key

### Getting a key:
1. Go to https://aistudio.google.com/
2. Sign in with your Google account
3. Click **"Get API Key"** in the left menu
4. Select an existing project or create a new one
5. Click **"Create API key"**
6. Copy the key (starts with `AIza...`)

### Alternative via Google Cloud:
1. Go to https://console.cloud.google.com/
2. Create a project
3. Enable **Generative Language API**
4. Create an API key under **APIs & Services → Credentials**

### Pricing (Jan 2026):
| Model | Input (1M tokens) | Output (1M tokens) |
|--------|-------------------|-------------------|
| gemini-2.0-flash (default) | $0.10 | $0.40 |
| gemini-2.5-flash | $0.08 | $0.32 |
| gemini-2.5-pro | $1.25 | $5.00 |
| gemini-3-pro-preview | $1.50 | $6.00 |

**Estimate for Gold Dataset**: ~$0.05 for 500 examples (gemini-2.0-flash)

⚠️ **Free tier**: Gemini has a free limit of 15 RPM (requests per minute), enough for testing.

---

## 4. .env file setup

Create a `.env` file in the project root:

```bash
# OpenAI
OPENAI_API_KEY=sk-...

# Anthropic
ANTHROPIC_API_KEY=sk-ant-...

# Google Gemini
GOOGLE_API_KEY=AIza...
```

### Verifying keys:

```bash
# Run test with real API calls
python scripts/test_gold_pipeline_real_data.py --with-llm
```

Expected output for each provider:
```
🔵 Testing OpenAI provider...
✅ OpenAI provider works!

🟠 Testing Anthropic provider...
✅ Anthropic provider works!

🟢 Testing Gemini provider...
✅ Gemini provider works!
```

---

## 5. Dependencies

Ensure required packages are installed:

```bash
# Install all LLM dependencies
pip install -e ".[llm]"

# Or manually:
pip install openai>=2.11.0
pip install anthropic>=0.75.0
pip install google-genai>=1.0.0
```

---

## 6. Troubleshooting

### OpenAI: "Incorrect API key provided"
- Check that the key starts with `sk-`
- Ensure balance > $0

### Anthropic: "Invalid API Key"
- Check that the key starts with `sk-ant-`
- Ensure the account is activated

### Gemini: "API key not valid"
- Check that Generative Language API is enabled for the project
- Try recreating the key

### Gemini: "RESOURCE_EXHAUSTED" / 429
- Free tier limit exceeded
- Check quota at https://ai.dev/rate-limit
- Wait ~1 minute and try again
- Or switch to a paid plan

### Rate limits
- OpenAI: New accounts have low limits (Tier 1)
- Anthropic: Batch API bypasses rate limits
- Gemini: 15 RPM on free tier

---

## 7. Gold Dataset cost estimate

For 500 examples × 3 providers:

| Provider | Model | Cost |
|-----------|--------|-----------|
| OpenAI | gpt-5-mini | ~$0.25 |
| Anthropic | claude-haiku-4-5 | ~$0.50 |
| Gemini | gemini-2.0-flash | ~$0.05 |
| **Total** | | **~$0.80** |

For full dataset (1000+ examples): **~$1–2**
