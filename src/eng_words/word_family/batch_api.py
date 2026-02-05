"""Pipeline B batch — Gemini API adapter (network only).

Thin wrappers over google.genai for: get_client, upload, create batch job,
download results, standard API retry. Injectable client for testing (mocks).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from eng_words.word_family.batch_core import build_retry_prompt


def get_client(api_key: str | None = None):
    """Return authenticated Gemini client. Raises ValueError if no API key."""
    key = api_key or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise ValueError("GOOGLE_API_KEY not set. Set it in .env or environment.")
    from google import genai
    return genai.Client(api_key=key)


def upload_requests_file(client: Any, requests_path: Path) -> str:
    """Upload requests.jsonl to Gemini Files. Returns uploaded file name (e.g. files/xxx)."""
    from google.genai import types
    path = Path(requests_path)
    if not path.exists():
        raise FileNotFoundError(f"Requests file not found: {path}")
    uploaded = client.files.upload(
        file=str(path),
        config=types.UploadFileConfig(mime_type="application/jsonl"),
    )
    return uploaded.name


def create_batch_job(
    client: Any,
    model: str,
    uploaded_file_name: str,
    display_name: str,
) -> Any:
    """Create a batch job. Returns job object with .name, .state, .dest."""
    from google.genai import types
    return client.batches.create(
        model=f"models/{model}",
        src=uploaded_file_name,
        config=types.CreateBatchJobConfig(display_name=display_name),
    )


def download_batch_results(client: Any, batch_name: str) -> bytes:
    """Fetch batch job, ensure SUCCEEDED, download results file content. Raises on failure."""
    job = client.batches.get(name=batch_name)
    if job.state.name != "JOB_STATE_SUCCEEDED":
        raise RuntimeError(f"Batch not complete: {job.state.name}")
    if not getattr(job, "dest", None) or not job.dest:
        raise RuntimeError("No destination file in batch job.")
    return client.files.download(file=job.dest.file_name)


def generate_content_for_prompt(
    client: Any,
    model: str,
    prompt: str,
) -> dict | None:
    """Call Gemini Standard API with one prompt (main cluster prompt). Returns response in batch shape or None.
    Use for small test runs instead of Batch API (no job queue, immediate results).
    """
    from google.genai import types
    config_kwargs: dict = {
        "temperature": 0,
        "max_output_tokens": 16384,
        "response_mime_type": "application/json",
    }
    if "flash" in model.lower():
        config_kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
    try:
        response = client.models.generate_content(
            model=f"models/{model}",
            contents=prompt,
            config=types.GenerateContentConfig(**config_kwargs),
        )
    except Exception:
        return None
    text = (response.text or "").strip()
    if not text:
        return None
    return {"candidates": [{"content": {"parts": [{"text": text}]}}]}


def call_standard_retry(
    client: Any,
    lemma: str,
    lemma_examples: dict[str, list[str]],
    model: str,
    thinking_model: str = "gemini-2.5-pro",
    use_thinking: bool = False,
) -> dict | None:
    """Call Gemini Standard API with retry prompt. Returns response dict (batch shape) or None.
    When use_thinking=True uses thinking_model (Pro with Thinking), not Flash."""
    from google.genai import types
    examples = lemma_examples.get(lemma, [])
    if not examples:
        return None
    prompt = build_retry_prompt(lemma, examples)
    actual_model = thinking_model if use_thinking else model
    config_kwargs: dict = {
        "temperature": 0,
        "max_output_tokens": 16384,
        "response_mime_type": "application/json",
    }
    if "flash" in actual_model.lower():
        config_kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
    try:
        response = client.models.generate_content(
            model=f"models/{actual_model}",
            contents=prompt,
            config=types.GenerateContentConfig(**config_kwargs),
        )
    except Exception:
        return None
    text = (response.text or "").strip()
    if not text:
        return None
    return {"candidates": [{"content": {"parts": [{"text": text}]}}]}
