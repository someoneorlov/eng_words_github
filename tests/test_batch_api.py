"""Tests for Pipeline B batch API adapter (batch_api.py)."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from eng_words.word_family import batch_api


class TestGetClient:
    def test_missing_api_key_raises(self, monkeypatch):
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        with pytest.raises(ValueError, match="GOOGLE_API_KEY"):
            batch_api.get_client()

    def test_with_api_key_returns_client(self, monkeypatch):
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        c = batch_api.get_client(api_key="test-key")
        assert c is not None


class TestUploadRequestsFile:
    def test_missing_file_raises(self, tmp_path: Path):
        mock_client = MagicMock()
        with pytest.raises(FileNotFoundError, match="not found"):
            batch_api.upload_requests_file(mock_client, tmp_path / "missing.jsonl")

    def test_uploads_and_returns_name(self, tmp_path: Path):
        req_path = tmp_path / "requests.jsonl"
        req_path.write_text('{"key":"lemma:a","request":{}}\n', encoding="utf-8")
        mock_client = MagicMock()
        uploaded = MagicMock()
        uploaded.name = "files/abc123"
        mock_client.files.upload.return_value = uploaded
        name = batch_api.upload_requests_file(mock_client, req_path)
        assert name == "files/abc123"
        mock_client.files.upload.assert_called_once()


class TestCreateBatchJob:
    def test_calls_batches_create(self):
        mock_client = MagicMock()
        mock_job = MagicMock()
        mock_job.name = "batches/xyz"
        mock_client.batches.create.return_value = mock_job
        job = batch_api.create_batch_job(
            mock_client,
            model="gemini-3-flash-preview",
            uploaded_file_name="files/abc",
            display_name="pipeline_b_10",
        )
        assert job.name == "batches/xyz"
        mock_client.batches.create.assert_called_once()
        call_kw = mock_client.batches.create.call_args
        assert "models/gemini-3-flash-preview" in str(call_kw)
        assert "files/abc" in str(call_kw)


class TestDownloadBatchResults:
    def test_pending_state_raises(self):
        mock_client = MagicMock()
        mock_job = MagicMock()
        mock_job.state.name = "JOB_STATE_PENDING"
        mock_job.dest = None
        mock_client.batches.get.return_value = mock_job
        with pytest.raises(RuntimeError, match="not complete"):
            batch_api.download_batch_results(mock_client, "batches/xyz")

    def test_succeeded_returns_content(self):
        mock_client = MagicMock()
        mock_job = MagicMock()
        mock_job.state.name = "JOB_STATE_SUCCEEDED"
        mock_job.dest = MagicMock()
        mock_job.dest.file_name = "files/result"
        mock_client.batches.get.return_value = mock_job
        mock_client.files.download.return_value = b'{"key":"lemma:a"}\n'
        content = batch_api.download_batch_results(mock_client, "batches/xyz")
        assert content == b'{"key":"lemma:a"}\n'
        mock_client.files.download.assert_called_once_with(file="files/result")


class TestCallStandardRetry:
    def test_empty_examples_returns_none(self):
        mock_client = MagicMock()
        out = batch_api.call_standard_retry(
            mock_client, "run", {}, "gemini-3-flash", use_thinking=False
        )
        assert out is None
        mock_client.models.generate_content.assert_not_called()

    def test_calls_generate_content_with_retry_prompt(self):
        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.text = '{"cards":[]}'
        mock_client.models.generate_content.return_value = mock_resp
        out = batch_api.call_standard_retry(
            mock_client,
            "run",
            {"run": ["He runs."]},
            "gemini-3-flash",
            use_thinking=False,
        )
        assert out is not None
        assert "candidates" in out
        mock_client.models.generate_content.assert_called_once()
