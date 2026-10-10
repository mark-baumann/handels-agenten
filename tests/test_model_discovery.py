"""Tests for live model discovery used by the Streamlit model picker."""

from __future__ import annotations

import pytest

from tradingagents.default_config import provider_default_models
from tradingagents.llm_clients import model_discovery
from tradingagents.llm_clients.model_discovery import (
    catalog_models,
    check_model_access,
    fetch_provider_models,
    is_openai_chat_model,
    is_openai_snapshot,
    list_models,
)


@pytest.mark.parametrize(
    "model_id", ["gpt-6-sol", "gpt-5.5", "gpt-5.4-mini", "o3", "o4-mini", "gpt-5.3-codex", "gpt-4o"]
)
def test_openai_chat_models_are_kept(model_id):
    assert is_openai_chat_model(model_id)


@pytest.mark.parametrize(
    "model_id",
    [
        "gpt-image-2", "gpt-audio", "gpt-realtime-2", "gpt-4o-mini-tts", "whisper-1",
        "text-embedding-3-large", "omni-moderation-latest", "sora-2", "dall-e-3",
        "gpt-3.5-turbo-instruct", "gpt-5-search-api", "o3-deep-research", "gpt-live-1",
        "gpt-4o-transcribe", "babbage-002",
    ],
)
def test_openai_non_chat_models_are_dropped(model_id):
    assert not is_openai_chat_model(model_id)


def test_openai_snapshot_detection():
    assert is_openai_snapshot("gpt-5-2025-08-07")
    assert is_openai_snapshot("gpt-4-0613")
    assert not is_openai_snapshot("gpt-5.4-mini")
    assert not is_openai_snapshot("o3")


def _stub_fetch(monkeypatch, ids):
    monkeypatch.setattr(model_discovery, "fetch_provider_models", lambda *a, **k: list(ids))


def test_openai_live_list_is_filtered_and_sorted_newest_first(monkeypatch):
    _stub_fetch(
        monkeypatch,
        ["o3", "gpt-5.5", "gpt-image-2", "gpt-6-sol", "gpt-5-2025-08-07", "gpt-5", "gpt-5.5"],
    )
    models, error = list_models("openai")
    assert error is None
    assert models == ["gpt-6-sol", "gpt-5.5", "gpt-5", "o3"]


def test_openai_snapshots_shown_on_request(monkeypatch):
    _stub_fetch(monkeypatch, ["gpt-5", "gpt-5-2025-08-07"])
    assert list_models("openai", include_snapshots=True)[0] == ["gpt-5", "gpt-5-2025-08-07"]


def test_ollama_cloud_lists_every_served_model(monkeypatch):
    _stub_fetch(monkeypatch, ["kimi-k3", "gpt-oss:120b", "deepseek-v4-pro:0813"])
    models, error = list_models("ollama_cloud")
    assert error is None
    assert models == ["deepseek-v4-pro:0813", "gpt-oss:120b", "kimi-k3"]


def test_failure_falls_back_to_catalog(monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("offline")

    monkeypatch.setattr(model_discovery, "fetch_provider_models", boom)
    models, error = list_models("ollama_cloud")
    assert error == "offline"
    assert models == catalog_models("ollama_cloud")
    assert models and "custom" not in models


def test_fetch_requires_key_for_keyed_provider(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        fetch_provider_models("openai")


def test_fetch_rejects_non_openai_compatible_provider():
    with pytest.raises(ValueError, match="anthropic"):
        fetch_provider_models("anthropic")


def test_provider_default_models():
    assert provider_default_models("ollama_cloud")["deep_think_llm"] == "gpt-oss:120b"
    assert provider_default_models("openai") == {
        "deep_think_llm": "gpt-5.5",
        "quick_think_llm": "gpt-5.4-mini",
    }


def test_check_model_access_reports_provider_error(monkeypatch):
    monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
    assert "OLLAMA_API_KEY" in check_model_access("ollama_cloud", "gpt-oss:20b")
