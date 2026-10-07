"""Tests for the hosted Ollama Cloud provider (``ollama_cloud``)."""

from __future__ import annotations

import pytest

from tradingagents.llm_clients.api_key_env import get_api_key_env
from tradingagents.llm_clients.factory import create_llm_client
from tradingagents.llm_clients.model_catalog import get_model_options


def test_ollama_cloud_uses_ollama_api_key_env():
    assert get_api_key_env("ollama_cloud") == "OLLAMA_API_KEY"


def test_ollama_cloud_defaults_to_hosted_endpoint(monkeypatch):
    monkeypatch.setenv("OLLAMA_API_KEY", "secret")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
    llm = create_llm_client("ollama_cloud", "gpt-oss:120b").get_llm()
    assert str(llm.openai_api_base) == "https://ollama.com/v1"
    assert llm.openai_api_key.get_secret_value() == "secret"


def test_ollama_cloud_requires_key(monkeypatch):
    monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
    with pytest.raises(ValueError, match="OLLAMA_API_KEY"):
        create_llm_client("ollama_cloud", "gpt-oss:120b").get_llm()


def test_ollama_cloud_has_model_options():
    assert get_model_options("ollama_cloud", "deep")
    assert get_model_options("ollama_cloud", "quick")
