"""Tests for the dashboard's LLM provider/model selection helpers."""

from __future__ import annotations

from tradingagents.ui_settings import (
    default_model,
    displayed_endpoint,
    load_llm_settings,
    model_choices,
    provider_choices,
    resolve_endpoint,
    save_llm_settings,
    stored_model,
)

_OLLAMA_CLOUD_CONFIG = {
    "llm_provider": "ollama_cloud",
    "deep_think_llm": "my-deep",
    "quick_think_llm": "my-quick",
    "backend_url": None,
}


def test_provider_choices_offer_openai_and_ollama():
    choices = provider_choices(_OLLAMA_CLOUD_CONFIG)
    assert list(choices) == ["openai", "ollama_cloud", "ollama"]


def test_provider_choices_keep_env_configured_provider():
    assert "anthropic" in provider_choices({"llm_provider": "anthropic"})


def test_default_model_uses_config_for_configured_provider():
    assert default_model("ollama_cloud", "deep", _OLLAMA_CLOUD_CONFIG) == "my-deep"
    assert default_model("ollama_cloud", "quick", _OLLAMA_CLOUD_CONFIG) == "my-quick"


def test_default_model_for_other_provider_is_served_by_that_provider():
    # The config's Ollama ids must not leak into an OpenAI run (and vice versa).
    assert default_model("openai", "deep", _OLLAMA_CLOUD_CONFIG) == "gpt-5.5"
    assert default_model("openai", "quick", _OLLAMA_CLOUD_CONFIG) == "gpt-5.4-mini"
    openai_config = {**_OLLAMA_CLOUD_CONFIG, "llm_provider": "openai"}
    assert default_model("ollama", "deep", openai_config) == "gpt-oss:120b"


def test_model_choices_put_extras_first_and_drop_custom_marker():
    choices = model_choices("ollama", "deep", "gpt-oss:120b", "gpt-oss:120b", "")
    assert choices[0] == "gpt-oss:120b"
    assert choices.count("gpt-oss:120b") == 1
    assert "custom" not in choices
    assert "glm-5.2" in choices


def test_backend_url_only_applies_to_configured_provider(monkeypatch):
    config = {**_OLLAMA_CLOUD_CONFIG, "llm_provider": "ollama", "backend_url": "http://box:11434/v1"}
    assert resolve_endpoint("ollama", config) == "http://box:11434/v1"
    assert resolve_endpoint("openai", config) is None
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://pi:11434/v1")
    assert displayed_endpoint("ollama", _OLLAMA_CLOUD_CONFIG) == "http://pi:11434/v1"
    assert displayed_endpoint("ollama_cloud", _OLLAMA_CLOUD_CONFIG) == "https://ollama.com/v1"


def test_settings_roundtrip(tmp_path):
    path = tmp_path / "nested" / "llm_settings.json"
    settings = {"provider": "openai", "models": {"openai": {"deep": "gpt-5.5", "quick": "gpt-5.4-mini"}}}
    save_llm_settings(path, settings)
    loaded = load_llm_settings(path)
    assert loaded == settings
    assert stored_model(loaded, "openai", "deep") == "gpt-5.5"
    assert stored_model(loaded, "ollama", "deep") is None


def test_load_tolerates_missing_and_corrupt_files(tmp_path):
    assert load_llm_settings(tmp_path / "missing.json") == {}
    broken = tmp_path / "broken.json"
    broken.write_text("{not json", encoding="utf-8")
    assert load_llm_settings(broken) == {}
    broken.write_text("[1, 2]", encoding="utf-8")
    assert load_llm_settings(broken) == {}
    assert stored_model({"models": {"openai": "oops"}}, "openai", "deep") is None
