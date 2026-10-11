"""Tests for the dashboard's LLM provider/model selection helpers."""

from __future__ import annotations

from tradingagents.ui_settings import (
    default_model,
    displayed_endpoint,
    load_llm_settings,
    model_choices,
    provider_choices,
    resolve_endpoint,
    run_config,
    save_llm_settings,
    selected_provider,
    stored_model,
)

_OLLAMA_CLOUD_CONFIG = {
    "llm_provider": "ollama_cloud",
    "deep_think_llm": "my-deep",
    "quick_think_llm": "my-quick",
    "backend_url": None,
}


def test_provider_choices_offer_ollama_cloud_first_without_local_ollama():
    choices = provider_choices(_OLLAMA_CLOUD_CONFIG)
    assert list(choices) == ["ollama_cloud", "openai"]
    assert "ollama" not in provider_choices({"llm_provider": "ollama"})


def test_selected_provider_defaults_to_ollama_cloud():
    assert selected_provider({}, {"llm_provider": "openai"}) == "ollama_cloud"
    assert selected_provider({"provider": "ollama"}, {"llm_provider": "openai"}) == "ollama_cloud"
    assert selected_provider({"provider": "openai"}, {"llm_provider": "openai"}) == "openai"


def test_run_config_defaults_to_free_gpt_oss_120b():
    base = {"llm_provider": "openai", "deep_think_llm": "gpt-5.5",
            "quick_think_llm": "gpt-5.4-mini", "backend_url": "https://x/v1", "other": 1}
    config = run_config({}, base)
    assert config["llm_provider"] == "ollama_cloud"
    assert config["deep_think_llm"] == "gpt-oss:120b"
    assert config["quick_think_llm"] == "gpt-oss:120b"
    assert config["backend_url"] is None  # env endpoint belongs to openai only
    assert config["other"] == 1
    saved = {"provider": "openai", "models": {"openai": {"deep": "gpt-5.5", "quick": "x"}}}
    config = run_config(saved, base)
    assert (config["deep_think_llm"], config["quick_think_llm"]) == ("gpt-5.5", "x")
    assert config["backend_url"] == "https://x/v1"


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
    assert default_model("ollama_cloud", "deep", openai_config) == "gpt-oss:120b"
    assert default_model("ollama_cloud", "quick", openai_config) == "gpt-oss:120b"


def test_model_choices_put_extras_first_and_drop_custom_marker():
    choices = model_choices("ollama", "deep", "gpt-oss:120b", "gpt-oss:120b", "")
    assert choices[0] == "gpt-oss:120b"
    assert choices.count("gpt-oss:120b") == 1
    assert "custom" not in choices
    assert "glm-5.2" in choices


def test_backend_url_only_applies_to_configured_provider():
    config = {**_OLLAMA_CLOUD_CONFIG, "llm_provider": "ollama", "backend_url": "http://box:11434/v1"}
    assert resolve_endpoint("ollama", config) == "http://box:11434/v1"
    assert resolve_endpoint("openai", config) is None
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
