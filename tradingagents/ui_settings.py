"""LLM selection for the Streamlit dashboard.

The dashboard lets the user switch between a small set of providers (OpenAI
and Ollama) and pick the models per provider. The choice is persisted as JSON
next to the portfolio database so it survives restarts and redeployments.

Credentials and endpoints are deliberately *not* part of these settings: they
stay in the server environment, so the dashboard can never be used to point a
stored API key at a foreign host.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from tradingagents.default_config import _PROVIDER_DEFAULT_MODELS
from tradingagents.llm_clients.model_catalog import MODEL_OPTIONS

# Providers offered in the dashboard, in display order.
UI_PROVIDERS: dict[str, str] = {
    "openai":       "OpenAI (ChatGPT)",
    "ollama_cloud": "Ollama Cloud",
    "ollama":       "Ollama (eigener Server)",
}

_OPENAI_DEFAULT_MODELS = {"deep_think_llm": "gpt-5.5", "quick_think_llm": "gpt-5.4-mini"}
_MODE_KEYS = {"deep": "deep_think_llm", "quick": "quick_think_llm"}


def provider_choices(config: dict) -> dict[str, str]:
    """Selectable providers; the env-configured one is always included."""
    choices = dict(UI_PROVIDERS)
    configured = str(config.get("llm_provider") or "").lower()
    if configured and configured not in choices:
        choices[configured] = configured
    return choices


def default_model(provider: str, mode: str, config: dict) -> str:
    """Default model id for ``provider`` in ``mode`` ("deep" or "quick")."""
    key = _MODE_KEYS[mode]
    if provider == str(config.get("llm_provider") or "").lower():
        return config[key]
    if provider == "openai":
        return _OPENAI_DEFAULT_MODELS[key]
    if provider in _PROVIDER_DEFAULT_MODELS:
        return _PROVIDER_DEFAULT_MODELS[provider][key]
    options = model_choices(provider, mode)
    return options[0] if options else ""


def model_choices(provider: str, mode: str, *extra: str) -> list[str]:
    """Catalog model ids for ``provider``, preceded by ``extra`` ids."""
    catalog = MODEL_OPTIONS.get(provider, {}).get(mode, [])
    ids = [*extra, *(model_id for _, model_id in catalog if model_id != "custom")]
    return [model_id for model_id in dict.fromkeys(ids) if model_id]


def resolve_endpoint(provider: str, config: dict) -> str | None:
    """Backend URL to pass to the client, or None for the provider default.

    ``backend_url`` from the environment belongs to the env-configured
    provider only; forwarding it to another provider would send requests (and
    that provider's key) to the wrong host.
    """
    if provider == str(config.get("llm_provider") or "").lower():
        return config.get("backend_url") or None
    return None


def displayed_endpoint(provider: str, config: dict) -> str:
    """Human-readable endpoint the selected provider will talk to."""
    endpoint = resolve_endpoint(provider, config)
    if endpoint:
        return endpoint
    if provider == "ollama":
        return os.environ.get("OLLAMA_BASE_URL") or "http://localhost:11434/v1"
    if provider == "ollama_cloud":
        return "https://ollama.com/v1"
    return "Provider-Standard"


def load_llm_settings(path: Path) -> dict:
    """Read persisted settings; a missing or unreadable file yields ``{}``."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def save_llm_settings(path: Path, settings: dict) -> None:
    """Persist settings atomically."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(settings, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def stored_model(settings: dict, provider: str, mode: str) -> str | None:
    """Model id previously saved for ``provider``/``mode``, if any."""
    models = settings.get("models")
    if not isinstance(models, dict):
        return None
    value = (models.get(provider) or {}).get(mode) if isinstance(models.get(provider), dict) else None
    return value if isinstance(value, str) and value.strip() else None
