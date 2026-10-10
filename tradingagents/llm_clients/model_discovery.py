"""Live model discovery for OpenAI-compatible providers.

The static catalog in ``model_catalog`` goes stale as providers ship models.
Interactive front-ends (the Streamlit UI) ask the provider's ``/models``
endpoint instead and fall back to the catalog when it cannot be reached.
"""

from __future__ import annotations

import os
import re

from .api_key_env import get_api_key_env
from .model_catalog import MODEL_OPTIONS
from .openai_client import OPENAI_COMPATIBLE_PROVIDERS

# OpenAI's /models also lists image, audio, embedding, ... endpoints that cannot
# run the agents (they need chat + tool calling), so those are filtered out.
_OPENAI_CHAT_PREFIX = re.compile(r"^(gpt-|chatgpt-|chat-|o\d)")
_OPENAI_NON_CHAT_MARKERS = (
    "image", "audio", "realtime", "transcribe", "tts", "whisper", "embedding",
    "moderation", "instruct", "search", "deep-research", "live", "translate",
)
# Dated snapshots (gpt-5-2025-08-07, gpt-4-0613) duplicate their alias.
_OPENAI_SNAPSHOT = re.compile(r"-(\d{4}-\d{2}-\d{2}|\d{4})$")
_OPENAI_VERSION = re.compile(r"^(?:chat)?gpt-(\d+(?:\.\d+)?)")
_OPENAI_O_SERIES = re.compile(r"^o(\d+)")


def is_openai_chat_model(model_id: str) -> bool:
    """Whether an OpenAI model id is a chat model the agents can run on."""
    return bool(_OPENAI_CHAT_PREFIX.match(model_id)) and not any(
        marker in model_id for marker in _OPENAI_NON_CHAT_MARKERS
    )


def is_openai_snapshot(model_id: str) -> bool:
    """Whether an OpenAI model id is a dated snapshot of an alias."""
    return bool(_OPENAI_SNAPSHOT.search(model_id))


def _openai_sort_key(model_id: str) -> tuple:
    """Newest GPT generation first, then the o-series, then everything else."""
    if match := _OPENAI_VERSION.match(model_id):
        return (0, -float(match.group(1)), model_id)
    if match := _OPENAI_O_SERIES.match(model_id):
        return (1, -float(match.group(1)), model_id)
    return (2, 0.0, model_id)


def _arrange(provider: str, model_ids: list[str], include_snapshots: bool) -> list[str]:
    unique = list(dict.fromkeys(model_ids))
    if provider != "openai":
        return sorted(unique)
    chat = [m for m in unique if is_openai_chat_model(m)]
    if not include_snapshots:
        chat = [m for m in chat if not is_openai_snapshot(m)]
    return sorted(chat, key=_openai_sort_key)


def catalog_models(provider: str, include_snapshots: bool = False) -> list[str]:
    """Model ids from the static catalog (no network), without "custom"."""
    options = MODEL_OPTIONS.get(provider.lower(), {})
    ids = [value for mode in ("deep", "quick") for _, value in options.get(mode, [])]
    return _arrange(provider.lower(), [m for m in ids if m != "custom"], include_snapshots)


def fetch_provider_models(
    provider: str, base_url: str | None = None, timeout: float = 8.0
) -> list[str]:
    """Return the raw model ids served by ``provider``'s ``/models`` endpoint.

    Raises ``ValueError`` when the provider cannot be queried (not
    OpenAI-compatible, missing key or base URL) and the SDK's error on
    network / auth failures.
    """
    provider = provider.lower()
    spec = OPENAI_COMPATIBLE_PROVIDERS.get(provider)
    if spec is None:
        raise ValueError(f"Provider '{provider}' has no OpenAI-compatible /models endpoint.")

    env_base_url = os.environ.get(spec.base_url_env) if spec.base_url_env else None
    resolved_url = base_url or env_base_url or spec.base_url
    if spec.require_base_url and not resolved_url:
        raise ValueError(f"Provider '{provider}' requires a base_url.")

    api_key_env = "OLLAMA_API_KEY" if provider == "ollama" else get_api_key_env(provider)
    api_key = os.environ.get(api_key_env or "", "").strip()
    if not api_key:
        if not spec.key_optional:
            raise ValueError(f"{api_key_env} is not set.")
        api_key = spec.placeholder_key

    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=resolved_url, timeout=timeout, max_retries=0)
    return [model.id for model in client.models.list()]


def list_models(
    provider: str,
    base_url: str | None = None,
    include_snapshots: bool = False,
    timeout: float = 8.0,
) -> tuple[list[str], str | None]:
    """Selectable model ids for ``provider`` plus an error message, if any.

    Tries the live ``/models`` endpoint first. On any failure the static
    catalog is returned together with the reason, so a UI can still offer a
    choice and tell the user the list may be incomplete.
    """
    provider = provider.lower()
    try:
        live = _arrange(
            provider, fetch_provider_models(provider, base_url, timeout), include_snapshots
        )
    except Exception as exc:  # noqa: BLE001 - any failure degrades to the catalog
        return catalog_models(provider, include_snapshots), str(exc)
    if not live:
        return catalog_models(provider, include_snapshots), "Endpoint returned no models."
    return live, None


def check_model_access(provider: str, model: str, base_url: str | None = None) -> str | None:
    """Send one minimal request to ``model``; return the error text, or None if it answers.

    A listed model is not necessarily usable (plan limits, exhausted credits),
    so a UI can call this before starting a long multi-agent run.
    """
    from .factory import create_llm_client

    try:
        create_llm_client(provider, model, base_url, max_retries=0).get_llm().invoke("ok")
    except Exception as exc:  # noqa: BLE001 - report whatever the provider said
        return str(exc)
    return None
