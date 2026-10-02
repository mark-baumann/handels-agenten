"""Shared model catalog for CLI selections and validation."""

from __future__ import annotations

ModelOption = tuple[str, str]
ProviderModeOptions = dict[str, dict[str, list[ModelOption]]]

# Providers that serve many / frequently-changing models: offer only "Custom
# model ID" rather than a list that goes stale.
_CUSTOM_ONLY: dict[str, list[ModelOption]] = {
    "quick": [("Custom model ID", "custom")],
    "deep": [("Custom model ID", "custom")],
}


# Shared model list for GLM via Z.AI (international) and BigModel (China).
# Source: docs.z.ai (GLM Coding Plan supported models + LLM guides).
# All GLM 4.7+ entries support thinking mode via thinking={"type":"enabled"}.
_GLM_MODELS: dict[str, list[ModelOption]] = {
    "quick": [
        ("GLM-5-Turbo - Fast, switchable thinking modes", "glm-5-turbo"),
        ("GLM-4.7 - Previous-gen flagship", "glm-4.7"),
        ("GLM-4.5-Air - Lightweight, cost-efficient", "glm-4.5-air"),
        ("Custom model ID", "custom"),
    ],
    "deep": [
        ("GLM-5.2 - Latest flagship, 1M ctx", "glm-5.2"),
        ("GLM-5.1 - 745B, 200K ctx", "glm-5.1"),
        ("GLM-5 - Flagship, 204K ctx", "glm-5"),
        ("GLM-4.7 - Previous-gen flagship", "glm-4.7"),
        ("Custom model ID", "custom"),
    ],
}


# Shared model list for Qwen's global (dashscope-intl) and CN (dashscope) endpoints.
# Source: modelstudio.console.alibabacloud.com (Featured Models — Flagship + Cost-optimized).
#
# Only versioned IDs are exposed in the dropdown. The version-less aliases
# (qwen-plus, qwen-flash) are documented by Alibaba as auto-upgrading
# pointers ("backbone, latest, and snapshot ... have been upgraded to the
# Qwen3 series"), which means their behavior shifts when Alibaba rotates
# the backing model. Users who want a specific generation pick it
# explicitly; users who really want auto-latest can enter the alias via
# "Custom model ID".
_QWEN_MODELS: dict[str, list[ModelOption]] = {
    "quick": [
        ("Qwen 3.7 Plus - Latest, balanced speed/cost", "qwen3.7-plus"),
        ("Qwen 3.6 Plus - Previous-gen balanced", "qwen3.6-plus"),
        ("Custom model ID", "custom"),
    ],
    "deep": [
        ("Qwen 3.7 Max - Latest flagship, most intelligent, 1M ctx", "qwen3.7-max"),
        ("Qwen 3.6 Max - Previous-gen flagship", "qwen3.6-max"),
        ("Qwen 3.7 Plus - Balanced alternative", "qwen3.7-plus"),
        ("Custom model ID", "custom"),
    ],
}


# Shared model list for MiniMax's global and CN endpoints (same IDs).
# Full official lineup per platform.minimax.io/docs/api-reference/text-openai-api.
# M3 carries a 1M-token context window; the M2.x line is 204,800 tokens.
_MINIMAX_MODELS: dict[str, list[ModelOption]] = {
    "quick": [
        ("MiniMax-M3 - Latest, 1M ctx, native multimodal", "MiniMax-M3"),
        ("MiniMax-M2.7-highspeed - Fast M2.7, 204K ctx, ~100 TPS", "MiniMax-M2.7-highspeed"),
        ("MiniMax-M2.5-highspeed - Previous-gen highspeed, 204K ctx", "MiniMax-M2.5-highspeed"),
        ("Custom model ID", "custom"),
    ],
    "deep": [
        ("MiniMax-M3 - Latest flagship, 1M ctx, multimodal coding/agent", "MiniMax-M3"),
        ("MiniMax-M2.7 - Previous flagship, 204K ctx", "MiniMax-M2.7"),
        ("MiniMax-M2.7-highspeed - Same quality as M2.7, ~100 TPS", "MiniMax-M2.7-highspeed"),
        ("MiniMax-M2.5 - Earlier flagship, 204K ctx", "MiniMax-M2.5"),
        ("Custom model ID", "custom"),
    ],
}


# OpenAI model IDs supplied for the UI. This deliberately includes legacy,
# realtime, audio, image, embedding, moderation, and text-to-speech endpoints so
# users can select every model available in their account. The TradingAgents
# workflow itself requires a chat-capable model; the Streamlit UI explains that
# limitation before an analysis can be started.
_OPENAI_MODEL_IDS = (
    "dall-e-2", "omni-moderation-latest", "o3-pro-2025-06-10",
    "gpt-4o-2024-11-20", "gpt-4o-mini-search-preview",
    "gpt-4o-mini-search-preview-2025-03-11", "gpt-4o-realtime-preview",
    "gpt-4-turbo", "gpt-4o-2024-05-13", "o4-mini-2025-04-16",
    "gpt-4.1-2025-04-14", "o3-2025-04-16", "gpt-4-turbo-2024-04-09",
    "gpt-4.1-nano-2025-04-14", "gpt-4.1-mini", "gpt-5-nano-2025-08-07",
    "gpt-4.1-mini-2025-04-14", "gpt-4.1", "o3-mini-2025-01-31",
    "gpt-4o-search-preview-2025-03-11", "gpt-3.5-turbo-16k",
    "gpt-4o-search-preview", "o1-mini", "gpt-4.1-nano", "o1-mini-2024-09-12",
    "gpt-image-1", "gpt-4o-mini-2024-07-18",
    "gpt-4o-mini-realtime-preview-2024-12-17", "gpt-4o-mini-transcribe",
    "o3", "o4-mini", "gpt-4o-mini-audio-preview",
    "gpt-4o-mini-audio-preview-2024-12-17", "gpt-5-chat-latest",
    "gpt-4o-mini-realtime-preview", "gpt-4o-audio-preview-2024-10-01",
    "o4-mini-deep-research-2025-06-26", "codex-mini-latest",
    "gpt-4o-realtime-preview-2024-10-01", "gpt-5-nano", "babbage-002",
    "tts-1-hd", "gpt-4-turbo-preview", "o3-deep-research", "tts-1-hd-1106",
    "chatgpt-4o-latest", "gpt-5-mini-2025-08-07", "gpt-4o-mini-tts",
    "gpt-audio-2025-08-28", "o1-pro-2025-03-19",
    "gpt-4o-audio-preview-2024-12-17", "o1", "dall-e-3", "davinci-002",
    "o1-pro", "gpt-4-0613", "gpt-4-0125-preview", "o3-pro",
    "o3-deep-research-2025-06-26", "o4-mini-deep-research",
    "gpt-4o-realtime-preview-2024-12-17", "gpt-realtime", "gpt-4o-mini",
    "whisper-1", "gpt-realtime-2025-08-28", "text-embedding-ada-002",
    "o3-mini", "gpt-audio", "gpt-4o-realtime-preview-2025-06-03",
    "gpt-3.5-turbo-1106", "text-embedding-3-small", "gpt-5",
    "gpt-4o-transcribe", "gpt-3.5-turbo-instruct", "gpt-3.5-turbo-instruct-0914",
    "text-embedding-3-large", "gpt-4-1106-preview", "tts-1", "tts-1-1106",
    "gpt-5-codex", "gpt-4o", "gpt-5-mini", "gpt-4o-audio-preview",
    "gpt-4o-audio-preview-2025-06-03", "gpt-5-2025-08-07", "gpt-4",
    "gpt-4o-2024-08-06", "o1-2024-12-17", "gpt-3.5-turbo",
    "gpt-3.5-turbo-0125", "omni-moderation-2024-09-26", "gpt-5.6-sol",
    # Keep current project defaults selectable as well.
    "gpt-5.5", "gpt-5.4", "gpt-5.4-mini", "gpt-5.4-nano", "gpt-5.2", "gpt-5.5-pro",
)
_OPENAI_MODELS = [(model, model) for model in dict.fromkeys(_OPENAI_MODEL_IDS)]


MODEL_OPTIONS: ProviderModeOptions = {
    "openai": {
        "quick": _OPENAI_MODELS,
        "deep": _OPENAI_MODELS,
    },
    "anthropic": {
        "quick": [
            ("Claude Sonnet 5 - Best speed and intelligence balance", "claude-sonnet-5"),
            ("Claude Haiku 4.5 - Fastest with near-frontier intelligence", "claude-haiku-4-5"),
        ],
        "deep": [
            ("Claude Fable 5 - Most capable, long-running agents", "claude-fable-5"),
            ("Claude Opus 4.8 - Frontier agentic coding and reasoning", "claude-opus-4-8"),
            ("Claude Sonnet 5 - Near-frontier intelligence at Sonnet cost", "claude-sonnet-5"),
            ("Claude Opus 4.7 - Previous frontier, long-running agents", "claude-opus-4-7"),
        ],
    },
    "google": {
        "quick": [
            ("Gemini 3.5 Flash - Latest, frontier agentic + coding (GA)", "gemini-3.5-flash"),
            ("Gemini 3.1 Flash Lite - Most cost-efficient", "gemini-3.1-flash-lite"),
        ],
        "deep": [
            ("Gemini 3.1 Pro - Reasoning-first, complex workflows (preview)", "gemini-3.1-pro-preview"),
            ("Gemini 3.5 Flash - Latest GA, strong agentic + coding", "gemini-3.5-flash"),
        ],
    },
    "xai": {
        "quick": [
            ("Grok 4.3 - Latest flagship, fast with built-in reasoning", "grok-4.3"),
            ("Grok 4.20 (Non-Reasoning) - Speed-optimized", "grok-4.20-0309-non-reasoning"),
            ("Grok Build 0.1 - Coding-specialized, 256K ctx", "grok-build-0.1"),
        ],
        "deep": [
            ("Grok 4.3 - Latest flagship, built-in reasoning, 1M ctx", "grok-4.3"),
            ("Grok 4.20 (Reasoning) - Previous-gen reasoning", "grok-4.20-0309-reasoning"),
            ("Grok 4.20 Multi-Agent - Multi-agent reasoning", "grok-4.20-multi-agent-0309"),
        ],
    },
    # DeepSeek: the deepseek-chat / deepseek-reasoner aliases are deprecated
    # (2026-07-24) and now map to V4 Flash; expose the V4 IDs directly. V4 Flash
    # serves both non-thinking and thinking modes (the DeepSeekChatOpenAI client
    # handles the reasoning_content round-trip).
    "deepseek": {
        "quick": [
            ("DeepSeek V4 Flash - Latest fast model, thinking + non-thinking", "deepseek-v4-flash"),
            ("Custom model ID", "custom"),
        ],
        "deep": [
            ("DeepSeek V4 Pro - Latest flagship", "deepseek-v4-pro"),
            ("DeepSeek V4 Flash - Fast, supports thinking", "deepseek-v4-flash"),
            ("Custom model ID", "custom"),
        ],
    },
    # Qwen: same model IDs across global (dashscope-intl) and China
    # (dashscope) endpoints, so the two provider keys share one model list.
    "qwen": _QWEN_MODELS,
    "qwen-cn": _QWEN_MODELS,
    # GLM: Z.AI (international) and BigModel (China) host the same model
    # IDs; the two provider keys share one model list.
    "glm": _GLM_MODELS,
    "glm-cn": _GLM_MODELS,
    # MiniMax: same model IDs across global (.io) and China (.com) regions,
    # so the two provider keys share one model list.
    "minimax": _MINIMAX_MODELS,
    "minimax-cn": _MINIMAX_MODELS,
    # OpenRouter: fetched dynamically. Azure: any deployed model name.
    # Ollama display labels intentionally omit a "local" marker — the
    # endpoint is now configurable via OLLAMA_BASE_URL, so the same labels
    # apply whether the user runs ollama-serve on localhost or against a
    # remote host. The actual resolved endpoint is surfaced separately by
    # cli.utils.confirm_ollama_endpoint() right after provider selection.
    # "Custom model ID" lets users pick any model they have pulled via
    # `ollama pull` beyond the three suggested defaults.
    # Defaults for Ollama Cloud (https://ollama.com/v1) so the shipped dropdown
    # actually resolves without a `model not found` 404. The endpoint serves
    # these model ids (no `:latest` tag suffix); local Ollama installs can pull
    # any `ollama pull <name>` and pick it via "Custom model ID".
    "ollama": {
        "quick": [
            ("GLM-5.3-Flash - Fast, switchable thinking", "glm-5.3-flash"),
            ("GPT-OSS (20B)", "gpt-oss:20b"),
            ("Qwen3.5 (397B)", "qwen3.5:397b"),
            ("Custom model ID", "custom"),
        ],
        "deep": [
            ("GLM-5.2 - Latest flagship", "glm-5.2"),
            ("GLM-5.1 - 745B, 200K ctx", "glm-5.1"),
            ("GLM-5.3-Flash - Fast, switchable thinking", "glm-5.3-flash"),
            ("Custom model ID", "custom"),
        ],
    },
    # Generic OpenAI-compatible endpoint: the model is whatever the user's
    # server serves, so only "Custom model ID" is offered.
    "openai_compatible": _CUSTOM_ONLY,
    # Hosted OpenAI-compatible providers that serve many (and frequently
    # changing) models — offer "Custom model ID" rather than a list that goes
    # stale. The endpoint + key are wired by the provider; the user picks the
    # model their account has access to.
    "mistral": _CUSTOM_ONLY,
    "kimi": _CUSTOM_ONLY,
    "groq": _CUSTOM_ONLY,
    "nvidia": _CUSTOM_ONLY,
    # Azure deployments use the deployment name configured by the user.
    "azure": _CUSTOM_ONLY,
    # Bedrock model IDs / cross-region inference profile IDs are user-specified.
    "bedrock": _CUSTOM_ONLY,
}


def get_model_options(provider: str, mode: str) -> list[ModelOption]:
    """Return shared model options for a provider and selection mode."""
    return MODEL_OPTIONS[provider.lower()][mode]


def get_known_models() -> dict[str, list[str]]:
    """Build known model names from the shared CLI catalog."""
    return {
        provider: sorted(
            {
                value
                for options in mode_options.values()
                for _, value in options
            }
        )
        for provider, mode_options in MODEL_OPTIONS.items()
    }
