from __future__ import annotations

from qq_llm_bot.config import LLMConfig
from qq_llm_bot.llm_config_helpers import is_llm_configured, resolve_api_key
from qq_llm_bot.llm_models import (
    DisabledLLMClient,
    GeneratedImage,
    LLMClient,
    LLMUsageRecord,
    LLMUsageRecorder,
)
from qq_llm_bot.llm_openai_client import OpenAICompatibleLLMClient
from qq_llm_bot.llm_response_helpers import (
    _image_generation_input,
    extract_generated_image,
    normalize_chat_completions_url,
    normalize_responses_url,
)

__all__ = [
    "GeneratedImage",
    "LLMClient",
    "LLMUsageRecord",
    "LLMUsageRecorder",
    "OpenAICompatibleLLMClient",
    "_image_generation_input",
    "build_llm_client",
    "extract_generated_image",
    "is_llm_configured",
    "normalize_chat_completions_url",
    "normalize_responses_url",
    "resolve_api_key",
]


def build_llm_client(
    config: LLMConfig,
    usage_recorder: LLMUsageRecorder | None = None,
) -> LLMClient:
    provider = config.provider.lower().replace("_", "-")
    if provider == "disabled":
        return DisabledLLMClient()
    if provider in {"openai-compatible", "openai"}:
        return OpenAICompatibleLLMClient(config, usage_recorder=usage_recorder)
    raise ValueError(
        f"Unsupported llm.provider={config.provider!r}. "
        "Supported providers: disabled, openai-compatible."
    )
