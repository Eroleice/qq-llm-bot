from __future__ import annotations

import os

from qq_llm_bot.config import LLMConfig


def resolve_api_key(config: LLMConfig) -> str:
    if config.api_key:
        return config.api_key
    return os.getenv(config.api_key_env, "").strip()


def is_llm_configured(config: LLMConfig) -> bool:
    provider = config.provider.lower().replace("_", "-")
    if provider == "disabled":
        return False
    return bool(config.base_url and config.model and resolve_api_key(config))
