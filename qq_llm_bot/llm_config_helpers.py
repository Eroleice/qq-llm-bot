from __future__ import annotations

import os
from dataclasses import dataclass

from qq_llm_bot.config_models import LLMConfig, LLMProviderConfig


OPENAI_COMPATIBLE_PROVIDER_TYPES = {"openai", "openai-compatible"}
GEMINI_PROVIDER_TYPES = {"gemini"}
SUPPORTED_TRANSPORT_PROVIDER_TYPES = OPENAI_COMPATIBLE_PROVIDER_TYPES | GEMINI_PROVIDER_TYPES


@dataclass(frozen=True)
class ResolvedLLMModel:
    provider_id: str
    provider_type: str
    model: str
    model_ref: str
    base_url: str
    api_key: str
    api_key_env: str = ""


def split_model_reference(model_ref: str) -> tuple[str, str]:
    clean = str(model_ref or "").strip()
    provider_id, sep, model = clean.partition(":")
    if not sep:
        return "", clean
    return provider_id.strip(), model.strip()


def resolve_api_key(config: LLMConfig, model_ref: str = "") -> str:
    if model_ref:
        try:
            return resolve_model_config(config, model_ref).api_key
        except ValueError:
            return ""
    if config.api_key:
        return config.api_key
    return os.getenv(config.api_key_env, "").strip()


def is_llm_configured(config: LLMConfig) -> bool:
    provider = config.provider.lower().replace("_", "-")
    if provider == "disabled":
        return False
    if not config.model:
        return False
    try:
        resolved = resolve_model_config(config, config.model, require_supported_transport=True)
    except ValueError:
        return False
    return bool(resolved.base_url and resolved.model and resolved.api_key)


def resolve_model_config(
    config: LLMConfig,
    model_ref: str,
    *,
    require_openai_compatible: bool = False,
    require_supported_transport: bool = False,
) -> ResolvedLLMModel:
    provider_id, model = split_model_reference(model_ref)
    if not model:
        raise ValueError("llm.router.chat_generation_model is required")
    if provider_id:
        provider = config.providers.get(provider_id)
        if provider is None:
            raise ValueError(f"provider.json missing provider id: {provider_id}")
        return _resolved_provider_model(
            provider,
            model,
            str(model_ref).strip(),
            require_openai_compatible=require_openai_compatible,
            require_supported_transport=require_supported_transport,
        )

    provider_type = config.provider.lower().replace("_", "-")
    if provider_type in {"", "provider-json", "registry"} and config.providers:
        raise ValueError(f"model must use [provider]:[model] format: {model_ref}")
    if require_openai_compatible and provider_type not in OPENAI_COMPATIBLE_PROVIDER_TYPES:
        raise ValueError(f"unsupported llm.provider for OpenAI-compatible client: {config.provider}")
    if require_supported_transport and provider_type not in SUPPORTED_TRANSPORT_PROVIDER_TYPES:
        raise ValueError(f"unsupported llm.provider type: {config.provider}")
    return ResolvedLLMModel(
        provider_id="",
        provider_type=provider_type,
        model=model,
        model_ref=model,
        base_url=config.base_url,
        api_key=resolve_api_key(config),
        api_key_env=config.api_key_env,
    )


def _resolved_provider_model(
    provider: LLMProviderConfig,
    model: str,
    model_ref: str,
    *,
    require_openai_compatible: bool,
    require_supported_transport: bool,
) -> ResolvedLLMModel:
    provider_type = provider.type.lower().replace("_", "-")
    if provider.models and model not in provider.models:
        raise ValueError(f"provider.json provider {provider.id} does not list model: {model}")
    if require_openai_compatible and provider_type not in OPENAI_COMPATIBLE_PROVIDER_TYPES:
        raise ValueError(
            f"provider.json provider {provider.id} type={provider.type!r} is not supported yet"
        )
    if require_supported_transport and provider_type not in SUPPORTED_TRANSPORT_PROVIDER_TYPES:
        raise ValueError(
            f"provider.json provider {provider.id} type={provider.type!r} is not supported yet"
        )
    return ResolvedLLMModel(
        provider_id=provider.id,
        provider_type=provider_type,
        model=model,
        model_ref=model_ref,
        base_url=provider.url,
        api_key=provider.key_string or os.getenv(provider.key_env, "").strip(),
        api_key_env=provider.key_env,
    )
