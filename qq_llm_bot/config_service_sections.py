from __future__ import annotations

from typing import Any

from qq_llm_bot.config_models import (
    DashboardConfig,
    LLMConfig,
    LLMProviderConfig,
    LLMRoutingConfig,
    StorageConfig,
)
from qq_llm_bot.config_values import (
    bool_value as _bool_value,
    float_in_range as _float_in_range,
    positive_int as _positive_int,
    route_prefix as _route_prefix,
)


def dashboard_config(raw: dict[str, Any]) -> DashboardConfig:
    return DashboardConfig(
        enabled=_bool_value(raw.get("enabled", True)),
        route_prefix=_route_prefix(raw.get("route_prefix", "/dashboard")),
        api_prefix=_route_prefix(raw.get("api_prefix", "/api/dashboard")),
        access_token=str(raw.get("access_token", "")).strip(),
        access_token_env=str(raw.get("access_token_env", "QQ_LLM_BOT_DASHBOARD_TOKEN")).strip()
        or "QQ_LLM_BOT_DASHBOARD_TOKEN",
    )


def storage_config(raw: dict[str, Any]) -> StorageConfig:
    return StorageConfig(
        sqlite_path=str(raw.get("sqlite_path", "data/bot.sqlite3")).strip() or "data/bot.sqlite3",
    )


def llm_config(
    raw: dict[str, Any],
    routing_raw: dict[str, Any],
    providers: dict[str, LLMProviderConfig] | None = None,
) -> LLMConfig:
    chat_generation_model = str(routing_raw.get("chat_generation_model", "")).strip()
    clean_provider = str(raw.get("provider", "")).strip()
    if not clean_provider:
        clean_provider = "provider-json" if providers and chat_generation_model else "disabled"
    return LLMConfig(
        provider=clean_provider,
        model=chat_generation_model,
        base_url=str(raw.get("base_url", "")).strip(),
        api_key=str(raw.get("api_key", "")).strip(),
        api_key_env=str(raw.get("api_key_env", "OPENAI_API_KEY")).strip() or "OPENAI_API_KEY",
        temperature=_float_in_range(raw.get("temperature", 0.8), "llm.temperature", 0, 2),
        max_tokens=_positive_int(raw.get("max_tokens", 4096), "llm.max_tokens"),
        timeout_seconds=_float_in_range(
            raw.get("timeout_seconds", 30.0),
            "llm.timeout_seconds",
            1,
            300,
        ),
        routing=LLMRoutingConfig(
            enabled=_bool_value(routing_raw.get("enabled", False)),
            chat_preprocess_model=str(routing_raw.get("chat_preprocess_model", "")).strip(),
            chat_generation_model=chat_generation_model,
            qa_model=str(routing_raw.get("qa_model", "")).strip(),
            fact_extraction_model=str(routing_raw.get("fact_extraction_model", "")).strip(),
            cognition_model=str(routing_raw.get("cognition_model", "")).strip(),
            simple_vision_model=str(routing_raw.get("simple_vision_model", "")).strip(),
            detailed_vision_model=str(routing_raw.get("detailed_vision_model", "")).strip(),
            image_generation_model=str(routing_raw.get("image_generation_model", "")).strip(),
        ),
        providers=dict(providers or {}),
    )
