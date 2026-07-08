from __future__ import annotations

from typing import Any

from qq_llm_bot.config_models import FactConfig, LexiconConfig, ObservationBatchConfig, ReflectionConfig
from qq_llm_bot.config_values import (
    bool_value as _bool_value,
    float_in_range as _float_in_range,
    int_in_range as _int_in_range,
    positive_int as _positive_int,
)


def reflection_config(raw: dict[str, Any]) -> ReflectionConfig:
    return ReflectionConfig(
        enabled=_bool_value(raw.get("enabled", True)),
        message_threshold=_positive_int(
            raw.get("message_threshold", 30),
            "reflection.message_threshold",
        ),
        recent_limit=_positive_int(raw.get("recent_limit", 40), "reflection.recent_limit"),
        min_interval_seconds=_positive_int(
            raw.get("min_interval_seconds", 600),
            "reflection.min_interval_seconds",
        ),
    )


def observation_batch_config(raw: dict[str, Any]) -> ObservationBatchConfig:
    return ObservationBatchConfig(
        enabled=_bool_value(raw.get("enabled", True)),
        batch_size=_positive_int(
            raw.get("batch_size", 30),
            "observation_batch.batch_size",
        ),
        max_interval_seconds=_positive_int(
            raw.get("max_interval_seconds", 600),
            "observation_batch.max_interval_seconds",
        ),
        max_messages_per_batch=_positive_int(
            raw.get("max_messages_per_batch", 40),
            "observation_batch.max_messages_per_batch",
        ),
        max_message_chars=_positive_int(
            raw.get("max_message_chars", 300),
            "observation_batch.max_message_chars",
        ),
    )


def fact_config(raw: dict[str, Any]) -> FactConfig:
    return FactConfig(
        fact_confidence_threshold=_float_in_range(
            raw.get("fact_confidence_threshold", 0.75),
            "facts.fact_confidence_threshold",
            0,
            1,
        ),
        third_party_trust_threshold=_int_in_range(
            raw.get("third_party_trust_threshold", 70),
            "facts.third_party_trust_threshold",
            0,
            100,
        ),
        third_party_confidence_threshold=_float_in_range(
            raw.get("third_party_confidence_threshold", 0.85),
            "facts.third_party_confidence_threshold",
            0,
            1,
        ),
        profile_fact_threshold=_positive_int(
            raw.get("profile_fact_threshold", 5),
            "facts.profile_fact_threshold",
        ),
        context_fact_limit=_positive_int(
            raw.get("context_fact_limit", 8),
            "facts.context_fact_limit",
        ),
        target_user_limit=_positive_int(
            raw.get("target_user_limit", 5),
            "facts.target_user_limit",
        ),
        low_importance_threshold=_float_in_range(
            raw.get("low_importance_threshold", 0.35),
            "facts.low_importance_threshold",
            0,
            1,
        ),
        fact_context_ttl_days=_positive_int(
            raw.get("fact_context_ttl_days", 30),
            "facts.fact_context_ttl_days",
        ),
    )


def lexicon_config(raw: dict[str, Any]) -> LexiconConfig:
    return LexiconConfig(
        enabled=_bool_value(raw.get("enabled", False)),
        provider=str(raw.get("provider", "disabled")).strip() or "disabled",
        base_url=str(raw.get("base_url", "")).strip(),
        api_key=str(raw.get("api_key", "")).strip(),
        api_key_env=str(raw.get("api_key_env", "WEB_SEARCH_API_KEY")).strip()
        or "WEB_SEARCH_API_KEY",
        min_interval_seconds=_positive_int(
            raw.get("min_interval_seconds", 300),
            "lexicon.min_interval_seconds",
        ),
        max_terms_per_message=_positive_int(
            raw.get("max_terms_per_message", 1),
            "lexicon.max_terms_per_message",
        ),
        max_results=_positive_int(raw.get("max_results", 5), "lexicon.max_results"),
        confidence_threshold=_float_in_range(
            raw.get("confidence_threshold", 0.78),
            "lexicon.confidence_threshold",
            0,
            1,
        ),
        timeout_seconds=_float_in_range(
            raw.get("timeout_seconds", 10.0),
            "lexicon.timeout_seconds",
            1,
            60,
        ),
    )
