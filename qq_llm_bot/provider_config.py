from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from qq_llm_bot.config_models import LLMProviderConfig


def load_provider_config(project_root: Path) -> dict[str, LLMProviderConfig]:
    provider_path = project_root / "provider.json"
    if not provider_path.exists():
        return {}

    with provider_path.open("r", encoding="utf-8") as fp:
        raw = json.load(fp)

    providers: dict[str, LLMProviderConfig] = {}
    for entry in _provider_entries(raw):
        provider = _provider_from_entry(entry)
        if provider.id in providers:
            raise ValueError(f"provider.json has duplicate provider id: {provider.id}")
        providers[provider.id] = provider
    return providers


def _provider_entries(raw: Any) -> list[dict[str, Any]]:
    if isinstance(raw, list):
        return [_dict_entry(item) for item in raw]
    if not isinstance(raw, dict):
        raise ValueError("provider.json must be an object or an array")

    if "providers" in raw:
        providers = raw["providers"]
        if not isinstance(providers, list):
            raise ValueError("provider.json providers must be an array")
        return [_dict_entry(item) for item in providers]

    if "id" in raw:
        return [_dict_entry(raw)]

    entries = []
    for provider_id, value in raw.items():
        entry = _dict_entry(value)
        entry.setdefault("id", provider_id)
        entries.append(entry)
    return entries


def _dict_entry(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("provider.json provider entries must be objects")
    return dict(value)


def _provider_from_entry(entry: dict[str, Any]) -> LLMProviderConfig:
    provider_id = str(entry.get("id", "")).strip()
    if not provider_id:
        raise ValueError("provider.json provider id is required")
    if ":" in provider_id:
        raise ValueError(f"provider.json provider id must not contain ':': {provider_id}")

    key_string = str(entry.get("key_string", entry.get("api_key", ""))).strip()
    key_env = str(entry.get("key_env", entry.get("api_key_env", ""))).strip()
    if bool(key_string) == bool(key_env):
        raise ValueError(
            f"provider.json provider {provider_id} must set exactly one of key_string/key_env"
        )

    models = _model_list(entry.get("model", entry.get("models", [])), provider_id)
    if not models:
        raise ValueError(f"provider.json provider {provider_id} must list at least one model")

    return LLMProviderConfig(
        id=provider_id,
        url=str(entry.get("url", entry.get("base_url", ""))).strip(),
        type=str(entry.get("type", "openai")).strip().lower() or "openai",
        key_string=key_string,
        key_env=key_env,
        models=models,
    )


def _model_list(value: Any, provider_id: str) -> list[str]:
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if not isinstance(value, list):
        raise ValueError(f"provider.json provider {provider_id} model must be an array")
    models = [str(item).strip() for item in value if str(item).strip()]
    return list(dict.fromkeys(models))
