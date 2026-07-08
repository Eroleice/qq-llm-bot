from __future__ import annotations

from typing import Any


def section(raw: dict[str, Any], key: str) -> dict[str, Any]:
    value = raw.get(key, {})
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be a TOML table")
    return value


def string_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if not isinstance(value, list):
        raise ValueError("Expected a string list")
    return [str(item).strip() for item in value if str(item).strip()]


def positive_int(value: Any, name: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{name} must be greater than 0")
    return parsed


def float_in_range(value: Any, name: str, minimum: float, maximum: float) -> float:
    parsed = float(value)
    if parsed < minimum or parsed > maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return parsed


def int_in_range(value: Any, name: str, minimum: int, maximum: int) -> int:
    parsed = int(value)
    if parsed < minimum or parsed > maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return parsed


def bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def route_prefix(value: Any) -> str:
    prefix = str(value).strip() or "/"
    if not prefix.startswith("/"):
        prefix = "/" + prefix
    return prefix.rstrip("/") or "/"


def image_generation_size(value: Any, max_dimension: int) -> str:
    fallback = "512x512"
    raw = str(value).strip().lower()
    if "x" not in raw:
        return fallback
    width_raw, height_raw = raw.split("x", 1)
    try:
        width = int(width_raw)
        height = int(height_raw)
    except ValueError:
        return fallback
    if width <= 0 or height <= 0:
        return fallback
    width = min(width, max_dimension)
    height = min(height, max_dimension)
    width = max(16, (width // 16) * 16)
    height = max(16, (height // 16) * 16)
    return f"{width}x{height}"
