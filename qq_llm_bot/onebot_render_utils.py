from __future__ import annotations

from typing import Any


def append_block(parts: list[str], block: str) -> None:
    if not block:
        return
    if parts and parts[-1] and not parts[-1].endswith(("\n", " ")):
        parts.append("\n")
    parts.append(block)
    if not block.endswith("\n"):
        parts.append("\n")


def first_existing(data: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in data:
            return data[key]
    return None


def first_text(data: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = str(data.get(key, "") or "").strip()
        if value:
            return value
    return ""


def truncate_text(text: str, limit: int) -> str:
    if limit <= 0 or len(text) <= limit:
        return text
    if limit <= 3:
        return text[:limit]
    return text[: limit - 3].rstrip() + "..."
