from __future__ import annotations

import json
from typing import Any

from qq_llm_bot.onebot_render_utils import first_text


def segment_placeholder(segment_type: str, data: dict[str, Any]) -> str:
    if not segment_type:
        return ""
    if segment_type == "image":
        label = first_text(data, "summary", "file", "url", "file_url")
        return f"[图片: {label}]" if label else "[图片]"
    if segment_type in {"record", "voice"}:
        return "[语音]"
    if segment_type == "video":
        return "[视频]"
    if segment_type == "face":
        return "[表情]"
    if segment_type == "json":
        return _json_placeholder(data)
    if segment_type == "xml":
        return "[XML消息]"
    if segment_type == "reply":
        return ""
    return f"[{segment_type}消息]"


def _json_placeholder(data: dict[str, Any]) -> str:
    raw = str(data.get("data", "") or "").strip()
    if not raw:
        return "[JSON消息]"
    try:
        decoded = json.loads(raw)
    except json.JSONDecodeError:
        return "[JSON消息]"
    title = _find_nested_text(decoded, {"title", "desc", "summary", "prompt"})
    return f"[JSON消息: {title}]" if title else "[JSON消息]"


def _find_nested_text(value: Any, keys: set[str]) -> str:
    if isinstance(value, dict):
        for key, item in value.items():
            if str(key).lower() in keys:
                text = str(item or "").strip()
                if text:
                    return text
        for item in value.values():
            text = _find_nested_text(item, keys)
            if text:
                return text
    if isinstance(value, list):
        for item in value:
            text = _find_nested_text(item, keys)
            if text:
                return text
    return ""
