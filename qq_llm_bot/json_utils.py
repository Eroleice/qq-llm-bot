from __future__ import annotations

import json
import re
from typing import Any


_JSON_OBJECT_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.I | re.S)


def parse_json_object(text: str) -> dict[str, Any]:
    cleaned = str(text or "").strip()
    fenced = _JSON_OBJECT_FENCE_RE.search(cleaned)
    if fenced:
        cleaned = fenced.group(1)
    elif not cleaned.startswith("{"):
        sliced = _slice_json_object(cleaned)
        if sliced is not None:
            cleaned = sliced

    data = json.loads(cleaned)
    if not isinstance(data, dict):
        raise ValueError("JSON root is not an object")
    return data


def extract_json_object(text: str | None) -> dict[str, Any] | None:
    if not text:
        return None
    raw = str(text or "").strip()
    raw = _strip_json_fence(raw)
    sliced = _slice_json_object(raw)
    if sliced is not None:
        raw = sliced
    try:
        data = json.loads(raw)
    except ValueError:
        return None
    return data if isinstance(data, dict) else None


def _strip_json_fence(text: str) -> str:
    raw = text.strip()
    if not raw.startswith("```"):
        return raw
    raw = raw.strip("`").strip()
    if raw.lower().startswith("json"):
        raw = raw[4:].strip()
    return raw


def _slice_json_object(text: str) -> str | None:
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        return text[start : end + 1]
    return None
