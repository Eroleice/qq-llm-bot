from __future__ import annotations

import json
import sqlite3
import time
from collections.abc import Iterable


def _clamp_score(value: int) -> int:
    return max(0, min(100, int(value)))

def _row_value(row: sqlite3.Row, key: str, default: str) -> str:
    try:
        value = row[key]
    except (IndexError, KeyError):
        return default
    return str(value or default)

def _row_int(row: sqlite3.Row, key: str, default: int) -> int:
    try:
        value = row[key]
    except (IndexError, KeyError):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default

def _safe_int(value: str, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default

def _local_usage_date(timestamp: int) -> str:
    return time.strftime("%Y-%m-%d", time.localtime(int(timestamp)))

def _row_float(row: sqlite3.Row, key: str, default: float) -> float:
    try:
        value = row[key]
    except (IndexError, KeyError):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default

def _compact_string_list(values: Iterable[str], limit: int = 10) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        item = " ".join(str(value).strip().split())[:80]
        if not item or item in seen:
            continue
        result.append(item)
        seen.add(item)
        if len(result) >= limit:
            break
    return result

def _sticker_text_key(value: str) -> str:
    return "".join(ch for ch in str(value).casefold() if ch.isalnum())

def _useful_sticker_text_key(value: str) -> bool:
    return len(value) >= 8 and len(set(value)) >= 3

def _decode_string_list(value: str, limit: int = 10) -> list[str]:
    try:
        decoded = json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return []
    if not isinstance(decoded, list):
        return []
    return _compact_string_list((str(item) for item in decoded), limit=limit)

def _compact_int_list(values: Iterable[int], limit: int = 20) -> list[int]:
    result: list[int] = []
    seen: set[int] = set()
    for value in values:
        try:
            item = int(value)
        except (TypeError, ValueError):
            continue
        if item in seen:
            continue
        result.append(item)
        seen.add(item)
        if len(result) >= limit:
            break
    return result

def _decode_int_list(value: str) -> list[int]:
    try:
        decoded = json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return []
    if not isinstance(decoded, list):
        return []
    return _compact_int_list(decoded, limit=80)

def _decode_traits(value: str) -> dict[str, object]:
    try:
        decoded = json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return {}
    return decoded if isinstance(decoded, dict) else {}
