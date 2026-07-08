from __future__ import annotations

import sqlite3


def _llm_usage_summary_to_dict(row: sqlite3.Row | None, since: int) -> dict[str, object]:
    if row is None:
        return {
            "since": since,
            "calls": 0,
            "prompt_chars": 0,
            "completion_chars": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "first_at": 0,
            "last_at": 0,
        }
    return {
        "since": since,
        "calls": int(row["calls"] or 0),
        "prompt_chars": int(row["prompt_chars"] or 0),
        "completion_chars": int(row["completion_chars"] or 0),
        "prompt_tokens": int(row["prompt_tokens"] or 0),
        "completion_tokens": int(row["completion_tokens"] or 0),
        "total_tokens": int(row["total_tokens"] or 0),
        "first_at": int(row["first_at"] or 0),
        "last_at": int(row["last_at"] or 0),
    }


def _llm_usage_group_to_dict(row: sqlite3.Row) -> dict[str, object]:
    return {
        "purpose": str(row["purpose"] or ""),
        "model": str(row["model"] or ""),
        "calls": int(row["calls"] or 0),
        "prompt_chars": int(row["prompt_chars"] or 0),
        "completion_chars": int(row["completion_chars"] or 0),
        "prompt_tokens": int(row["prompt_tokens"] or 0),
        "completion_tokens": int(row["completion_tokens"] or 0),
        "total_tokens": int(row["total_tokens"] or 0),
        "last_at": int(row["last_at"] or 0),
    }


def _llm_usage_row_to_dict(row: sqlite3.Row) -> dict[str, object]:
    return {
        "id": int(row["id"]),
        "created_at": int(row["created_at"]),
        "purpose": str(row["purpose"] or ""),
        "model": str(row["model"] or ""),
        "prompt_chars": int(row["prompt_chars"] or 0),
        "completion_chars": int(row["completion_chars"] or 0),
        "prompt_tokens": int(row["prompt_tokens"] or 0),
        "completion_tokens": int(row["completion_tokens"] or 0),
        "total_tokens": int(row["total_tokens"] or 0),
    }
