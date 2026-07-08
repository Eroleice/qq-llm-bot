from __future__ import annotations

from typing import Any

from qq_llm_bot.storage_records import (
    _llm_usage_group_to_dict,
    _llm_usage_row_to_dict,
    _llm_usage_summary_to_dict,
)


def list_dashboard_llm_usage(
    storage: Any,
    since: int,
    limit: int = 100,
) -> dict[str, object]:
    since = max(0, int(since))
    limit = max(1, int(limit))
    with storage._connect() as conn:
        summary = conn.execute(
            """
            SELECT
                COUNT(1) AS calls,
                COALESCE(SUM(prompt_chars), 0) AS prompt_chars,
                COALESCE(SUM(completion_chars), 0) AS completion_chars,
                COALESCE(SUM(prompt_tokens), 0) AS prompt_tokens,
                COALESCE(SUM(completion_tokens), 0) AS completion_tokens,
                COALESCE(SUM(total_tokens), 0) AS total_tokens,
                MIN(created_at) AS first_at,
                MAX(created_at) AS last_at
            FROM llm_usage
            WHERE created_at >= ?
            """,
            (since,),
        ).fetchone()
        by_purpose_rows = conn.execute(
            """
            SELECT
                purpose,
                model,
                COUNT(1) AS calls,
                COALESCE(SUM(prompt_chars), 0) AS prompt_chars,
                COALESCE(SUM(completion_chars), 0) AS completion_chars,
                COALESCE(SUM(prompt_tokens), 0) AS prompt_tokens,
                COALESCE(SUM(completion_tokens), 0) AS completion_tokens,
                COALESCE(SUM(total_tokens), 0) AS total_tokens,
                MAX(created_at) AS last_at
            FROM llm_usage
            WHERE created_at >= ?
            GROUP BY purpose, model
            ORDER BY total_tokens DESC, calls DESC, purpose ASC
            """,
            (since,),
        ).fetchall()
        recent_rows = conn.execute(
            """
            SELECT id, created_at, purpose, model, prompt_chars, completion_chars,
                   prompt_tokens, completion_tokens, total_tokens
            FROM llm_usage
            WHERE created_at >= ?
            ORDER BY created_at DESC, id DESC
            LIMIT ?
            """,
            (since, limit),
        ).fetchall()
    return {
        "summary": _llm_usage_summary_to_dict(summary, since),
        "by_purpose": [_llm_usage_group_to_dict(row) for row in by_purpose_rows],
        "recent": [_llm_usage_row_to_dict(row) for row in recent_rows],
    }
