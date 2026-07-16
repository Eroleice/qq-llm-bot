from __future__ import annotations

import time
from typing import Any

from qq_llm_bot.knowledge_models import GuessWhoScoreRecord
from qq_llm_bot.storage_record_identity import _dashboard_user_id


def record_guesswho_result(
    storage: Any,
    group_id: str,
    user_id: str,
    *,
    correct: bool,
    updated_at: int | None = None,
) -> None:
    canonical_user_id = _dashboard_user_id(user_id)
    if not canonical_user_id:
        return
    correct_delta = 1 if correct else 0
    wrong_delta = 0 if correct else 1
    timestamp = int(time.time()) if updated_at is None else int(updated_at)
    with storage._connect() as conn:
        conn.execute(
            """
            INSERT INTO guesswho_scores (
                group_id, user_id, correct_count, wrong_count, updated_at
            )
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(group_id, user_id) DO UPDATE SET
                correct_count = guesswho_scores.correct_count + excluded.correct_count,
                wrong_count = guesswho_scores.wrong_count + excluded.wrong_count,
                updated_at = excluded.updated_at
            """,
            (str(group_id), canonical_user_id, correct_delta, wrong_delta, timestamp),
        )


def list_guesswho_scores(
    storage: Any,
    group_id: str,
    *,
    wrong: bool = False,
    limit: int = 10,
) -> list[GuessWhoScoreRecord]:
    ranking_column = "wrong_count" if wrong else "correct_count"
    limit = max(1, min(100, int(limit)))
    with storage._connect() as conn:
        rows = conn.execute(
            f"""
            SELECT scores.group_id, scores.user_id,
                   scores.correct_count, scores.wrong_count, scores.updated_at,
                   COALESCE(NULLIF(profiles.display_name, ''), profiles.nickname, '') AS nickname
            FROM guesswho_scores AS scores
            LEFT JOIN user_profiles AS profiles ON profiles.user_id = scores.user_id
            WHERE scores.group_id = ? AND scores.{ranking_column} > 0
            ORDER BY scores.{ranking_column} DESC, scores.user_id ASC
            LIMIT ?
            """,
            (str(group_id), limit),
        ).fetchall()
    return [
        GuessWhoScoreRecord(
            group_id=str(row["group_id"]),
            user_id=str(row["user_id"]),
            correct_count=int(row["correct_count"]),
            wrong_count=int(row["wrong_count"]),
            nickname=str(row["nickname"] or ""),
            updated_at=int(row["updated_at"]),
        )
        for row in rows
    ]
