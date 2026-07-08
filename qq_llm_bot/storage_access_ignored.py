from __future__ import annotations

import time
from typing import Any

from qq_llm_bot.storage_records import _dashboard_user_id


def is_user_ignored(storage: Any, user_id: str) -> bool:
    user_id = _dashboard_user_id(user_id)
    if not user_id:
        return False
    with storage._connect() as conn:
        row = conn.execute(
            "SELECT ignored FROM ignored_users WHERE user_id = ?",
            (user_id,),
        ).fetchone()
    return bool(row["ignored"]) if row is not None else False


def add_ignored_user(storage: Any, user_id: str) -> None:
    user_id = _dashboard_user_id(user_id)
    if not user_id:
        return
    now = int(time.time())
    with storage._connect() as conn:
        conn.execute(
            """
            INSERT INTO ignored_users (user_id, ignored, updated_at)
            VALUES (?, 1, ?)
            ON CONFLICT(user_id) DO UPDATE SET
                ignored = 1,
                updated_at = excluded.updated_at
            """,
            (user_id, now),
        )


def remove_ignored_user(storage: Any, user_id: str) -> None:
    user_id = _dashboard_user_id(user_id)
    if not user_id:
        return
    now = int(time.time())
    with storage._connect() as conn:
        conn.execute(
            """
            INSERT INTO ignored_users (user_id, ignored, updated_at)
            VALUES (?, 0, ?)
            ON CONFLICT(user_id) DO UPDATE SET
                ignored = 0,
                updated_at = excluded.updated_at
            """,
            (user_id, now),
        )


def list_ignored_users(storage: Any) -> list[str]:
    with storage._connect() as conn:
        rows = conn.execute(
            "SELECT user_id FROM ignored_users WHERE ignored = 1 ORDER BY user_id"
        ).fetchall()
    return [str(row["user_id"]) for row in rows]
