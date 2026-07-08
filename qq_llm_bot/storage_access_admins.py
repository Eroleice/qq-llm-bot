from __future__ import annotations

import time
from typing import Any


def is_admin(storage: Any, user_id: str) -> bool:
    with storage._connect() as conn:
        row = conn.execute(
            "SELECT 1 FROM admins WHERE user_id = ?",
            (str(user_id),),
        ).fetchone()
    return row is not None


def add_admin(storage: Any, user_id: str) -> None:
    with storage._connect() as conn:
        conn.execute(
            """
            INSERT INTO admins (user_id, added_at)
            VALUES (?, ?)
            ON CONFLICT(user_id) DO NOTHING
            """,
            (str(user_id), int(time.time())),
        )


def remove_admin(storage: Any, user_id: str) -> None:
    if str(user_id) in storage.initial_admins:
        return
    with storage._connect() as conn:
        conn.execute("DELETE FROM admins WHERE user_id = ?", (str(user_id),))


def list_admins(storage: Any) -> list[str]:
    with storage._connect() as conn:
        rows = conn.execute("SELECT user_id FROM admins ORDER BY user_id").fetchall()
    return [str(row["user_id"]) for row in rows]
