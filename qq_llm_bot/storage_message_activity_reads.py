from __future__ import annotations

import time


def get_recent_activity_counts(
    storage: object,
    group_id: str,
    human_window_seconds: int = 60,
    bot_window_seconds: int = 120,
) -> tuple[int, int]:
    now = int(time.time())
    with storage._connect() as conn:
        human_count = int(
            conn.execute(
                """
                SELECT COUNT(1) AS count
                FROM messages
                WHERE group_id = ?
                  AND time >= ?
                  AND sender_role != 'bot'
                """,
                (str(group_id), now - int(human_window_seconds)),
            ).fetchone()["count"]
        )
        bot_count = int(
            conn.execute(
                """
                SELECT COUNT(1) AS count
                FROM messages
                WHERE group_id = ?
                  AND time >= ?
                  AND sender_role = 'bot'
                """,
                (str(group_id), now - int(bot_window_seconds)),
            ).fetchone()["count"]
        )
    return human_count, bot_count
