from __future__ import annotations

import sqlite3

from qq_llm_bot.models import MessageContext
from qq_llm_bot.storage_record_identity import _dashboard_user_id


def upsert_user_profile(
    conn: sqlite3.Connection,
    context: MessageContext,
    now: int,
) -> None:
    user_id = _dashboard_user_id(context.user_id)
    if not user_id:
        return
    nickname = " ".join(str(context.sender_nickname or "").split())
    display_name = " ".join(str(context.sender_name or "").split())
    conn.execute(
        """
        INSERT INTO user_profiles (user_id, nickname, display_name, updated_at, last_seen_at)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(user_id) DO UPDATE SET
            nickname = CASE
                WHEN excluded.nickname != '' THEN excluded.nickname
                ELSE user_profiles.nickname
            END,
            display_name = CASE
                WHEN excluded.display_name != '' THEN excluded.display_name
                ELSE user_profiles.display_name
            END,
            updated_at = excluded.updated_at,
            last_seen_at = excluded.last_seen_at
        """,
        (user_id, nickname, display_name, now, now),
    )
