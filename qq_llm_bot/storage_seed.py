from __future__ import annotations

import sqlite3
import time
from collections.abc import Iterable, Mapping


def seed_storage_config(
    conn: sqlite3.Connection,
    *,
    admins: Iterable[str],
    enabled_groups: Iterable[str],
    ignored_users: Iterable[str],
    persona: Mapping[str, str],
) -> None:
    now = int(time.time())
    conn.executemany(
        """
        INSERT INTO admins (user_id, added_at)
        VALUES (?, ?)
        ON CONFLICT(user_id) DO NOTHING
        """,
        [(user_id, now) for user_id in admins],
    )
    conn.executemany(
        """
        INSERT INTO group_whitelist (group_id, enabled, updated_at)
        VALUES (?, 1, ?)
        ON CONFLICT(group_id) DO UPDATE SET enabled = 1, updated_at = excluded.updated_at
        """,
        [(group_id, now) for group_id in enabled_groups],
    )
    conn.executemany(
        """
        INSERT INTO ignored_users (user_id, ignored, updated_at)
        VALUES (?, 1, ?)
        ON CONFLICT(user_id) DO UPDATE SET ignored = 1, updated_at = excluded.updated_at
        """,
        [(user_id, now) for user_id in ignored_users],
    )
    conn.executemany(
        """
        INSERT INTO persona_state (key, value, updated_at)
        VALUES (?, ?, ?)
        ON CONFLICT(key) DO NOTHING
        """,
        [(key, value, now) for key, value in persona.items()],
    )
