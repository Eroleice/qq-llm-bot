from __future__ import annotations

import time
from typing import Any

from qq_llm_bot.config import ParticipationMode


def is_group_enabled(storage: Any, group_id: str) -> bool:
    with storage._connect() as conn:
        row = conn.execute(
            "SELECT enabled FROM group_whitelist WHERE group_id = ?",
            (str(group_id),),
        ).fetchone()
    if row is None:
        return False
    return bool(row["enabled"])


def set_group_enabled(storage: Any, group_id: str, enabled: bool) -> None:
    now = int(time.time())
    with storage._connect() as conn:
        conn.execute(
            """
            INSERT INTO group_whitelist (group_id, enabled, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(group_id) DO UPDATE SET
                enabled = excluded.enabled,
                updated_at = excluded.updated_at
            """,
            (str(group_id), 1 if enabled else 0, now),
        )


def list_enabled_groups(storage: Any) -> list[str]:
    with storage._connect() as conn:
        rows = conn.execute(
            "SELECT group_id FROM group_whitelist WHERE enabled = 1 ORDER BY group_id"
        ).fetchall()
    return [str(row["group_id"]) for row in rows]


def get_group_mode(
    storage: Any,
    group_id: str,
    default_mode: ParticipationMode,
) -> ParticipationMode:
    with storage._connect() as conn:
        row = conn.execute(
            "SELECT mode FROM group_modes WHERE group_id = ?",
            (str(group_id),),
        ).fetchone()
    if row is None:
        return default_mode
    mode = str(row["mode"])
    if mode not in {"silent", "passive", "active"}:
        return default_mode
    return mode  # type: ignore[return-value]


def set_group_mode(storage: Any, group_id: str, mode: ParticipationMode) -> None:
    now = int(time.time())
    with storage._connect() as conn:
        conn.execute(
            """
            INSERT INTO group_modes (group_id, mode, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(group_id) DO UPDATE SET
                mode = excluded.mode,
                updated_at = excluded.updated_at
            """,
            (str(group_id), mode, now),
        )
