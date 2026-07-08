from __future__ import annotations

import time
from typing import Any


def get_persona_lines(storage: Any) -> list[str]:
    with storage._connect() as conn:
        rows = conn.execute("SELECT key, value FROM persona_state ORDER BY key").fetchall()
    lines = [f"{row['key']}: {row['value']}" for row in rows]
    self_memories = storage.list_memories("self", "bot", limit=8)
    lines.extend(f"self_memory#{record.id}: {record.content}" for record in self_memories)
    return lines


def format_persona(storage: Any) -> str:
    return "\n".join(storage.get_persona_lines()) or "(empty)"


def should_reflect(
    storage: Any,
    group_id: str,
    threshold: int,
    min_interval_seconds: int,
) -> bool:
    now = int(time.time())
    with storage._connect() as conn:
        latest = conn.execute(
            """
            SELECT created_at
            FROM memory_items
            WHERE owner_type = 'group'
              AND owner_id = ?
              AND kind = 'reflection'
              AND status = 'active'
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (str(group_id),),
        ).fetchone()
        latest_time = int(latest["created_at"]) if latest else 0
        if latest_time and now - latest_time < min_interval_seconds:
            return False
        count = conn.execute(
            "SELECT COUNT(1) AS count FROM messages WHERE group_id = ? AND time > ?",
            (str(group_id), latest_time),
        ).fetchone()
    return int(count["count"]) >= threshold
