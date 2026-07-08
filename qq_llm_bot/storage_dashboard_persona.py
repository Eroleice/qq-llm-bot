from __future__ import annotations

from typing import Any

from qq_llm_bot.storage_records import _memory_to_dict


def get_dashboard_persona(storage: Any) -> dict[str, object]:
    with storage._connect() as conn:
        rows = conn.execute(
            "SELECT key, value, updated_at FROM persona_state ORDER BY key"
        ).fetchall()
    return {
        "persona_state": [
            {
                "key": str(row["key"]),
                "value": str(row["value"]),
                "updated_at": int(row["updated_at"]),
            }
            for row in rows
        ],
        "self_memories": [
            _memory_to_dict(record)
            for record in storage.list_memories("self", "bot", limit=100, status="active")
        ],
        "pending_self_memories": [
            _memory_to_dict(record)
            for record in storage.list_memories(
                "self",
                "bot",
                limit=100,
                status="pending_confirmation",
            )
        ],
        "conflict_self_memories": [
            _memory_to_dict(record)
            for record in storage.list_memories("self", "bot", limit=100, status="conflict")
        ],
    }
