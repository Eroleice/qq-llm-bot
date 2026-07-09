from __future__ import annotations

from typing import Any

from qq_llm_bot.models import MemoryRecord
from qq_llm_bot.storage_records import (
    _dashboard_user_id_variants,
    _memory_record,
    format_memory_record,
)


def list_memories(
    storage: Any,
    owner_type: str,
    owner_id: str,
    limit: int = 8,
    status: str = "active",
) -> list[MemoryRecord]:
    owner_variants = _memory_owner_variants(owner_type, owner_id)
    if not owner_variants:
        return []
    placeholders = ", ".join("?" for _ in owner_variants)
    with storage._connect() as conn:
        rows = conn.execute(
            f"""
            SELECT id, owner_type, owner_id, kind, content, confidence, importance, status,
                   updated_at, source_user_id, source_group_id, subject_user_id,
                   claim_scope, verification_status
            FROM memory_items
            WHERE owner_type = ? AND owner_id IN ({placeholders}) AND status = ?
            ORDER BY importance DESC, updated_at DESC, id DESC
            LIMIT ?
            """,
            (str(owner_type), *owner_variants, str(status), limit),
        ).fetchall()
    return [_memory_record(row) for row in rows]


def list_user_memories(storage: Any, user_id: str, limit: int = 8) -> list[str]:
    return [format_memory_record(record) for record in list_memories(storage, "user", user_id, limit)]


def list_self_memories(storage: Any, status: str = "active", limit: int = 16) -> list[str]:
    return [
        format_memory_record(record)
        for record in list_memories(storage, "self", "bot", limit=limit, status=status)
    ]


def list_memories_by_status(storage: Any, status: str, limit: int = 12) -> list[str]:
    with storage._connect() as conn:
        rows = conn.execute(
            """
            SELECT id, owner_type, owner_id, kind, content, confidence, importance, status,
                   updated_at, source_user_id, source_group_id, subject_user_id,
                   claim_scope, verification_status
            FROM memory_items
            WHERE status = ?
            ORDER BY updated_at DESC, id DESC
            LIMIT ?
            """,
            (str(status), limit),
        ).fetchall()
    return [format_memory_record(_memory_record(row)) for row in rows]


def get_memory_record(storage: Any, memory_id: int) -> MemoryRecord | None:
    with storage._connect() as conn:
        row = conn.execute(
            """
            SELECT id, owner_type, owner_id, kind, content, confidence, importance, status,
                   updated_at, source_user_id, source_group_id, subject_user_id,
                   claim_scope, verification_status
            FROM memory_items
            WHERE id = ?
            """,
            (int(memory_id),),
        ).fetchone()
    return _memory_record(row) if row else None


def _memory_owner_variants(owner_type: str, owner_id: str) -> list[str]:
    if str(owner_type) == "user":
        return _dashboard_user_id_variants(owner_id)
    owner = str(owner_id)
    return [owner] if owner else []
