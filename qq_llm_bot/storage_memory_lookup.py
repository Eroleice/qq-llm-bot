from __future__ import annotations

import sqlite3

from qq_llm_bot.models import MemoryCandidate, MemoryRecord
from qq_llm_bot.storage_fact_constants import CONFLICT_SENSITIVE_KINDS
from qq_llm_bot.storage_helpers import (
    looks_like_self_direct_conflict as _looks_like_self_direct_conflict,
)
from qq_llm_bot.storage_records import _memory_record


def find_duplicate_memory(
    conn: sqlite3.Connection,
    item: MemoryCandidate,
) -> MemoryRecord | None:
    if item.kind == "lexicon" and item.subject_user_id:
        row = conn.execute(
            """
            SELECT id, owner_type, owner_id, kind, content, confidence, importance, status,
                   updated_at, source_user_id, source_group_id, subject_user_id,
                   claim_scope, verification_status
            FROM memory_items
            WHERE owner_type = ?
              AND owner_id = ?
              AND kind = 'lexicon'
              AND subject_user_id = ?
              AND status = 'active'
            LIMIT 1
            """,
            (item.owner_type, item.owner_id, item.subject_user_id),
        ).fetchone()
        if row:
            return _memory_record(row)

    row = conn.execute(
        """
        SELECT id, owner_type, owner_id, kind, content, confidence, importance, status,
               updated_at, source_user_id, source_group_id, subject_user_id,
               claim_scope, verification_status
        FROM memory_items
        WHERE owner_type = ?
          AND owner_id = ?
          AND kind = ?
          AND content = ?
          AND status = 'active'
        LIMIT 1
        """,
        (item.owner_type, item.owner_id, item.kind, item.content),
    ).fetchone()
    return _memory_record(row) if row else None


def find_conflicting_memory(
    conn: sqlite3.Connection,
    item: MemoryCandidate,
) -> MemoryRecord | None:
    if item.owner_type == "self" and item.kind in {"self_preference", "self_boundary"}:
        return find_conflicting_self_memory(conn, item)

    if item.kind not in CONFLICT_SENSITIVE_KINDS:
        return None
    row = conn.execute(
        """
        SELECT id, owner_type, owner_id, kind, content, confidence, importance, status,
               updated_at, source_user_id, source_group_id, subject_user_id,
               claim_scope, verification_status
        FROM memory_items
        WHERE owner_type = ?
          AND owner_id = ?
          AND kind = ?
          AND content != ?
          AND status = 'active'
        ORDER BY confidence DESC, updated_at DESC
        LIMIT 1
        """,
        (item.owner_type, item.owner_id, item.kind, item.content),
    ).fetchone()
    return _memory_record(row) if row else None


def find_conflicting_self_memory(
    conn: sqlite3.Connection,
    item: MemoryCandidate,
) -> MemoryRecord | None:
    rows = conn.execute(
        """
        SELECT id, owner_type, owner_id, kind, content, confidence, importance, status,
               updated_at, source_user_id, source_group_id, subject_user_id,
               claim_scope, verification_status
        FROM memory_items
        WHERE owner_type = 'self'
          AND owner_id = ?
          AND kind = ?
          AND content != ?
          AND status = 'active'
        ORDER BY confidence DESC, updated_at DESC
        """,
        (item.owner_id, item.kind, item.content),
    ).fetchall()
    for row in rows:
        record = _memory_record(row)
        if _looks_like_self_direct_conflict(item.content, record.content):
            return record
    return None
