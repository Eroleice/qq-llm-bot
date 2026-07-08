from __future__ import annotations

import sqlite3

from qq_llm_bot.models import MemoryCandidate


def insert_memory(conn: sqlite3.Connection, item: MemoryCandidate, now: int) -> None:
    conn.execute(
        """
        INSERT INTO memory_items (
            owner_type, owner_id, kind, content, confidence, importance, status,
            evidence_message_id, source_text, source_user_id, source_group_id,
            subject_user_id, claim_scope, verification_status,
            conflict_of, created_at, updated_at, last_seen_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            item.owner_type,
            item.owner_id,
            item.kind,
            item.content,
            item.confidence,
            item.importance,
            item.status,
            item.evidence_message_id,
            item.source_text,
            item.source_user_id,
            item.source_group_id,
            item.subject_user_id,
            item.claim_scope,
            item.verification_status,
            item.conflict_of,
            now,
            now,
            now,
        ),
    )
