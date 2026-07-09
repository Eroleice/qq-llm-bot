from __future__ import annotations

import sqlite3
from typing import Any

from qq_llm_bot.models import MemoryCandidate
from qq_llm_bot.storage_fact_constants import TRUSTED_THIRD_PARTY_THRESHOLD
from qq_llm_bot.storage_records import _dashboard_user_id
from qq_llm_bot.storage_relationship_keys import GLOBAL_RELATIONSHIP_GROUP_ID


def acceptance_status(
    storage: Any,
    conn: sqlite3.Connection,
    item: MemoryCandidate,
    confidence_threshold: float,
) -> str:
    if not item.content or item.confidence < confidence_threshold:
        return "rejected"

    if item.owner_type == "self" or item.subject_user_id == "bot" or item.claim_scope == "bot_directed":
        if item.source_user_id == "bot" or is_admin_conn(storage, conn, item.source_user_id):
            return "accepted"
        return "pending_confirmation"

    if item.claim_scope == "third_party":
        source = source_trust(conn, item.source_group_id, item.source_user_id)
        return "accepted" if source >= TRUSTED_THIRD_PARTY_THRESHOLD else "pending_confirmation"

    if item.claim_scope == "group_fact":
        if item.source_user_id == "bot":
            return "accepted" if item.confidence >= confidence_threshold else "rejected"
        source = source_trust(conn, item.source_group_id, item.source_user_id)
        if source >= TRUSTED_THIRD_PARTY_THRESHOLD and item.confidence >= 0.85:
            return "accepted"
        return "pending_confirmation"

    return "accepted"


def source_trust(conn: sqlite3.Connection, group_id: str, user_id: str) -> int:
    subject = _dashboard_user_id(user_id)
    if not subject:
        return 0
    row = conn.execute(
        """
        SELECT trust
        FROM relationships
        WHERE group_id = ? AND user_id = ?
        """,
        (GLOBAL_RELATIONSHIP_GROUP_ID, subject),
    ).fetchone()
    return int(row["trust"]) if row else 0


def is_admin_conn(storage: Any, conn: sqlite3.Connection, user_id: str) -> bool:
    if str(user_id) in storage.initial_admins:
        return True
    row = conn.execute("SELECT 1 FROM admins WHERE user_id = ?", (str(user_id),)).fetchone()
    return row is not None
