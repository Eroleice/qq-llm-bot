from __future__ import annotations

import sqlite3

from qq_llm_bot.models import FactRecord
from qq_llm_bot.storage_records import _fact_record


def fetch_fact_record(conn: sqlite3.Connection, fact_id: int) -> FactRecord | None:
    row = conn.execute(
        """
        SELECT id, subject_user_id, fact_type, claim_text, topic, stance,
               confidence, status, claim_scope, source_user_id, source_group_id,
               evidence_message_id, evidence_text, created_at, updated_at,
               importance, last_seen_at, superseded_by_fact_id, forget_reason
        FROM member_facts
        WHERE id = ?
        """,
        (int(fact_id),),
    ).fetchone()
    return _fact_record(row) if row else None
