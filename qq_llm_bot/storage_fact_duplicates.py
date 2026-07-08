from __future__ import annotations

import sqlite3

from qq_llm_bot.models import FactCandidate, FactRecord
from qq_llm_bot.storage_records import _fact_record


def find_duplicate_fact(
    conn: sqlite3.Connection,
    item: FactCandidate,
    acceptance_status: str,
) -> FactRecord | None:
    target_status = "accepted" if acceptance_status == "accepted" else "pending_confirmation"
    row = conn.execute(
        """
        SELECT id, subject_user_id, fact_type, claim_text, topic, stance,
               confidence, status, claim_scope, source_user_id, source_group_id,
               evidence_message_id, evidence_text, created_at, updated_at
        FROM member_facts
        WHERE subject_user_id = ?
          AND fact_type = ?
          AND claim_text = ?
          AND status = ?
        ORDER BY confidence DESC, updated_at DESC
        LIMIT 1
        """,
        (item.subject_user_id, item.fact_type, item.claim_text, target_status),
    ).fetchone()
    if row:
        return _fact_record(row)
    if not item.evidence_message_id:
        return None
    row = conn.execute(
        """
        SELECT id, subject_user_id, fact_type, claim_text, topic, stance,
               confidence, status, claim_scope, source_user_id, source_group_id,
               evidence_message_id, evidence_text, created_at, updated_at
        FROM member_facts
        WHERE subject_user_id = ?
          AND evidence_message_id = ?
          AND claim_text = ?
          AND status = ?
        ORDER BY confidence DESC, updated_at DESC
        LIMIT 1
        """,
        (item.subject_user_id, item.evidence_message_id, item.claim_text, target_status),
    ).fetchone()
    return _fact_record(row) if row else None
