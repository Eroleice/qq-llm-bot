from __future__ import annotations

import sqlite3

from qq_llm_bot.models import FactCandidate, FactRecord
from qq_llm_bot.storage_fact_constants import FACT_INACTIVE_STATUSES
from qq_llm_bot.storage_helpers import extract_denied_aliases as _extract_denied_aliases
from qq_llm_bot.storage_records import _fact_record


def find_conflicting_facts(
    conn: sqlite3.Connection,
    item: FactCandidate,
) -> list[FactRecord]:
    if item.status in FACT_INACTIVE_STATUSES:
        return []
    if item.fact_type in {"identity", "alias"}:
        denied_aliases = _extract_denied_aliases(item.claim_text, item.evidence_text)
        if not denied_aliases:
            return []
        clauses = []
        params: list[object] = [item.subject_user_id]
        for alias in denied_aliases:
            clauses.append("(claim_text LIKE ? OR evidence_text LIKE ? OR topic LIKE ?)")
            like = f"%{alias}%"
            params.extend([like, like, like])
        rows = conn.execute(
            f"""
            SELECT id, subject_user_id, fact_type, claim_text, topic, stance,
                   confidence, status, claim_scope, source_user_id, source_group_id,
                   evidence_message_id, evidence_text, created_at, updated_at,
                   importance, last_seen_at, superseded_by_fact_id, forget_reason
            FROM member_facts
            WHERE subject_user_id = ?
              AND status = 'accepted'
              AND fact_type IN ('identity', 'alias')
              AND ({' OR '.join(clauses)})
            ORDER BY confidence DESC, updated_at DESC
            LIMIT 10
            """,
            params,
        ).fetchall()
        return [_fact_record(row) for row in rows]

    if item.fact_type not in {"preference", "dislike", "opinion", "habit", "boundary", "event_stance"}:
        return []
    rows = conn.execute(
        """
        SELECT id, subject_user_id, fact_type, claim_text, topic, stance,
               confidence, status, claim_scope, source_user_id, source_group_id,
               evidence_message_id, evidence_text, created_at, updated_at,
               importance, last_seen_at, superseded_by_fact_id, forget_reason
        FROM member_facts
        WHERE subject_user_id = ?
          AND fact_type = ?
          AND topic = ?
          AND status = 'accepted'
          AND claim_text != ?
        ORDER BY confidence DESC, updated_at DESC
        LIMIT 5
        """,
        (item.subject_user_id, item.fact_type, item.topic, item.claim_text),
    ).fetchall()
    return [_fact_record(row) for row in rows]
