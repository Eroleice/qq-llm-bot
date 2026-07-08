from __future__ import annotations

import sqlite3

from qq_llm_bot.models import FactCandidate, FactRecord


def insert_fact(
    conn: sqlite3.Connection,
    item: FactCandidate,
    now: int,
) -> FactRecord:
    cursor = conn.execute(
        """
        INSERT INTO member_facts (
            subject_user_id, fact_type, claim_text, topic, stance,
            confidence, status, claim_scope, source_user_id, source_group_id,
            evidence_message_id, evidence_text, created_at, updated_at,
            importance, last_seen_at, superseded_by_fact_id, forget_reason
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            item.subject_user_id,
            item.fact_type,
            item.claim_text,
            item.topic,
            item.stance,
            item.confidence,
            item.status,
            item.claim_scope,
            item.source_user_id,
            item.source_group_id,
            item.evidence_message_id,
            item.evidence_text,
            now,
            now,
            item.importance,
            now,
            None,
            "",
        ),
    )
    return FactRecord(
        id=int(cursor.lastrowid),
        subject_user_id=item.subject_user_id,
        fact_type=item.fact_type,
        claim_text=item.claim_text,
        topic=item.topic,
        stance=item.stance,
        confidence=item.confidence,
        status=item.status,
        claim_scope=item.claim_scope,
        source_user_id=item.source_user_id,
        source_group_id=item.source_group_id,
        evidence_message_id=item.evidence_message_id,
        evidence_text=item.evidence_text,
        created_at=now,
        updated_at=now,
        importance=item.importance,
        last_seen_at=now,
    )
