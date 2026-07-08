from __future__ import annotations

import sqlite3
import time

from qq_llm_bot.models import FactRecord
from qq_llm_bot.storage_helpers import (
    extract_aliases_from_fact as _extract_aliases_from_fact,
    extract_denied_aliases as _extract_denied_aliases,
)


def supersede_facts(
    conn: sqlite3.Connection,
    records: list[FactRecord],
    replacement_fact_id: int,
    now: int,
) -> None:
    ids = [record.id for record in records if record.status == "accepted"]
    if not ids:
        return
    placeholders = ", ".join("?" for _ in ids)
    conn.execute(
        f"""
        UPDATE member_facts
        SET status = 'superseded',
            superseded_by_fact_id = ?,
            forget_reason = 'superseded_by_new_self_report',
            updated_at = ?
        WHERE id IN ({placeholders})
        """,
        [replacement_fact_id, now, *ids],
    )
    conn.execute(
        f"""
        UPDATE member_aliases
        SET status = 'superseded',
            updated_at = ?
        WHERE source_fact_id IN ({placeholders})
          AND status = 'active'
        """,
        [now, *ids],
    )


def sync_aliases_for_fact(conn: sqlite3.Connection, fact: FactRecord) -> None:
    if fact.status != "accepted" or fact.fact_type not in {"identity", "alias"}:
        return
    now = int(time.time())
    denied_aliases = _extract_denied_aliases(fact.claim_text, fact.evidence_text)
    for alias in denied_aliases:
        conn.execute(
            """
            UPDATE member_aliases
            SET status = 'superseded',
                updated_at = ?
            WHERE user_id = ?
              AND alias = ?
              AND status = 'active'
            """,
            (now, fact.subject_user_id, alias),
        )

    for alias, alias_type in _extract_aliases_from_fact(fact):
        row = conn.execute(
            """
            SELECT id
            FROM member_aliases
            WHERE user_id = ?
              AND alias = ?
              AND alias_type = ?
              AND status = 'active'
            ORDER BY updated_at DESC, id DESC
            LIMIT 1
            """,
            (fact.subject_user_id, alias, alias_type),
        ).fetchone()
        if row is not None:
            conn.execute(
                """
                UPDATE member_aliases
                SET confidence = MAX(confidence, ?),
                    source_fact_id = ?,
                    updated_at = ?,
                    last_seen_at = ?
                WHERE id = ?
                """,
                (fact.confidence, fact.id, now, now, int(row["id"])),
            )
            continue
        conn.execute(
            """
            INSERT INTO member_aliases (
                user_id, alias, alias_type, status, confidence,
                source_fact_id, created_at, updated_at, last_seen_at
            )
            VALUES (?, ?, ?, 'active', ?, ?, ?, ?, ?)
            """,
            (fact.subject_user_id, alias, alias_type, fact.confidence, fact.id, now, now, now),
        )
