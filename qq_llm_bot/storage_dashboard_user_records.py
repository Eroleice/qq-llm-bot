from __future__ import annotations

import sqlite3

from qq_llm_bot.models import FactRecord, UserProfileRecord
from qq_llm_bot.storage_records import (
    _dashboard_user_id_variants,
    _fact_record,
    _user_profile_record,
)
from qq_llm_bot.storage_relationship_keys import GLOBAL_RELATIONSHIP_GROUP_ID


def list_dashboard_relationship_rows(
    conn: sqlite3.Connection,
    user_id: str,
    limit: int,
) -> list[sqlite3.Row]:
    variants = _dashboard_user_id_variants(user_id)
    placeholders = ", ".join("?" for _ in variants)
    return conn.execute(
        f"""
        SELECT group_id, user_id, closeness, trust, familiarity, tension, summary, updated_at
        FROM relationships
        WHERE group_id = ?
          AND user_id IN ({placeholders})
        ORDER BY updated_at DESC
        LIMIT ?
        """,
        [GLOBAL_RELATIONSHIP_GROUP_ID, *variants, int(limit)],
    ).fetchall()


def list_dashboard_user_fact_records(
    conn: sqlite3.Connection,
    user_id: str,
    limit: int,
) -> list[FactRecord]:
    variants = _dashboard_user_id_variants(user_id)
    placeholders = ", ".join("?" for _ in variants)
    rows = conn.execute(
        f"""
        SELECT id, subject_user_id, fact_type, claim_text, topic, stance,
               confidence, status, claim_scope, source_user_id, source_group_id,
               evidence_message_id, evidence_text, created_at, updated_at
        FROM member_facts
        WHERE status = 'accepted'
          AND subject_user_id IN ({placeholders})
        ORDER BY updated_at DESC, id DESC
        LIMIT ?
        """,
        [*variants, int(limit)],
    ).fetchall()
    return [_fact_record(row) for row in rows]


def dashboard_member_profile(
    conn: sqlite3.Connection,
    user_id: str,
) -> UserProfileRecord | None:
    variants = _dashboard_user_id_variants(user_id)
    placeholders = ", ".join("?" for _ in variants)
    row = conn.execute(
        f"""
        SELECT user_id, summary, traits_json, supporting_fact_ids,
               fact_count, version, updated_at
        FROM member_profiles
        WHERE user_id IN ({placeholders})
        ORDER BY updated_at DESC
        LIMIT 1
        """,
        variants,
    ).fetchone()
    return _user_profile_record(row) if row else None


def dashboard_user_profile(
    conn: sqlite3.Connection,
    user_id: str,
) -> dict[str, object]:
    variants = _dashboard_user_id_variants(user_id)
    placeholders = ", ".join("?" for _ in variants)
    row = conn.execute(
        f"""
        SELECT nickname, display_name, updated_at, last_seen_at
        FROM user_profiles
        WHERE user_id IN ({placeholders})
        ORDER BY last_seen_at DESC
        LIMIT 1
        """,
        variants,
    ).fetchone()
    if row is not None:
        return {
            "nickname": str(row["nickname"] or ""),
            "display_name": str(row["display_name"] or ""),
            "updated_at": int(row["updated_at"]),
            "last_seen_at": int(row["last_seen_at"]),
        }

    row = conn.execute(
        f"""
        SELECT sender_name, time
        FROM messages
        WHERE user_id IN ({placeholders})
          AND sender_name != ''
        ORDER BY time DESC, id DESC
        LIMIT 1
        """,
        variants,
    ).fetchone()
    if row is None:
        return {"nickname": "", "display_name": "", "updated_at": 0, "last_seen_at": 0}
    return {
        "nickname": "",
        "display_name": str(row["sender_name"] or ""),
        "updated_at": int(row["time"]),
        "last_seen_at": int(row["time"]),
    }
