from __future__ import annotations

import sqlite3

from qq_llm_bot.relationship_summary import merge_relationship_summary
from qq_llm_bot.storage_record_identity import _dashboard_user_id
from qq_llm_bot.storage_relationship_keys import GLOBAL_RELATIONSHIP_GROUP_ID


INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_member_facts_subject_status
    ON member_facts(subject_user_id, status, updated_at);
CREATE INDEX IF NOT EXISTS idx_member_facts_topic
    ON member_facts(subject_user_id, fact_type, topic, status);
CREATE INDEX IF NOT EXISTS idx_member_aliases_lookup
    ON member_aliases(alias, status, updated_at);
CREATE INDEX IF NOT EXISTS idx_member_aliases_user
    ON member_aliases(user_id, status, updated_at);
CREATE INDEX IF NOT EXISTS idx_image_generation_usage_user_date
    ON image_generation_usage(user_id, usage_date, created_at);
CREATE INDEX IF NOT EXISTS idx_llm_usage_created_at
    ON llm_usage(created_at);
CREATE INDEX IF NOT EXISTS idx_llm_usage_purpose_created_at
    ON llm_usage(purpose, created_at);
CREATE INDEX IF NOT EXISTS idx_final_qa_blocks_created_at
    ON final_qa_blocks(created_at);
CREATE INDEX IF NOT EXISTS idx_final_qa_blocks_group_user
    ON final_qa_blocks(group_id, user_id, created_at);
CREATE INDEX IF NOT EXISTS idx_final_qa_blocks_message
    ON final_qa_blocks(group_id, message_id);
CREATE INDEX IF NOT EXISTS idx_sticker_assets_last_sent
    ON sticker_assets(last_sent_at, created_at);
CREATE INDEX IF NOT EXISTS idx_sticker_usage_daily_group_date
    ON sticker_usage_daily(group_id, usage_date);
CREATE INDEX IF NOT EXISTS idx_relationships_global_user
    ON relationships(group_id, user_id, updated_at);
"""


def migrate_storage_schema(conn: sqlite3.Connection) -> None:
    _ensure_column(conn, "member_facts", "importance", "REAL NOT NULL DEFAULT 0.5")
    _ensure_column(conn, "member_facts", "last_seen_at", "INTEGER NOT NULL DEFAULT 0")
    _ensure_column(conn, "member_facts", "superseded_by_fact_id", "INTEGER")
    _ensure_column(conn, "member_facts", "forget_reason", "TEXT NOT NULL DEFAULT ''")
    _ensure_column(conn, "sticker_assets", "send_count", "INTEGER NOT NULL DEFAULT 0")
    _ensure_column(conn, "sticker_assets", "last_sent_at", "INTEGER NOT NULL DEFAULT 0")
    _migrate_relationships_to_global_user_scope(conn)
    conn.execute(
        """
        UPDATE member_facts
        SET last_seen_at = updated_at
        WHERE last_seen_at = 0
        """
    )
    conn.executescript(INDEX_SQL)


def _migrate_relationships_to_global_user_scope(conn: sqlite3.Connection) -> None:
    rows = conn.execute(
        """
        SELECT rowid AS row_id, group_id, user_id, closeness, trust,
               familiarity, tension, summary, updated_at
        FROM relationships
        ORDER BY user_id, updated_at ASC, row_id ASC
        """
    ).fetchall()
    if not rows:
        return

    grouped: dict[str, dict[str, object]] = {}
    needs_rewrite = False
    for row in rows:
        raw_user_id = str(row["user_id"] or "")
        user_id = _dashboard_user_id(raw_user_id)
        if not user_id:
            needs_rewrite = True
            continue
        group_id = str(row["group_id"] or "")
        needs_rewrite = needs_rewrite or group_id != GLOBAL_RELATIONSHIP_GROUP_ID
        needs_rewrite = needs_rewrite or user_id != raw_user_id
        current = grouped.setdefault(
            user_id,
            {
                "closeness": 0,
                "trust": 0,
                "familiarity": 0,
                "tension": 0,
                "summary": "",
                "updated_at": 0,
            },
        )
        current["closeness"] = int(current["closeness"]) + int(row["closeness"] or 0)
        current["trust"] = int(current["trust"]) + int(row["trust"] or 0)
        current["familiarity"] = int(current["familiarity"]) + int(row["familiarity"] or 0)
        current["tension"] = int(current["tension"]) + int(row["tension"] or 0)
        current["summary"] = merge_relationship_summary(
            str(current["summary"]),
            str(row["summary"] or ""),
        )
        current["updated_at"] = max(int(current["updated_at"]), int(row["updated_at"] or 0))

    if not needs_rewrite:
        return

    conn.execute("DELETE FROM relationships")
    conn.executemany(
        """
        INSERT INTO relationships (
            group_id, user_id, closeness, trust, familiarity, tension, summary, updated_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                GLOBAL_RELATIONSHIP_GROUP_ID,
                user_id,
                _clamp_relationship_score(values["closeness"]),
                _clamp_relationship_score(values["trust"]),
                _clamp_relationship_score(values["familiarity"]),
                _clamp_relationship_score(values["tension"]),
                str(values["summary"]),
                int(values["updated_at"]),
            )
            for user_id, values in grouped.items()
        ],
    )


def _clamp_relationship_score(value: object) -> int:
    return max(0, min(100, int(value)))


def _ensure_column(
    conn: sqlite3.Connection,
    table: str,
    column: str,
    definition: str,
) -> None:
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    existing = {str(row["name"]) for row in rows}
    if column not in existing:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")
