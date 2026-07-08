from __future__ import annotations

import sqlite3


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
"""


def migrate_storage_schema(conn: sqlite3.Connection) -> None:
    _ensure_column(conn, "member_facts", "importance", "REAL NOT NULL DEFAULT 0.5")
    _ensure_column(conn, "member_facts", "last_seen_at", "INTEGER NOT NULL DEFAULT 0")
    _ensure_column(conn, "member_facts", "superseded_by_fact_id", "INTEGER")
    _ensure_column(conn, "member_facts", "forget_reason", "TEXT NOT NULL DEFAULT ''")
    _ensure_column(conn, "sticker_assets", "send_count", "INTEGER NOT NULL DEFAULT 0")
    _ensure_column(conn, "sticker_assets", "last_sent_at", "INTEGER NOT NULL DEFAULT 0")
    conn.execute(
        """
        UPDATE member_facts
        SET last_seen_at = updated_at
        WHERE last_seen_at = 0
        """
    )
    conn.executescript(INDEX_SQL)


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
