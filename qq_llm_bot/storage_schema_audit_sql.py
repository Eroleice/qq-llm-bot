from __future__ import annotations


AUDIT_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS bot_decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    time INTEGER NOT NULL,
    group_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    message_id TEXT NOT NULL,
    mode TEXT NOT NULL,
    action TEXT NOT NULL,
    reason TEXT NOT NULL,
    score REAL NOT NULL DEFAULT 0,
    value_type TEXT NOT NULL DEFAULT '',
    value_score REAL NOT NULL DEFAULT 0,
    traffic_level TEXT NOT NULL DEFAULT 'normal',
    reply TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS final_qa_blocks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at INTEGER NOT NULL,
    message_time INTEGER NOT NULL DEFAULT 0,
    group_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    message_id TEXT NOT NULL,
    sender_name TEXT NOT NULL DEFAULT '',
    sender_role TEXT NOT NULL DEFAULT '',
    trigger_text TEXT NOT NULL DEFAULT '',
    raw_message TEXT NOT NULL DEFAULT '',
    is_direct INTEGER NOT NULL DEFAULT 0,
    bot_mentioned INTEGER NOT NULL DEFAULT 0,
    mode TEXT NOT NULL DEFAULT '',
    action TEXT NOT NULL DEFAULT '',
    decision_reason TEXT NOT NULL DEFAULT '',
    score REAL NOT NULL DEFAULT 0,
    value_type TEXT NOT NULL DEFAULT '',
    value_score REAL NOT NULL DEFAULT 0,
    traffic_level TEXT NOT NULL DEFAULT 'normal',
    candidate_reply TEXT NOT NULL DEFAULT '',
    qa_reason TEXT NOT NULL DEFAULT '',
    qa_categories TEXT NOT NULL DEFAULT '[]',
    qa_confidence REAL NOT NULL DEFAULT 0,
    recent_messages TEXT NOT NULL DEFAULT '[]',
    speaker_recent_messages TEXT NOT NULL DEFAULT '[]',
    other_recent_messages TEXT NOT NULL DEFAULT '[]',
    recent_image_descriptions TEXT NOT NULL DEFAULT '[]'
);

CREATE TABLE IF NOT EXISTS llm_usage (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at INTEGER NOT NULL,
    purpose TEXT NOT NULL DEFAULT '',
    model TEXT NOT NULL DEFAULT '',
    prompt_chars INTEGER NOT NULL DEFAULT 0,
    completion_chars INTEGER NOT NULL DEFAULT 0,
    prompt_tokens INTEGER NOT NULL DEFAULT 0,
    completion_tokens INTEGER NOT NULL DEFAULT 0,
    total_tokens INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS bot_maintenance_state (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL DEFAULT '',
    updated_at INTEGER NOT NULL
);
"""
