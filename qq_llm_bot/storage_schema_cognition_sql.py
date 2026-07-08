from __future__ import annotations


COGNITION_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS memory_items (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    owner_type TEXT NOT NULL,
    owner_id TEXT NOT NULL,
    kind TEXT NOT NULL,
    content TEXT NOT NULL,
    confidence REAL NOT NULL,
    importance REAL NOT NULL DEFAULT 0.5,
    status TEXT NOT NULL DEFAULT 'active',
    evidence_message_id TEXT NOT NULL,
    source_text TEXT NOT NULL DEFAULT '',
    source_user_id TEXT NOT NULL DEFAULT '',
    source_group_id TEXT NOT NULL DEFAULT '',
    subject_user_id TEXT NOT NULL DEFAULT '',
    claim_scope TEXT NOT NULL DEFAULT 'self_report',
    verification_status TEXT NOT NULL DEFAULT 'accepted',
    conflict_of INTEGER,
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL,
    last_seen_at INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS member_facts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    subject_user_id TEXT NOT NULL,
    fact_type TEXT NOT NULL,
    claim_text TEXT NOT NULL,
    topic TEXT NOT NULL,
    stance TEXT NOT NULL DEFAULT '',
    confidence REAL NOT NULL,
    status TEXT NOT NULL DEFAULT 'accepted',
    claim_scope TEXT NOT NULL DEFAULT 'self_report',
    source_user_id TEXT NOT NULL DEFAULT '',
    source_group_id TEXT NOT NULL DEFAULT '',
    evidence_message_id TEXT NOT NULL DEFAULT '',
    evidence_text TEXT NOT NULL DEFAULT '',
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL,
    importance REAL NOT NULL DEFAULT 0.5,
    last_seen_at INTEGER NOT NULL DEFAULT 0,
    superseded_by_fact_id INTEGER,
    forget_reason TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS member_aliases (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    alias TEXT NOT NULL,
    alias_type TEXT NOT NULL DEFAULT 'alias',
    status TEXT NOT NULL DEFAULT 'active',
    confidence REAL NOT NULL DEFAULT 0.5,
    source_fact_id INTEGER,
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL,
    last_seen_at INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS member_profiles (
    user_id TEXT PRIMARY KEY,
    summary TEXT NOT NULL DEFAULT '',
    traits_json TEXT NOT NULL DEFAULT '{}',
    supporting_fact_ids TEXT NOT NULL DEFAULT '[]',
    fact_count INTEGER NOT NULL DEFAULT 0,
    version INTEGER NOT NULL DEFAULT 0,
    updated_at INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS relationships (
    group_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    closeness INTEGER NOT NULL DEFAULT 0,
    trust INTEGER NOT NULL DEFAULT 0,
    familiarity INTEGER NOT NULL DEFAULT 0,
    tension INTEGER NOT NULL DEFAULT 0,
    summary TEXT NOT NULL DEFAULT '',
    updated_at INTEGER NOT NULL,
    PRIMARY KEY (group_id, user_id)
);

CREATE TABLE IF NOT EXISTS persona_state (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at INTEGER NOT NULL
);
"""
