from __future__ import annotations


ACCESS_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS group_whitelist (
    group_id TEXT PRIMARY KEY,
    enabled INTEGER NOT NULL,
    updated_at INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS group_modes (
    group_id TEXT PRIMARY KEY,
    mode TEXT NOT NULL,
    updated_at INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS admins (
    user_id TEXT PRIMARY KEY,
    added_at INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS ignored_users (
    user_id TEXT PRIMARY KEY,
    ignored INTEGER NOT NULL,
    updated_at INTEGER NOT NULL
);
"""
