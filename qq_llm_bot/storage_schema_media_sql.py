from __future__ import annotations


MEDIA_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS image_vision_cache (
    url TEXT PRIMARY KEY,
    description TEXT NOT NULL DEFAULT '',
    ocr_text TEXT NOT NULL DEFAULT '',
    topics TEXT NOT NULL DEFAULT '[]',
    memory TEXT NOT NULL DEFAULT '',
    confidence REAL NOT NULL DEFAULT 0,
    importance REAL NOT NULL DEFAULT 0.5,
    model TEXT NOT NULL DEFAULT '',
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL,
    last_seen_at INTEGER NOT NULL,
    hit_count INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS sticker_assets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    group_id TEXT NOT NULL,
    source_user_id TEXT NOT NULL DEFAULT '',
    source_message_id TEXT NOT NULL DEFAULT '',
    url TEXT NOT NULL DEFAULT '',
    file TEXT NOT NULL DEFAULT '',
    local_path TEXT NOT NULL DEFAULT '',
    sha256 TEXT NOT NULL DEFAULT '',
    description TEXT NOT NULL DEFAULT '',
    ocr_text TEXT NOT NULL DEFAULT '',
    mood TEXT NOT NULL DEFAULT '',
    usage TEXT NOT NULL DEFAULT '',
    tags TEXT NOT NULL DEFAULT '[]',
    confidence REAL NOT NULL DEFAULT 0,
    enabled INTEGER NOT NULL DEFAULT 1,
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL,
    last_seen_at INTEGER NOT NULL,
    hit_count INTEGER NOT NULL DEFAULT 0,
    send_count INTEGER NOT NULL DEFAULT 0,
    last_sent_at INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS sticker_usage_daily (
    sticker_id INTEGER NOT NULL,
    group_id TEXT NOT NULL,
    usage_date TEXT NOT NULL,
    send_count INTEGER NOT NULL DEFAULT 0,
    first_sent_at INTEGER NOT NULL,
    last_sent_at INTEGER NOT NULL,
    PRIMARY KEY (sticker_id, usage_date)
);

CREATE TABLE IF NOT EXISTS image_generation_usage (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    usage_date TEXT NOT NULL,
    group_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    prompt TEXT NOT NULL DEFAULT '',
    image_ref TEXT NOT NULL DEFAULT '',
    created_at INTEGER NOT NULL
);
"""
