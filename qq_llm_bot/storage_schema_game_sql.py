from __future__ import annotations


GAME_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS guesswho_scores (
    group_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    correct_count INTEGER NOT NULL DEFAULT 0,
    wrong_count INTEGER NOT NULL DEFAULT 0,
    updated_at INTEGER NOT NULL,
    PRIMARY KEY (group_id, user_id)
);

CREATE INDEX IF NOT EXISTS idx_guesswho_scores_correct
    ON guesswho_scores(group_id, correct_count DESC, user_id);

CREATE INDEX IF NOT EXISTS idx_guesswho_scores_wrong
    ON guesswho_scores(group_id, wrong_count DESC, user_id);
"""
