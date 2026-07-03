from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from threading import RLock
from typing import Iterable

from qq_llm_bot.config import AppConfig, ParticipationMode
from qq_llm_bot.models import MemoryCandidate, MessageContext


class BotStorage:
    def __init__(self, db_path: Path, initial_admins: Iterable[str], initial_groups: Iterable[str]) -> None:
        self.db_path = db_path
        self.initial_admins = {str(item) for item in initial_admins}
        self.initial_groups = {str(item) for item in initial_groups}
        self._lock = RLock()

    @classmethod
    def from_config(cls, config: AppConfig) -> "BotStorage":
        return cls(
            config.resolve_path(config.storage.sqlite_path),
            config.bot.admin_ids,
            config.bot.enabled_groups,
        )

    def setup(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    time INTEGER NOT NULL,
                    group_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    message_id TEXT NOT NULL,
                    raw_message TEXT NOT NULL,
                    plain_text TEXT NOT NULL,
                    sender_name TEXT NOT NULL DEFAULT '',
                    sender_role TEXT NOT NULL DEFAULT ''
                );

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

                CREATE TABLE IF NOT EXISTS memory_items (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    owner_type TEXT NOT NULL,
                    owner_id TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    content TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    evidence_message_id TEXT NOT NULL,
                    created_at INTEGER NOT NULL,
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
                """
            )

            now = int(time.time())
            conn.executemany(
                """
                INSERT INTO admins (user_id, added_at)
                VALUES (?, ?)
                ON CONFLICT(user_id) DO NOTHING
                """,
                [(user_id, now) for user_id in self.initial_admins],
            )
            conn.executemany(
                """
                INSERT INTO group_whitelist (group_id, enabled, updated_at)
                VALUES (?, 1, ?)
                ON CONFLICT(group_id) DO UPDATE SET enabled = 1, updated_at = excluded.updated_at
                """,
                [(group_id, now) for group_id in self.initial_groups],
            )

    def record_message(self, context: MessageContext) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO messages (
                    time, group_id, user_id, message_id, raw_message, plain_text,
                    sender_name, sender_role
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    context.timestamp or int(time.time()),
                    context.group_id,
                    context.user_id,
                    context.message_id,
                    context.raw_message,
                    context.plain_text,
                    context.sender_name,
                    context.sender_role,
                ),
            )

    def record_memories(self, memories: Iterable[MemoryCandidate]) -> None:
        rows = list(memories)
        if not rows:
            return
        now = int(time.time())
        with self._connect() as conn:
            conn.executemany(
                """
                INSERT INTO memory_items (
                    owner_type, owner_id, kind, content, confidence,
                    evidence_message_id, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        item.owner_type,
                        item.owner_id,
                        item.kind,
                        item.content,
                        item.confidence,
                        item.evidence_message_id,
                        now,
                        now,
                    )
                    for item in rows
                ],
            )

    def touch_relationship(self, group_id: str, user_id: str, familiarity_delta: int = 1) -> None:
        now = int(time.time())
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO relationships (group_id, user_id, familiarity, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(group_id, user_id) DO UPDATE SET
                    familiarity = MIN(100, relationships.familiarity + excluded.familiarity),
                    updated_at = excluded.updated_at
                """,
                (group_id, user_id, familiarity_delta, now),
            )

    def is_group_enabled(self, group_id: str) -> bool:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT enabled FROM group_whitelist WHERE group_id = ?",
                (str(group_id),),
            ).fetchone()
        if row is None:
            return False
        return bool(row["enabled"])

    def set_group_enabled(self, group_id: str, enabled: bool) -> None:
        now = int(time.time())
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO group_whitelist (group_id, enabled, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(group_id) DO UPDATE SET
                    enabled = excluded.enabled,
                    updated_at = excluded.updated_at
                """,
                (str(group_id), 1 if enabled else 0, now),
            )

    def list_enabled_groups(self) -> list[str]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT group_id FROM group_whitelist WHERE enabled = 1 ORDER BY group_id"
            ).fetchall()
        return [str(row["group_id"]) for row in rows]

    def get_group_mode(self, group_id: str, default_mode: ParticipationMode) -> ParticipationMode:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT mode FROM group_modes WHERE group_id = ?",
                (str(group_id),),
            ).fetchone()
        if row is None:
            return default_mode
        mode = str(row["mode"])
        if mode not in {"silent", "passive", "active"}:
            return default_mode
        return mode  # type: ignore[return-value]

    def set_group_mode(self, group_id: str, mode: ParticipationMode) -> None:
        now = int(time.time())
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO group_modes (group_id, mode, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(group_id) DO UPDATE SET
                    mode = excluded.mode,
                    updated_at = excluded.updated_at
                """,
                (str(group_id), mode, now),
            )

    def is_admin(self, user_id: str) -> bool:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM admins WHERE user_id = ?",
                (str(user_id),),
            ).fetchone()
        return row is not None

    def add_admin(self, user_id: str) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO admins (user_id, added_at)
                VALUES (?, ?)
                ON CONFLICT(user_id) DO NOTHING
                """,
                (str(user_id), int(time.time())),
            )

    def remove_admin(self, user_id: str) -> None:
        if str(user_id) in self.initial_admins:
            return
        with self._connect() as conn:
            conn.execute("DELETE FROM admins WHERE user_id = ?", (str(user_id),))

    def list_admins(self) -> list[str]:
        with self._connect() as conn:
            rows = conn.execute("SELECT user_id FROM admins ORDER BY user_id").fetchall()
        return [str(row["user_id"]) for row in rows]

    def list_user_memories(self, user_id: str, limit: int = 8) -> list[str]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, kind, content, confidence
                FROM memory_items
                WHERE owner_type = 'user' AND owner_id = ?
                ORDER BY updated_at DESC, id DESC
                LIMIT ?
                """,
                (str(user_id), limit),
            ).fetchall()
        return [
            f"#{row['id']} [{row['kind']}] {row['content']} ({row['confidence']:.2f})"
            for row in rows
        ]

    def _connect(self) -> sqlite3.Connection:
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            return conn


from qq_llm_bot.cognitive_storage import BotStorage as BotStorage  # noqa: E402,F401
