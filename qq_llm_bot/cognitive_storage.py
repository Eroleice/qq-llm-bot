from __future__ import annotations

import json
import sqlite3
import re
import time
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from threading import RLock
from typing import Iterator, Iterable

from qq_llm_bot.config import AppConfig, ParticipationMode
from qq_llm_bot.llm import LLMUsageRecord
from qq_llm_bot.models import (
    ConversationSnapshot,
    FactCandidate,
    FactRecord,
    FactWriteSet,
    ImageVisionCacheRecord,
    MemoryCandidate,
    MemoryRecord,
    MemoryWriteSet,
    MessageContext,
    ParticipationDecision,
    RelationDelta,
    RelationshipState,
    StickerAssetRecord,
    StickerCandidate,
    TargetUserContext,
    UserProfileRecord,
    UserProfileDraft,
)
from qq_llm_bot.onebot_messages import strip_quoted_messages
from qq_llm_bot.relationship_summary import merge_relationship_summary

CONFLICT_SENSITIVE_KINDS = {
    "alias",
    "identity",
    "location",
    "preference",
    "dislike",
    "experience",
    "self_experience",
    "self_preference",
    "self_boundary",
    "persona_fact",
}
SENSITIVE_CONFIRMATION_KINDS = {"identity", "location"}
TRUSTED_THIRD_PARTY_THRESHOLD = 70
FACT_INACTIVE_STATUSES = {"rejected", "superseded", "forgotten"}
PROTECTED_FACT_TYPES = {"identity", "boundary", "skill", "habit"}
RELATIONAL_ALIAS_TERMS = {
    "主人",
    "主子",
    "老板",
    "老板娘",
    "领导",
    "管理员",
    "管理",
    "群主",
    "版主",
    "爸爸",
    "爸",
    "爹",
    "父亲",
    "妈妈",
    "妈",
    "母亲",
    "哥哥",
    "哥",
    "姐姐",
    "姐",
    "弟弟",
    "弟",
    "妹妹",
    "妹",
    "儿子",
    "女儿",
    "老婆",
    "老公",
    "媳妇",
    "丈夫",
    "妻子",
    "对象",
    "男朋友",
    "女朋友",
}


class BotStorage:
    def __init__(
        self,
        db_path: Path,
        initial_admins: Iterable[str],
        initial_ignored_users: Iterable[str],
        initial_groups: Iterable[str],
        initial_persona: dict[str, str],
        sticker_context_limit: int = 24,
        fact_confidence_threshold: float = 0.75,
        third_party_trust_threshold: int = 70,
        third_party_confidence_threshold: float = 0.85,
        profile_fact_threshold: int = 5,
        context_fact_limit: int = 8,
        target_user_limit: int = 5,
        low_importance_threshold: float = 0.35,
        fact_context_ttl_days: int = 30,
        interaction_followup_seconds: int = 180,
    ) -> None:
        self.db_path = db_path
        self.initial_admins = {str(item) for item in initial_admins}
        self.initial_ignored_users = {_dashboard_user_id(str(item)) for item in initial_ignored_users}
        self.initial_ignored_users.discard("")
        self.initial_groups = {str(item) for item in initial_groups}
        self.initial_persona = initial_persona
        self.sticker_context_limit = max(1, int(sticker_context_limit))
        self.fact_confidence_threshold = _clamp_float(fact_confidence_threshold)
        self.third_party_trust_threshold = _clamp_score(third_party_trust_threshold)
        self.third_party_confidence_threshold = _clamp_float(third_party_confidence_threshold)
        self.profile_fact_threshold = max(1, int(profile_fact_threshold))
        self.context_fact_limit = max(1, int(context_fact_limit))
        self.target_user_limit = max(1, int(target_user_limit))
        self.low_importance_threshold = _clamp_float(low_importance_threshold)
        self.fact_context_ttl_seconds = max(1, int(fact_context_ttl_days)) * 24 * 60 * 60
        self.interaction_followup_seconds = max(1, int(interaction_followup_seconds))
        self._lock = RLock()

    @classmethod
    def from_config(cls, config: AppConfig) -> "BotStorage":
        persona = {
            "self_name": config.persona.self_name,
            "core_traits": "、".join(config.persona.core_traits),
            "speech_style": "、".join(config.persona.speech_style),
            "boundaries": "、".join(config.persona.boundaries),
            "current_mood": config.persona.current_mood,
            "relationship_tendency": config.persona.relationship_tendency,
            "activity_level": str(config.persona.activity_level),
        }
        persona.update(
            _compact_persona_items(
                {
                    "full_name": config.persona.full_name,
                    "gender": config.persona.gender,
                    "age": str(config.persona.age) if config.persona.age else "",
                    "city": config.persona.city,
                    "education_school": config.persona.education_school,
                    "education_major": config.persona.education_major,
                    "education_degree": config.persona.education_degree,
                    "employer": config.persona.employer,
                    "occupation": config.persona.occupation,
                    "work_years": str(config.persona.work_years)
                    if config.persona.work_years
                    else "",
                    "relationship_status": config.persona.relationship_status,
                    "background_summary": config.persona.background_summary,
                    "appearance_prompt": config.persona.appearance_prompt,
                }
            )
        )
        return cls(
            config.resolve_path(config.storage.sqlite_path),
            config.bot.admin_ids,
            config.bot.ignored_user_ids,
            config.bot.enabled_groups,
            persona,
            config.stickers.max_context_stickers,
            config.facts.fact_confidence_threshold,
            config.facts.third_party_trust_threshold,
            config.facts.third_party_confidence_threshold,
            config.facts.profile_fact_threshold,
            config.facts.context_fact_limit,
            config.facts.target_user_limit,
            config.facts.low_importance_threshold,
            config.facts.fact_context_ttl_days,
            config.bot.interaction_followup_seconds,
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

                CREATE TABLE IF NOT EXISTS message_attachments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    time INTEGER NOT NULL,
                    group_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    message_id TEXT NOT NULL,
                    attachment_type TEXT NOT NULL,
                    file TEXT NOT NULL DEFAULT '',
                    url TEXT NOT NULL DEFAULT '',
                    summary TEXT NOT NULL DEFAULT '',
                    raw_data TEXT NOT NULL DEFAULT ''
                );

                CREATE TABLE IF NOT EXISTS message_mentions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    time INTEGER NOT NULL,
                    group_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    message_id TEXT NOT NULL,
                    mentioned_user_id TEXT NOT NULL,
                    display_name TEXT NOT NULL DEFAULT '',
                    is_bot INTEGER NOT NULL DEFAULT 0,
                    raw_data TEXT NOT NULL DEFAULT ''
                );

                CREATE TABLE IF NOT EXISTS user_profiles (
                    user_id TEXT PRIMARY KEY,
                    nickname TEXT NOT NULL DEFAULT '',
                    display_name TEXT NOT NULL DEFAULT '',
                    updated_at INTEGER NOT NULL,
                    last_seen_at INTEGER NOT NULL
                );

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

                CREATE TABLE IF NOT EXISTS image_generation_usage (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    usage_date TEXT NOT NULL,
                    group_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    prompt TEXT NOT NULL DEFAULT '',
                    image_ref TEXT NOT NULL DEFAULT '',
                    created_at INTEGER NOT NULL
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
            )
            self._migrate_schema(conn)
            self._backfill_member_aliases(conn)
            self._reject_unreasonable_member_aliases(conn)
            self._seed_config(conn)

    def _migrate_schema(self, conn: sqlite3.Connection) -> None:
        self._ensure_column(conn, "member_facts", "importance", "REAL NOT NULL DEFAULT 0.5")
        self._ensure_column(conn, "member_facts", "last_seen_at", "INTEGER NOT NULL DEFAULT 0")
        self._ensure_column(conn, "member_facts", "superseded_by_fact_id", "INTEGER")
        self._ensure_column(conn, "member_facts", "forget_reason", "TEXT NOT NULL DEFAULT ''")
        self._ensure_column(conn, "sticker_assets", "send_count", "INTEGER NOT NULL DEFAULT 0")
        self._ensure_column(conn, "sticker_assets", "last_sent_at", "INTEGER NOT NULL DEFAULT 0")
        conn.execute(
            """
            UPDATE member_facts
            SET last_seen_at = updated_at
            WHERE last_seen_at = 0
            """
        )
        conn.executescript(
            """
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
            CREATE INDEX IF NOT EXISTS idx_sticker_assets_last_sent
                ON sticker_assets(last_sent_at, created_at);
            CREATE INDEX IF NOT EXISTS idx_sticker_usage_daily_group_date
                ON sticker_usage_daily(group_id, usage_date);
            """
        )

    def _ensure_column(
        self,
        conn: sqlite3.Connection,
        table: str,
        column: str,
        definition: str,
    ) -> None:
        rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
        existing = {str(row["name"]) for row in rows}
        if column not in existing:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")

    def _backfill_member_aliases(self, conn: sqlite3.Connection) -> None:
        rows = conn.execute(
            """
            SELECT id, subject_user_id, fact_type, claim_text, topic, stance,
                   confidence, status, claim_scope, source_user_id, source_group_id,
                   evidence_message_id, evidence_text, created_at, updated_at,
                   importance, last_seen_at, superseded_by_fact_id, forget_reason
            FROM member_facts
            WHERE status = 'accepted'
              AND fact_type IN ('identity', 'alias')
            ORDER BY updated_at ASC, id ASC
            """
        ).fetchall()
        for row in rows:
            self._sync_aliases_for_fact(conn, _fact_record(row))

    def _reject_unreasonable_member_aliases(self, conn: sqlite3.Connection) -> None:
        rows = conn.execute(
            """
            SELECT id, alias
            FROM member_aliases
            WHERE status = 'active'
            """
        ).fetchall()
        ids = [
            int(row["id"])
            for row in rows
            if not _is_reasonable_member_alias(str(row["alias"] or ""))
        ]
        if not ids:
            return
        placeholders = ", ".join("?" for _ in ids)
        conn.execute(
            f"""
            UPDATE member_aliases
            SET status = 'rejected',
                updated_at = ?
            WHERE id IN ({placeholders})
            """,
            [int(time.time()), *ids],
        )

    def record_message(self, context: MessageContext) -> None:
        now = context.timestamp or int(time.time())
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
                    now,
                    context.group_id,
                    context.user_id,
                    context.message_id,
                    context.raw_message,
                    context.plain_text,
                    context.sender_name,
                    context.sender_role,
                ),
            )
            self._upsert_user_profile(conn, context, now)
            self._record_attachments(conn, context)
            self._record_mentions(conn, context)

    def _upsert_user_profile(
        self,
        conn: sqlite3.Connection,
        context: MessageContext,
        now: int,
    ) -> None:
        user_id = _dashboard_user_id(context.user_id)
        if not user_id:
            return
        nickname = " ".join(str(context.sender_nickname or "").split())
        display_name = " ".join(str(context.sender_name or "").split())
        conn.execute(
            """
            INSERT INTO user_profiles (user_id, nickname, display_name, updated_at, last_seen_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(user_id) DO UPDATE SET
                nickname = CASE
                    WHEN excluded.nickname != '' THEN excluded.nickname
                    ELSE user_profiles.nickname
                END,
                display_name = CASE
                    WHEN excluded.display_name != '' THEN excluded.display_name
                    ELSE user_profiles.display_name
                END,
                updated_at = excluded.updated_at,
                last_seen_at = excluded.last_seen_at
            """,
            (user_id, nickname, display_name, now, now),
        )

    def record_bot_reply(self, group_id: str, bot_id: str, reply: str) -> None:
        self.record_bot_reply_parts(group_id, bot_id, [reply])

    def record_bot_reply_parts(self, group_id: str, bot_id: str, replies: Iterable[str]) -> None:
        now = int(time.time())
        clean_replies = [str(reply or "").strip() for reply in replies if str(reply or "").strip()]
        for index, reply in enumerate(clean_replies, start=1):
            self.record_message(
                MessageContext(
                    group_id=str(group_id),
                    user_id=str(bot_id),
                    message_id=f"bot-{now}-{index}",
                    plain_text=reply,
                    raw_message=reply,
                    sender_name="bot",
                    sender_role="bot",
                    timestamp=now,
                )
            )

    def get_recent_bot_reply_texts(self, group_id: str, limit: int = 10) -> list[str]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT plain_text
                FROM messages
                WHERE group_id = ?
                  AND sender_role = 'bot'
                  AND plain_text != ''
                ORDER BY id DESC
                LIMIT ?
                """,
                (str(group_id), max(1, int(limit))),
            ).fetchall()
        return [str(row["plain_text"] or "") for row in rows if str(row["plain_text"] or "").strip()]

    def _record_attachments(self, conn: sqlite3.Connection, context: MessageContext) -> None:
        if not context.attachments:
            return
        now = context.timestamp or int(time.time())
        conn.executemany(
            """
            INSERT INTO message_attachments (
                time, group_id, user_id, message_id, attachment_type,
                file, url, summary, raw_data
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    now,
                    context.group_id,
                    context.user_id,
                    context.message_id,
                    attachment.attachment_type,
                    attachment.file,
                    attachment.url,
                    attachment.summary,
                    attachment.raw_data,
                )
                for attachment in context.attachments
            ],
        )

    def _record_mentions(self, conn: sqlite3.Connection, context: MessageContext) -> None:
        if not context.mentions:
            return
        now = context.timestamp or int(time.time())
        conn.executemany(
            """
            INSERT INTO message_mentions (
                time, group_id, user_id, message_id, mentioned_user_id,
                display_name, is_bot, raw_data
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    now,
                    context.group_id,
                    context.user_id,
                    context.message_id,
                    mention.user_id,
                    mention.display_name,
                    1 if mention.is_bot else 0,
                    mention.raw_data,
                )
                for mention in context.mentions
            ],
        )

    def update_image_descriptions(
        self,
        group_id: str,
        message_id: str,
        descriptions: list[str],
    ) -> None:
        if not descriptions:
            return
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id
                FROM message_attachments
                WHERE group_id = ?
                  AND message_id = ?
                  AND attachment_type = 'image'
                ORDER BY id
                """,
                (str(group_id), str(message_id)),
            ).fetchall()
            for row, description in zip(rows, descriptions):
                if not str(description).strip():
                    continue
                conn.execute(
                    """
                    UPDATE message_attachments
                    SET summary = ?
                    WHERE id = ?
                    """,
                    (description[:500], int(row["id"])),
                )

    def get_image_vision_cache(self, url: str) -> ImageVisionCacheRecord | None:
        normalized_url = str(url).strip()
        if not normalized_url:
            return None
        now = int(time.time())
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT url, description, ocr_text, topics, memory, confidence, importance,
                       model, created_at, updated_at, last_seen_at, hit_count
                FROM image_vision_cache
                WHERE url = ?
                """,
                (normalized_url,),
            ).fetchone()
            if row is None:
                return None
            conn.execute(
                """
                UPDATE image_vision_cache
                SET last_seen_at = ?,
                    hit_count = hit_count + 1
                WHERE url = ?
                """,
                (now, normalized_url),
            )
        record = _image_vision_cache_record(row)
        return replace(record, last_seen_at=now, hit_count=record.hit_count + 1)

    def upsert_image_vision_cache(
        self,
        *,
        url: str,
        description: str,
        ocr_text: str = "",
        topics: Iterable[str] = (),
        memory: str = "",
        confidence: float = 0.0,
        importance: float = 0.5,
        model: str = "",
    ) -> None:
        normalized_url = str(url).strip()
        description = str(description).strip()
        ocr_text = str(ocr_text).strip()
        memory = str(memory).strip()
        if not normalized_url or not (description or ocr_text or memory):
            return
        now = int(time.time())
        topics_json = json.dumps(_compact_string_list(topics, limit=10), ensure_ascii=False)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO image_vision_cache (
                    url, description, ocr_text, topics, memory, confidence, importance,
                    model, created_at, updated_at, last_seen_at, hit_count
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
                ON CONFLICT(url) DO UPDATE SET
                    description = excluded.description,
                    ocr_text = excluded.ocr_text,
                    topics = excluded.topics,
                    memory = excluded.memory,
                    confidence = excluded.confidence,
                    importance = excluded.importance,
                    model = excluded.model,
                    updated_at = excluded.updated_at,
                    last_seen_at = excluded.last_seen_at,
                    hit_count = image_vision_cache.hit_count + 1
                """,
                (
                    normalized_url,
                    description[:1000],
                    ocr_text[:1000],
                    topics_json,
                    memory[:1000],
                    float(confidence),
                    float(importance),
                    str(model).strip()[:80],
                    now,
                    now,
                    now,
                ),
            )

    def upsert_sticker_asset(
        self,
        context: MessageContext,
        candidate: StickerCandidate,
        local_path: str,
        sha256: str = "",
    ) -> StickerAssetRecord | None:
        group_id = str(context.group_id)
        url = str(candidate.url).strip()
        file = str(candidate.file).strip()
        local_path = str(local_path).strip()
        sha256 = str(sha256).strip()
        if not group_id or not local_path:
            return None

        now = int(time.time())
        tags_json = json.dumps(_compact_string_list(candidate.tags, limit=12), ensure_ascii=False)
        with self._connect() as conn:
            existing = self._find_sticker_asset(conn, group_id, sha256=sha256, url=url)
            preserve_existing_file = False
            if existing is None:
                existing = self._find_similar_sticker_asset(conn, group_id, candidate)
                preserve_existing_file = existing is not None
            if existing is not None:
                asset_id = int(existing["id"])
                next_local_path = "" if preserve_existing_file else local_path
                next_sha256 = "" if preserve_existing_file else sha256
                conn.execute(
                    """
                    UPDATE sticker_assets
                    SET source_user_id = ?,
                        source_message_id = ?,
                        url = CASE WHEN ? != '' THEN ? ELSE url END,
                        file = CASE WHEN ? != '' THEN ? ELSE file END,
                        local_path = CASE WHEN ? != '' THEN ? ELSE local_path END,
                        sha256 = CASE WHEN ? != '' THEN ? ELSE sha256 END,
                        description = ?,
                        ocr_text = ?,
                        mood = ?,
                        usage = ?,
                        tags = ?,
                        confidence = MAX(confidence, ?),
                        updated_at = ?,
                        last_seen_at = ?,
                        hit_count = hit_count + 1
                    WHERE id = ?
                    """,
                    (
                        context.user_id,
                        context.message_id,
                        url,
                        url,
                        file,
                        file,
                        next_local_path,
                        next_local_path,
                        next_sha256,
                        next_sha256,
                        candidate.description[:1000],
                        candidate.ocr_text[:1000],
                        candidate.mood[:80],
                        candidate.usage[:500],
                        tags_json,
                        float(candidate.confidence),
                        now,
                        now,
                        asset_id,
                    ),
                )
            else:
                cursor = conn.execute(
                    """
                    INSERT INTO sticker_assets (
                        group_id, source_user_id, source_message_id, url, file,
                        local_path, sha256, description, ocr_text, mood, usage,
                        tags, confidence, enabled, created_at, updated_at,
                        last_seen_at, hit_count
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?, ?, 1)
                    """,
                    (
                        group_id,
                        context.user_id,
                        context.message_id,
                        url,
                        file,
                        local_path,
                        sha256,
                        candidate.description[:1000],
                        candidate.ocr_text[:1000],
                        candidate.mood[:80],
                        candidate.usage[:500],
                        tags_json,
                        float(candidate.confidence),
                        now,
                        now,
                        now,
                    ),
                )
                asset_id = int(cursor.lastrowid)
            row = conn.execute(
                """
                SELECT id, group_id, source_user_id, source_message_id, url, file,
                       local_path, sha256, description, ocr_text, mood, usage,
                       tags, confidence, enabled, created_at, updated_at,
                       last_seen_at, hit_count, send_count, last_sent_at
                FROM sticker_assets
                WHERE id = ?
                """,
                (asset_id,),
            ).fetchone()
        return _sticker_asset_record(row) if row is not None else None

    def find_existing_sticker_asset(
        self,
        group_id: str,
        candidate: StickerCandidate,
    ) -> StickerAssetRecord | None:
        group_id = str(group_id)
        if not group_id:
            return None
        url = str(candidate.url).strip()
        with self._connect() as conn:
            row = self._find_sticker_asset(conn, group_id, url=url)
            if row is None:
                row = self._find_similar_sticker_asset(conn, group_id, candidate)
            if row is None:
                return None
            asset_row = conn.execute(
                """
                SELECT id, group_id, source_user_id, source_message_id, url, file,
                       local_path, sha256, description, ocr_text, mood, usage,
                       tags, confidence, enabled, created_at, updated_at,
                       last_seen_at, hit_count, send_count, last_sent_at
                FROM sticker_assets
                WHERE id = ?
                """,
                (int(row["id"]),),
            ).fetchone()
        return _sticker_asset_record(asset_row) if asset_row is not None else None

    def list_sticker_assets(
        self,
        group_id: str,
        limit: int = 24,
        enabled_only: bool = True,
    ) -> list[StickerAssetRecord]:
        where = ["group_id = ?", "local_path != ''"]
        params: list[object] = [str(group_id)]
        if enabled_only:
            where.append("enabled = 1")
        params.append(int(limit))
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT id, group_id, source_user_id, source_message_id, url, file,
                       local_path, sha256, description, ocr_text, mood, usage,
                       tags, confidence, enabled, created_at, updated_at,
                       last_seen_at, hit_count, send_count, last_sent_at
                FROM sticker_assets
                WHERE {' AND '.join(where)}
                ORDER BY confidence DESC, hit_count DESC, updated_at DESC, id DESC
                LIMIT ?
                """,
                params,
            ).fetchall()
        return [_sticker_asset_record(row) for row in rows]

    def list_stickers(self, group_id: str, limit: int = 20) -> list[str]:
        assets = self.list_sticker_assets(group_id, limit=limit, enabled_only=False)
        return [format_sticker_asset(asset) for asset in assets]

    def get_sticker_asset(self, sticker_id: int) -> StickerAssetRecord | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT id, group_id, source_user_id, source_message_id, url, file,
                       local_path, sha256, description, ocr_text, mood, usage,
                       tags, confidence, enabled, created_at, updated_at,
                       last_seen_at, hit_count, send_count, last_sent_at
                FROM sticker_assets
                WHERE id = ?
                """,
                (int(sticker_id),),
            ).fetchone()
        return _sticker_asset_record(row) if row is not None else None

    def set_sticker_enabled(self, sticker_id: int, enabled: bool) -> bool:
        with self._connect() as conn:
            cursor = conn.execute(
                """
                UPDATE sticker_assets
                SET enabled = ?, updated_at = ?
                WHERE id = ?
                """,
                (1 if enabled else 0, int(time.time()), int(sticker_id)),
            )
        return cursor.rowcount > 0

    def delete_sticker_asset(self, sticker_id: int) -> StickerAssetRecord | None:
        asset = self.get_sticker_asset(sticker_id)
        if asset is None:
            return None
        with self._connect() as conn:
            conn.execute("DELETE FROM sticker_assets WHERE id = ?", (int(sticker_id),))
            conn.execute("DELETE FROM sticker_usage_daily WHERE sticker_id = ?", (int(sticker_id),))
        return asset

    def record_sticker_sent(
        self,
        sticker_id: int,
        usage_date: str = "",
        sent_at: int | None = None,
    ) -> None:
        now = int(sent_at or time.time())
        usage_day = str(usage_date).strip() or _local_usage_date(now)
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE sticker_assets
                SET last_seen_at = ?,
                    hit_count = hit_count + 1,
                    send_count = send_count + 1,
                    last_sent_at = ?
                WHERE id = ?
                """,
                (now, now, int(sticker_id)),
            )
            row = conn.execute(
                """
                SELECT group_id
                FROM sticker_assets
                WHERE id = ?
                """,
                (int(sticker_id),),
            ).fetchone()
            if row is None:
                return
            conn.execute(
                """
                INSERT INTO sticker_usage_daily (
                    sticker_id, group_id, usage_date, send_count, first_sent_at, last_sent_at
                )
                VALUES (?, ?, ?, 1, ?, ?)
                ON CONFLICT(sticker_id, usage_date) DO UPDATE SET
                    group_id = excluded.group_id,
                    send_count = sticker_usage_daily.send_count + 1,
                    last_sent_at = excluded.last_sent_at
                """,
                (int(sticker_id), str(row["group_id"]), usage_day, now, now),
            )

    def count_sticker_usage(self, sticker_id: int, usage_date: str = "") -> int:
        where = ["sticker_id = ?"]
        params: list[object] = [int(sticker_id)]
        if usage_date:
            where.append("usage_date = ?")
            params.append(str(usage_date))
        with self._connect() as conn:
            row = conn.execute(
                f"""
                SELECT COALESCE(SUM(send_count), 0) AS count
                FROM sticker_usage_daily
                WHERE {' AND '.join(where)}
                """,
                params,
            ).fetchone()
        return int(row["count"]) if row is not None else 0

    def list_sticker_usage_daily(
        self,
        group_id: str = "",
        usage_date: str = "",
        limit: int = 200,
    ) -> list[dict[str, object]]:
        where: list[str] = []
        params: list[object] = []
        if group_id:
            where.append("group_id = ?")
            params.append(str(group_id))
        if usage_date:
            where.append("usage_date = ?")
            params.append(str(usage_date))
        params.append(int(limit))
        where_sql = f"WHERE {' AND '.join(where)}" if where else ""
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT sticker_id, group_id, usage_date, send_count, first_sent_at, last_sent_at
                FROM sticker_usage_daily
                {where_sql}
                ORDER BY usage_date DESC, send_count DESC, last_sent_at DESC
                LIMIT ?
                """,
                params,
            ).fetchall()
        return [
            {
                "sticker_id": int(row["sticker_id"]),
                "group_id": str(row["group_id"]),
                "usage_date": str(row["usage_date"]),
                "send_count": int(row["send_count"] or 0),
                "first_sent_at": int(row["first_sent_at"] or 0),
                "last_sent_at": int(row["last_sent_at"] or 0),
            }
            for row in rows
        ]

    def claim_sticker_cleanup(self, interval_seconds: int, now: int | None = None) -> bool:
        timestamp = int(now or time.time())
        interval = max(1, int(interval_seconds))
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT value
                FROM bot_maintenance_state
                WHERE key = 'sticker_cleanup.last_run_at'
                """
            ).fetchone()
            last_run_at = _safe_int(str(row["value"] or "0"), 0) if row is not None else 0
            if row is not None and timestamp - last_run_at < interval:
                return False
            conn.execute(
                """
                INSERT INTO bot_maintenance_state (key, value, updated_at)
                VALUES ('sticker_cleanup.last_run_at', ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    value = excluded.value,
                    updated_at = excluded.updated_at
                """,
                (str(timestamp), timestamp),
            )
        return True

    def delete_unused_sticker_assets(
        self,
        unused_seconds: int,
        now: int | None = None,
    ) -> list[StickerAssetRecord]:
        timestamp = int(now or time.time())
        cutoff = timestamp - max(1, int(unused_seconds))
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, group_id, source_user_id, source_message_id, url, file,
                       local_path, sha256, description, ocr_text, mood, usage,
                       tags, confidence, enabled, created_at, updated_at,
                       last_seen_at, hit_count, send_count, last_sent_at
                FROM sticker_assets
                WHERE local_path != ''
                  AND (
                    (last_sent_at > 0 AND last_sent_at <= ?)
                    OR (last_sent_at = 0 AND created_at <= ?)
                  )
                ORDER BY COALESCE(NULLIF(last_sent_at, 0), created_at) ASC, id ASC
                """,
                (cutoff, cutoff),
            ).fetchall()
            assets = [_sticker_asset_record(row) for row in rows]
            if assets:
                ids = [asset.id for asset in assets]
                placeholders = ", ".join("?" for _ in ids)
                conn.execute(f"DELETE FROM sticker_assets WHERE id IN ({placeholders})", ids)
                conn.execute(f"DELETE FROM sticker_usage_daily WHERE sticker_id IN ({placeholders})", ids)
        return assets

    def _find_sticker_asset(
        self,
        conn: sqlite3.Connection,
        group_id: str,
        sha256: str = "",
        url: str = "",
    ) -> sqlite3.Row | None:
        if sha256:
            row = conn.execute(
                """
                SELECT id
                FROM sticker_assets
                WHERE group_id = ? AND sha256 = ?
                LIMIT 1
                """,
                (str(group_id), str(sha256)),
            ).fetchone()
            if row is not None:
                return row
        if url:
            return conn.execute(
                """
                SELECT id
                FROM sticker_assets
                WHERE group_id = ? AND url = ?
                LIMIT 1
                """,
                (str(group_id), str(url)),
            ).fetchone()
        return None

    def _find_similar_sticker_asset(
        self,
        conn: sqlite3.Connection,
        group_id: str,
        candidate: StickerCandidate,
    ) -> sqlite3.Row | None:
        ocr_key = _sticker_text_key(candidate.ocr_text)
        if not _useful_sticker_text_key(ocr_key):
            return None
        rows = conn.execute(
            """
            SELECT id, ocr_text
            FROM sticker_assets
            WHERE group_id = ?
              AND ocr_text != ''
            ORDER BY updated_at DESC, id DESC
            LIMIT 300
            """,
            (str(group_id),),
        ).fetchall()
        for row in rows:
            if _sticker_text_key(str(row["ocr_text"] or "")) == ocr_key:
                return row
        return None

    def record_memories(self, memories: Iterable[MemoryCandidate]) -> None:
        self.record_memory_candidates(memories)

    def record_memory_candidates(
        self,
        memories: Iterable[MemoryCandidate],
        confidence_threshold: float = 0.75,
    ) -> MemoryWriteSet:
        accepted: list[MemoryCandidate] = []
        pending: list[MemoryCandidate] = []
        conflicts: list[MemoryCandidate] = []
        rejected: list[MemoryCandidate] = []
        now = int(time.time())

        with self._connect() as conn:
            for item in memories:
                normalized = self._normalize_memory_candidate(item)
                acceptance_status = self._acceptance_status(conn, normalized, confidence_threshold)
                if acceptance_status == "rejected":
                    rejected.append(replace(normalized, status="rejected", verification_status="rejected"))
                    continue

                if acceptance_status == "pending_confirmation":
                    pending_item = replace(
                        normalized,
                        status="pending_confirmation",
                        verification_status="pending_confirmation",
                    )
                    self._insert_memory(conn, pending_item, now)
                    pending.append(pending_item)
                    continue

                duplicate = self._find_duplicate_memory(conn, normalized)
                if duplicate:
                    conn.execute(
                        """
                        UPDATE memory_items
                        SET confidence = MAX(confidence, ?),
                            importance = MAX(importance, ?),
                            source_text = CASE WHEN source_text = '' THEN ? ELSE source_text END,
                            updated_at = ?,
                            last_seen_at = ?
                        WHERE id = ?
                        """,
                        (
                            normalized.confidence,
                            normalized.importance,
                            normalized.source_text,
                            now,
                            now,
                            duplicate.id,
                        ),
                    )
                    accepted.append(replace(normalized, status="active", verification_status="accepted"))
                    continue

                conflict = self._find_conflicting_memory(conn, normalized)
                if conflict:
                    conflict_item = replace(
                        normalized,
                        status="conflict",
                        verification_status="conflict",
                        conflict_of=conflict.id,
                    )
                    self._insert_memory(conn, conflict_item, now)
                    conflicts.append(conflict_item)
                    continue

                active_item = replace(normalized, status="active", verification_status="accepted")
                self._insert_memory(conn, active_item, now)
                accepted.append(active_item)

        return MemoryWriteSet(
            accepted=accepted,
            pending=pending,
            conflicts=conflicts,
            rejected=rejected,
        )

    def record_fact_candidates(self, facts: Iterable[FactCandidate]) -> FactWriteSet:
        accepted: list[FactRecord] = []
        pending: list[FactRecord] = []
        rejected: list[FactCandidate] = []
        now = int(time.time())

        with self._connect() as conn:
            for item in facts:
                normalized = self._normalize_fact_candidate(item)
                acceptance_status = self._fact_acceptance_status(conn, normalized)
                if acceptance_status == "rejected":
                    rejected.append(replace(normalized, status="rejected"))
                    continue

                conflicting_facts = self._find_conflicting_facts(conn, normalized)
                if conflicting_facts and normalized.claim_scope != "self_report":
                    acceptance_status = "pending_confirmation"

                duplicate = self._find_duplicate_fact(conn, normalized, acceptance_status)
                if duplicate:
                    conn.execute(
                        """
                        UPDATE member_facts
                        SET confidence = MAX(confidence, ?),
                            importance = MAX(importance, ?),
                            updated_at = ?,
                            last_seen_at = ?
                        WHERE id = ?
                        """,
                        (normalized.confidence, normalized.importance, now, now, duplicate.id),
                    )
                    updated = replace(
                        duplicate,
                        confidence=max(duplicate.confidence, normalized.confidence),
                        importance=max(duplicate.importance, normalized.importance),
                        updated_at=now,
                        last_seen_at=now,
                    )
                    if updated.status == "accepted":
                        self._sync_aliases_for_fact(conn, updated)
                    if updated.status == "accepted":
                        accepted.append(updated)
                    elif updated.status == "pending_confirmation":
                        pending.append(updated)
                    continue

                inserted = self._insert_fact(
                    conn,
                    replace(
                        normalized,
                        status="accepted" if acceptance_status == "accepted" else "pending_confirmation",
                    ),
                    now,
                )
                if inserted.status == "accepted":
                    if conflicting_facts and normalized.claim_scope == "self_report":
                        self._supersede_facts(conn, conflicting_facts, inserted.id, now)
                    self._sync_aliases_for_fact(conn, inserted)
                    accepted.append(inserted)
                elif inserted.status == "pending_confirmation":
                    pending.append(inserted)

        return FactWriteSet(accepted=accepted, pending=pending, rejected=rejected)

    def list_user_facts(
        self,
        user_id: str,
        limit: int = 20,
        status: str = "accepted",
        group_id: str = "",
        include_faded: bool = False,
    ) -> list[FactRecord]:
        subject = _dashboard_user_id(user_id)
        if not subject:
            return []
        where = ["subject_user_id = ?", "status = ?"]
        params: list[object] = [subject, _safe_fact_status(status)]
        if group_id:
            where.append("source_group_id = ?")
            params.append(str(group_id))
        if status == "accepted" and not include_faded:
            cutoff = int(time.time()) - self.fact_context_ttl_seconds
            protected = ", ".join("?" for _ in PROTECTED_FACT_TYPES)
            where.append(
                f"(importance >= ? OR fact_type IN ({protected}) OR last_seen_at >= ?)"
            )
            params.extend([self.low_importance_threshold, *sorted(PROTECTED_FACT_TYPES), cutoff])
        limit_value = int(limit)
        limit_sql = "LIMIT ?" if limit_value > 0 else ""
        if limit_value > 0:
            params.append(limit_value)
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT id, subject_user_id, fact_type, claim_text, topic, stance,
                       confidence, status, claim_scope, source_user_id, source_group_id,
                       evidence_message_id, evidence_text, created_at, updated_at,
                       importance, last_seen_at, superseded_by_fact_id, forget_reason
                FROM member_facts
                WHERE {' AND '.join(where)}
                ORDER BY
                    CASE WHEN fact_type IN ('identity', 'alias', 'boundary') THEN 0 ELSE 1 END,
                    importance DESC, updated_at DESC, id DESC
                {limit_sql}
                """,
                params,
            ).fetchall()
        records = [_fact_record(row) for row in rows]
        if status == "accepted":
            records = [record for record in records if not _is_unreasonable_alias_fact(record)]
        return records

    def list_user_facts_text(self, user_id: str, limit: int = 20, status: str = "accepted") -> list[str]:
        return [format_fact_record(record) for record in self.list_user_facts(user_id, limit, status)]

    def list_pending_facts(self, limit: int = 20) -> list[FactRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, subject_user_id, fact_type, claim_text, topic, stance,
                       confidence, status, claim_scope, source_user_id, source_group_id,
                       evidence_message_id, evidence_text, created_at, updated_at,
                       importance, last_seen_at, superseded_by_fact_id, forget_reason
                FROM member_facts
                WHERE status = 'pending_confirmation'
                ORDER BY updated_at DESC, id DESC
                LIMIT ?
                """,
                (int(limit),),
            ).fetchall()
        return [_fact_record(row) for row in rows]

    def get_fact_record(self, fact_id: int) -> FactRecord | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT id, subject_user_id, fact_type, claim_text, topic, stance,
                       confidence, status, claim_scope, source_user_id, source_group_id,
                       evidence_message_id, evidence_text, created_at, updated_at,
                       importance, last_seen_at, superseded_by_fact_id, forget_reason
                FROM member_facts
                WHERE id = ?
                """,
                (int(fact_id),),
            ).fetchone()
        return _fact_record(row) if row else None

    def approve_fact(self, fact_id: int) -> FactRecord | None:
        now = int(time.time())
        with self._connect() as conn:
            cursor = conn.execute(
                """
                UPDATE member_facts
                SET status = 'accepted', updated_at = ?, last_seen_at = ?
                WHERE id = ? AND status = 'pending_confirmation'
                """,
                (now, now, int(fact_id)),
            )
            if cursor.rowcount <= 0:
                return None
            row = conn.execute(
                """
                SELECT id, subject_user_id, fact_type, claim_text, topic, stance,
                       confidence, status, claim_scope, source_user_id, source_group_id,
                       evidence_message_id, evidence_text, created_at, updated_at,
                       importance, last_seen_at, superseded_by_fact_id, forget_reason
                FROM member_facts
                WHERE id = ?
                """,
                (int(fact_id),),
            ).fetchone()
            record = _fact_record(row) if row else None
            if record is not None:
                self._sync_aliases_for_fact(conn, record)
        return record

    def approve_user_pending_fact(self, user_id: str, fact_id: int) -> FactRecord | None:
        subject = _dashboard_user_id(user_id)
        if not subject:
            return None
        now = int(time.time())
        with self._connect() as conn:
            cursor = conn.execute(
                """
                UPDATE member_facts
                SET status = 'accepted', updated_at = ?, last_seen_at = ?
                WHERE id = ? AND subject_user_id = ? AND status = 'pending_confirmation'
                """,
                (now, now, int(fact_id), subject),
            )
            if cursor.rowcount <= 0:
                return None
            row = conn.execute(
                """
                SELECT id, subject_user_id, fact_type, claim_text, topic, stance,
                       confidence, status, claim_scope, source_user_id, source_group_id,
                       evidence_message_id, evidence_text, created_at, updated_at,
                       importance, last_seen_at, superseded_by_fact_id, forget_reason
                FROM member_facts
                WHERE id = ?
                """,
                (int(fact_id),),
            ).fetchone()
            record = _fact_record(row) if row else None
            if record is not None:
                self._sync_aliases_for_fact(conn, record)
        return record

    def reject_fact(self, fact_id: int) -> bool:
        with self._connect() as conn:
            cursor = conn.execute(
                """
                UPDATE member_facts
                SET status = 'rejected', updated_at = ?
                WHERE id = ? AND status = 'pending_confirmation'
                """,
                (int(time.time()), int(fact_id)),
            )
        return cursor.rowcount > 0

    def reject_user_pending_fact(self, user_id: str, fact_id: int) -> bool:
        subject = _dashboard_user_id(user_id)
        if not subject:
            return False
        with self._connect() as conn:
            cursor = conn.execute(
                """
                UPDATE member_facts
                SET status = 'rejected', updated_at = ?
                WHERE id = ? AND subject_user_id = ? AND status = 'pending_confirmation'
                """,
                (int(time.time()), int(fact_id), subject),
            )
        return cursor.rowcount > 0

    def forget_fact(self, fact_id: int, reason: str = "manual") -> FactRecord | None:
        now = int(time.time())
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT id, subject_user_id, fact_type, claim_text, topic, stance,
                       confidence, status, claim_scope, source_user_id, source_group_id,
                       evidence_message_id, evidence_text, created_at, updated_at,
                       importance, last_seen_at, superseded_by_fact_id, forget_reason
                FROM member_facts
                WHERE id = ?
                """,
                (int(fact_id),),
            ).fetchone()
            if row is None:
                return None
            record = _fact_record(row)
            cursor = conn.execute(
                """
                UPDATE member_facts
                SET status = 'forgotten',
                    forget_reason = ?,
                    updated_at = ?
                WHERE id = ?
                  AND status IN ('accepted', 'pending_confirmation', 'superseded')
                """,
                (_clean_fact_field(reason, 120) or "manual", now, int(fact_id)),
            )
            if cursor.rowcount <= 0:
                return None
            conn.execute(
                """
                UPDATE member_aliases
                SET status = 'forgotten',
                    updated_at = ?
                WHERE source_fact_id = ?
                  AND status = 'active'
                """,
                (now, int(fact_id)),
            )
        return replace(record, status="forgotten", updated_at=now, forget_reason=reason or "manual")

    def get_user_profile(self, user_id: str) -> UserProfileRecord | None:
        subject = _dashboard_user_id(user_id)
        if not subject:
            return None
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT user_id, summary, traits_json, supporting_fact_ids,
                       fact_count, version, updated_at
                FROM member_profiles
                WHERE user_id = ?
                """,
                (subject,),
            ).fetchone()
        return _user_profile_record(row) if row else None

    def should_update_user_profile(self, user_id: str, threshold: int | None = None) -> bool:
        subject = _dashboard_user_id(user_id)
        if not subject or not subject.isdigit():
            return False
        threshold = max(1, int(threshold or self.profile_fact_threshold))
        with self._connect() as conn:
            accepted_count = int(
                conn.execute(
                    """
                    SELECT COUNT(*) AS count
                    FROM member_facts
                    WHERE subject_user_id = ? AND status = 'accepted'
                    """,
                    (subject,),
                ).fetchone()["count"]
            )
            row = conn.execute(
                "SELECT fact_count FROM member_profiles WHERE user_id = ?",
                (subject,),
            ).fetchone()
        profiled_count = int(row["fact_count"]) if row else 0
        return accepted_count - profiled_count >= threshold

    def maybe_update_user_profile(
        self,
        user_id: str,
        draft: UserProfileDraft,
        facts: list[FactRecord],
        force: bool = False,
    ) -> UserProfileRecord | None:
        subject = _dashboard_user_id(user_id)
        if not subject or not draft.summary.strip():
            return None
        if not force and not self.should_update_user_profile(subject):
            return None
        accepted_facts = [fact for fact in facts if fact.status == "accepted"]
        fact_count = len(accepted_facts)
        supporting_ids = draft.supporting_fact_ids or tuple(fact.id for fact in accepted_facts[-20:])
        now = int(time.time())
        with self._connect() as conn:
            current = conn.execute(
                "SELECT version FROM member_profiles WHERE user_id = ?",
                (subject,),
            ).fetchone()
            version = int(current["version"]) + 1 if current else 1
            conn.execute(
                """
                INSERT INTO member_profiles (
                    user_id, summary, traits_json, supporting_fact_ids,
                    fact_count, version, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(user_id) DO UPDATE SET
                    summary = excluded.summary,
                    traits_json = excluded.traits_json,
                    supporting_fact_ids = excluded.supporting_fact_ids,
                    fact_count = excluded.fact_count,
                    version = excluded.version,
                    updated_at = excluded.updated_at
                """,
                (
                    subject,
                    " ".join(draft.summary.split())[:500],
                    json.dumps(draft.traits, ensure_ascii=False),
                    json.dumps(_compact_int_list(supporting_ids, limit=80), ensure_ascii=False),
                    fact_count,
                    version,
                    now,
                ),
            )
        return UserProfileRecord(
            user_id=subject,
            summary=" ".join(draft.summary.split())[:500],
            traits=draft.traits,
            supporting_fact_ids=tuple(_compact_int_list(supporting_ids, limit=80)),
            fact_count=fact_count,
            version=version,
            updated_at=now,
        )

    def clear_user_profile(self, user_id: str) -> bool:
        subject = _dashboard_user_id(user_id)
        if not subject:
            return False
        with self._connect() as conn:
            cursor = conn.execute("DELETE FROM member_profiles WHERE user_id = ?", (subject,))
        return cursor.rowcount > 0

    def format_user_profile(self, user_id: str) -> str:
        profile = self.get_user_profile(user_id)
        facts = self.list_user_facts(user_id, limit=8)
        if profile is None and not facts:
            return "暂无该用户画像。"
        lines = []
        if profile is not None:
            lines.append(
                f"QQ {profile.user_id} profile v{profile.version} "
                f"(facts={profile.fact_count})\n{profile.summary}"
            )
            if profile.traits:
                lines.append(json.dumps(profile.traits, ensure_ascii=False))
        if facts:
            lines.append("近期 FACT：")
            lines.extend(format_fact_record(record) for record in facts)
        return "\n".join(lines)

    def list_memories(
        self,
        owner_type: str,
        owner_id: str,
        limit: int = 8,
        status: str = "active",
    ) -> list[MemoryRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, owner_type, owner_id, kind, content, confidence, importance, status,
                       updated_at, source_user_id, source_group_id, subject_user_id,
                       claim_scope, verification_status
                FROM memory_items
                WHERE owner_type = ? AND owner_id = ? AND status = ?
                ORDER BY importance DESC, updated_at DESC, id DESC
                LIMIT ?
                """,
                (str(owner_type), str(owner_id), str(status), limit),
            ).fetchall()
        return [_memory_record(row) for row in rows]

    def list_user_memories(self, user_id: str, limit: int = 8) -> list[str]:
        return [format_memory_record(record) for record in self.list_memories("user", user_id, limit)]

    def list_self_memories(self, status: str = "active", limit: int = 16) -> list[str]:
        return [
            format_memory_record(record)
            for record in self.list_memories("self", "bot", limit=limit, status=status)
        ]

    def list_memories_by_status(self, status: str, limit: int = 12) -> list[str]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, owner_type, owner_id, kind, content, confidence, importance, status,
                       updated_at, source_user_id, source_group_id, subject_user_id,
                       claim_scope, verification_status
                FROM memory_items
                WHERE status = ?
                ORDER BY updated_at DESC, id DESC
                LIMIT ?
                """,
                (str(status), limit),
            ).fetchall()
        return [format_memory_record(_memory_record(row)) for row in rows]

    def forget_memory(self, memory_id: int) -> bool:
        with self._connect() as conn:
            cursor = conn.execute(
                """
                UPDATE memory_items
                SET status = 'forgotten', updated_at = ?
                WHERE id = ? AND status != 'forgotten'
                """,
                (int(time.time()), int(memory_id)),
            )
        return cursor.rowcount > 0

    def approve_memory(self, memory_id: int) -> bool:
        now = int(time.time())
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT conflict_of
                FROM memory_items
                WHERE id = ? AND status IN ('pending_confirmation', 'conflict')
                """,
                (int(memory_id),),
            ).fetchone()
            if row is None:
                return False

            conflict_of = row["conflict_of"]
            if conflict_of:
                conn.execute(
                    """
                    UPDATE memory_items
                    SET status = 'forgotten',
                        verification_status = 'rejected',
                        updated_at = ?
                    WHERE id = ? AND status = 'active'
                    """,
                    (now, int(conflict_of)),
                )

            cursor = conn.execute(
                """
                UPDATE memory_items
                SET status = 'active',
                    verification_status = 'accepted',
                    conflict_of = NULL,
                    updated_at = ?,
                    last_seen_at = ?
                WHERE id = ? AND status IN ('pending_confirmation', 'conflict')
                """,
                (now, now, int(memory_id)),
            )
        return cursor.rowcount > 0

    def reject_memory(self, memory_id: int) -> bool:
        with self._connect() as conn:
            cursor = conn.execute(
                """
                UPDATE memory_items
                SET status = 'rejected',
                    verification_status = 'rejected',
                    updated_at = ?
                WHERE id = ? AND status IN ('pending_confirmation', 'conflict')
                """,
                (int(time.time()), int(memory_id)),
            )
        return cursor.rowcount > 0

    def get_relationship(self, group_id: str, user_id: str) -> RelationshipState:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT group_id, user_id, closeness, trust, familiarity, tension, summary
                FROM relationships
                WHERE group_id = ? AND user_id = ?
                """,
                (str(group_id), str(user_id)),
            ).fetchone()
        if row is None:
            return RelationshipState(group_id=str(group_id), user_id=str(user_id))
        return RelationshipState(
            group_id=str(row["group_id"]),
            user_id=str(row["user_id"]),
            closeness=int(row["closeness"]),
            trust=int(row["trust"]),
            familiarity=int(row["familiarity"]),
            tension=int(row["tension"]),
            summary=merge_relationship_summary("", str(row["summary"] or "")),
        )

    def apply_relationship_delta(self, group_id: str, user_id: str, delta: RelationDelta) -> RelationshipState:
        current = self.get_relationship(group_id, user_id)
        updated = RelationshipState(
            group_id=str(group_id),
            user_id=str(user_id),
            closeness=_clamp_score(current.closeness + delta.closeness),
            trust=_clamp_score(current.trust + delta.trust),
            familiarity=_clamp_score(current.familiarity + delta.familiarity),
            tension=_clamp_score(current.tension + delta.tension),
            summary=_merge_summary(current.summary, delta.summary_patch),
        )
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO relationships (
                    group_id, user_id, closeness, trust, familiarity, tension, summary, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(group_id, user_id) DO UPDATE SET
                    closeness = excluded.closeness,
                    trust = excluded.trust,
                    familiarity = excluded.familiarity,
                    tension = excluded.tension,
                    summary = excluded.summary,
                    updated_at = excluded.updated_at
                """,
                (
                    updated.group_id,
                    updated.user_id,
                    updated.closeness,
                    updated.trust,
                    updated.familiarity,
                    updated.tension,
                    updated.summary,
                    int(time.time()),
                ),
            )
        return updated

    def touch_relationship(self, group_id: str, user_id: str, familiarity_delta: int = 1) -> None:
        self.apply_relationship_delta(
            group_id,
            user_id,
            RelationDelta(familiarity=familiarity_delta, reason="message observed"),
        )

    def format_relationship(self, group_id: str, user_id: str) -> str:
        relation = self.get_relationship(group_id, user_id)
        return (
            f"QQ={user_id}\n"
            f"closeness={relation.closeness}\n"
            f"trust={relation.trust}\n"
            f"familiarity={relation.familiarity}\n"
            f"tension={relation.tension}\n"
            f"summary={relation.summary or '(empty)'}"
        )

    def format_relationship_ranking(self, group_id: str, limit: int = 5) -> str:
        limit = max(1, min(50, int(limit)))
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT group_id, user_id, closeness, trust, familiarity, tension, updated_at,
                       (closeness + familiarity) AS relationship_score
                FROM relationships
                WHERE group_id = ?
                ORDER BY relationship_score DESC,
                         closeness DESC,
                         familiarity DESC,
                         trust DESC,
                         updated_at DESC
                LIMIT ?
                """,
                (str(group_id), limit),
            ).fetchall()
            profiles = {
                str(row["user_id"]): self._dashboard_user_profile(conn, str(row["user_id"]))
                for row in rows
            }

        if not rows:
            return "本群暂无关系记录。"

        lines = [f"本群亲密/了解程度 TOP {limit}（按 亲密+了解 排序）："]
        for index, row in enumerate(rows, start=1):
            user_id = str(row["user_id"])
            closeness = int(row["closeness"])
            familiarity = int(row["familiarity"])
            trust = int(row["trust"])
            tension = int(row["tension"])
            score = int(row["relationship_score"])
            label = _relationship_rank_label(user_id, profiles.get(user_id))
            lines.append(
                f"{index}. {label} "
                f"亲密={closeness} 了解={familiarity} 信任={trust} 紧张={tension} 综合={score}"
            )
        return "\n".join(lines)

    def get_recent_messages(self, group_id: str, limit: int = 12) -> list[str]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT sender_name, user_id, plain_text
                FROM messages
                WHERE group_id = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (str(group_id), limit),
            ).fetchall()
        lines = []
        for row in reversed(rows):
            line = _format_message_context_line(row)
            if line:
                lines.append(line)
        return lines

    def get_focused_recent_messages(
        self,
        group_id: str,
        user_id: str,
        speaker_limit: int = 5,
        other_limit: int = 10,
    ) -> tuple[list[str], list[str]]:
        group_id = str(group_id)
        user_id = str(user_id)
        with self._connect() as conn:
            speaker_rows = conn.execute(
                """
                SELECT sender_name, user_id, plain_text
                FROM messages
                WHERE group_id = ?
                  AND user_id = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (group_id, user_id, int(speaker_limit)),
            ).fetchall()
            other_rows = conn.execute(
                """
                SELECT sender_name, user_id, plain_text
                FROM messages
                WHERE group_id = ?
                  AND user_id != ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (group_id, user_id, int(other_limit)),
            ).fetchall()

        speaker_lines = [
            line
            for row in reversed(speaker_rows)
            if (line := _format_message_context_line(row))
        ]
        other_lines = [
            line
            for row in reversed(other_rows)
            if (line := _format_message_context_line(row))
        ]
        return speaker_lines, other_lines

    def get_recent_bot_reply_to_user(
        self,
        group_id: str,
        user_id: str,
        window_seconds: int,
    ) -> tuple[str, int]:
        now = int(time.time())
        cutoff = now - max(1, int(window_seconds))
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT time, reply
                FROM bot_decisions
                WHERE group_id = ?
                  AND user_id = ?
                  AND reply != ''
                  AND time >= ?
                ORDER BY id DESC
                LIMIT 1
                """,
                (str(group_id), str(user_id), cutoff),
            ).fetchone()
        if row is None:
            return "", 0
        reply = str(row["reply"] or "").strip()
        if not reply:
            return "", 0
        return reply, max(0, now - int(row["time"]))

    def get_recent_activity_counts(
        self,
        group_id: str,
        human_window_seconds: int = 60,
        bot_window_seconds: int = 120,
    ) -> tuple[int, int]:
        now = int(time.time())
        with self._connect() as conn:
            human_count = int(
                conn.execute(
                    """
                    SELECT COUNT(1) AS count
                    FROM messages
                    WHERE group_id = ?
                      AND time >= ?
                      AND sender_role != 'bot'
                    """,
                    (str(group_id), now - int(human_window_seconds)),
                ).fetchone()["count"]
            )
            bot_count = int(
                conn.execute(
                    """
                    SELECT COUNT(1) AS count
                    FROM messages
                    WHERE group_id = ?
                      AND time >= ?
                      AND sender_role = 'bot'
                    """,
                    (str(group_id), now - int(bot_window_seconds)),
                ).fetchone()["count"]
            )
        return human_count, bot_count

    def get_recent_image_descriptions(self, group_id: str, limit: int = 8) -> list[str]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT sender_name, messages.user_id, message_attachments.summary
                FROM message_attachments
                LEFT JOIN messages
                  ON messages.group_id = message_attachments.group_id
                 AND messages.message_id = message_attachments.message_id
                WHERE message_attachments.group_id = ?
                  AND message_attachments.attachment_type = 'image'
                  AND message_attachments.summary != ''
                ORDER BY message_attachments.id DESC
                LIMIT ?
                """,
                (str(group_id), int(limit)),
            ).fetchall()
        lines = []
        for row in reversed(rows):
            name = str(row["sender_name"] or row["user_id"] or "unknown")
            lines.append(f"{name}: [图片] {row['summary']}")
        return lines

    def count_image_generation_usage(self, user_id: str, usage_date: str) -> int:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT COUNT(1) AS count
                FROM image_generation_usage
                WHERE user_id = ? AND usage_date = ?
                """,
                (str(user_id), str(usage_date)),
            ).fetchone()
        return int(row["count"]) if row is not None else 0

    def record_image_generation_usage(
        self,
        group_id: str,
        user_id: str,
        usage_date: str,
        prompt: str,
        image_ref: str,
        created_at: int | None = None,
    ) -> None:
        now = int(created_at or time.time())
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO image_generation_usage (
                    usage_date, group_id, user_id, prompt, image_ref, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    str(usage_date),
                    str(group_id),
                    str(user_id),
                    str(prompt)[:1000],
                    str(image_ref)[:1000],
                    now,
                ),
            )

    def record_llm_usage(self, record: LLMUsageRecord) -> None:
        created_at = int(record.created_at or time.time())
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO llm_usage (
                    created_at, purpose, model, prompt_chars, completion_chars,
                    prompt_tokens, completion_tokens, total_tokens
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    created_at,
                    str(record.purpose)[:80],
                    str(record.model)[:120],
                    max(0, int(record.prompt_chars)),
                    max(0, int(record.completion_chars)),
                    max(0, int(record.prompt_tokens)),
                    max(0, int(record.completion_tokens)),
                    max(0, int(record.total_tokens)),
                ),
            )

    def list_dashboard_llm_usage(
        self,
        since: int,
        limit: int = 100,
    ) -> dict[str, object]:
        since = max(0, int(since))
        limit = max(1, int(limit))
        with self._connect() as conn:
            summary = conn.execute(
                """
                SELECT
                    COUNT(1) AS calls,
                    COALESCE(SUM(prompt_chars), 0) AS prompt_chars,
                    COALESCE(SUM(completion_chars), 0) AS completion_chars,
                    COALESCE(SUM(prompt_tokens), 0) AS prompt_tokens,
                    COALESCE(SUM(completion_tokens), 0) AS completion_tokens,
                    COALESCE(SUM(total_tokens), 0) AS total_tokens,
                    MIN(created_at) AS first_at,
                    MAX(created_at) AS last_at
                FROM llm_usage
                WHERE created_at >= ?
                """,
                (since,),
            ).fetchone()
            by_purpose_rows = conn.execute(
                """
                SELECT
                    purpose,
                    model,
                    COUNT(1) AS calls,
                    COALESCE(SUM(prompt_chars), 0) AS prompt_chars,
                    COALESCE(SUM(completion_chars), 0) AS completion_chars,
                    COALESCE(SUM(prompt_tokens), 0) AS prompt_tokens,
                    COALESCE(SUM(completion_tokens), 0) AS completion_tokens,
                    COALESCE(SUM(total_tokens), 0) AS total_tokens,
                    MAX(created_at) AS last_at
                FROM llm_usage
                WHERE created_at >= ?
                GROUP BY purpose, model
                ORDER BY total_tokens DESC, calls DESC, purpose ASC
                """,
                (since,),
            ).fetchall()
            recent_rows = conn.execute(
                """
                SELECT id, created_at, purpose, model, prompt_chars, completion_chars,
                       prompt_tokens, completion_tokens, total_tokens
                FROM llm_usage
                WHERE created_at >= ?
                ORDER BY created_at DESC, id DESC
                LIMIT ?
                """,
                (since, limit),
            ).fetchall()
        return {
            "summary": _llm_usage_summary_to_dict(summary, since),
            "by_purpose": [_llm_usage_group_to_dict(row) for row in by_purpose_rows],
            "recent": [_llm_usage_row_to_dict(row) for row in recent_rows],
        }

    def get_dashboard_persona(self) -> dict[str, object]:
        with self._connect() as conn:
            rows = conn.execute("SELECT key, value, updated_at FROM persona_state ORDER BY key").fetchall()
        return {
            "persona_state": [
                {
                    "key": str(row["key"]),
                    "value": str(row["value"]),
                    "updated_at": int(row["updated_at"]),
                }
                for row in rows
            ],
            "self_memories": [
                _memory_to_dict(record)
                for record in self.list_memories("self", "bot", limit=100, status="active")
            ],
            "pending_self_memories": [
                _memory_to_dict(record)
                for record in self.list_memories("self", "bot", limit=100, status="pending_confirmation")
            ],
            "conflict_self_memories": [
                _memory_to_dict(record)
                for record in self.list_memories("self", "bot", limit=100, status="conflict")
            ],
        }

    def list_dashboard_groups(self) -> list[str]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT group_id FROM group_whitelist
                UNION
                SELECT group_id FROM messages
                UNION
                SELECT group_id FROM relationships
                UNION
                SELECT source_group_id AS group_id FROM memory_items WHERE source_group_id != ''
                UNION
                SELECT source_group_id AS group_id FROM member_facts WHERE source_group_id != ''
                UNION
                SELECT group_id FROM sticker_assets
                ORDER BY group_id
                """
            ).fetchall()
        return [str(row["group_id"]) for row in rows if str(row["group_id"] or "").strip()]

    def list_dashboard_user_cognition(
        self,
        group_id: str = "",
        user_id: str = "",
        limit: int = 100,
    ) -> list[dict[str, object]]:
        requested_group_id = str(group_id).strip()
        requested_user_id = str(user_id).strip()
        limit = max(1, int(limit))
        query_limit = max(limit * 4, limit)
        candidates: dict[str, dict[str, object]] = {}

        def candidate_for(raw_user_id: str) -> dict[str, object]:
            key = _dashboard_user_id(raw_user_id)
            if key not in candidates:
                candidates[key] = {
                    "user_id": key,
                    "group_ids": set(),
                    "sort_at": 0,
                }
            return candidates[key]

        user_variants = _dashboard_user_id_variants(requested_user_id) if requested_user_id else []
        user_filter_sql = ""
        user_filter_params: list[object] = []
        if user_variants:
            placeholders = ", ".join("?" for _ in user_variants)
            user_filter_sql = f" AND user_id IN ({placeholders})"
            user_filter_params.extend(user_variants)

        fact_user_filter_sql = ""
        fact_user_filter_params: list[object] = []
        if user_variants:
            placeholders = ", ".join("?" for _ in user_variants)
            fact_user_filter_sql = f" AND subject_user_id IN ({placeholders})"
            fact_user_filter_params.extend(user_variants)

        relationship_where = "WHERE 1 = 1"
        relationship_params: list[object] = []
        if requested_group_id:
            relationship_where += " AND group_id = ?"
            relationship_params.append(requested_group_id)
        relationship_where += user_filter_sql
        relationship_params.extend(user_filter_params)

        with self._connect() as conn:
            relation_rows = conn.execute(
                f"""
                SELECT group_id, user_id, closeness, trust, familiarity, tension, summary, updated_at
                FROM relationships
                {relationship_where}
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                [*relationship_params, query_limit],
            ).fetchall()

            fact_where = ["status = 'accepted'"]
            fact_params: list[object] = []
            if requested_group_id:
                fact_where.append("source_group_id = ?")
                fact_params.append(requested_group_id)
            fact_rows = conn.execute(
                f"""
                SELECT subject_user_id AS user_id, source_group_id, MAX(updated_at) AS updated_at
                FROM member_facts
                WHERE {' AND '.join(fact_where)}
                {fact_user_filter_sql}
                GROUP BY subject_user_id, source_group_id
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                [*fact_params, *fact_user_filter_params, query_limit],
            ).fetchall()

            profile_where = "WHERE 1 = 1"
            profile_params: list[object] = []
            if user_variants:
                placeholders = ", ".join("?" for _ in user_variants)
                profile_where += f" AND user_id IN ({placeholders})"
                profile_params.extend(user_variants)
            profile_rows = []
            if not requested_group_id:
                profile_rows = conn.execute(
                    f"""
                    SELECT user_id, updated_at
                    FROM member_profiles
                    {profile_where}
                    ORDER BY updated_at DESC
                    LIMIT ?
                    """,
                    [*profile_params, query_limit],
                ).fetchall()

            for row in relation_rows:
                entry = candidate_for(str(row["user_id"]))
                group_ids = entry["group_ids"]
                assert isinstance(group_ids, set)
                group_ids.add(str(row["group_id"]))
                entry["sort_at"] = max(int(entry["sort_at"]), int(row["updated_at"]))

            for row in fact_rows:
                entry = candidate_for(str(row["user_id"]))
                source_group_id = str(row["source_group_id"] or "")
                if source_group_id:
                    group_ids = entry["group_ids"]
                    assert isinstance(group_ids, set)
                    group_ids.add(source_group_id)
                entry["sort_at"] = max(int(entry["sort_at"]), int(row["updated_at"]))

            for row in profile_rows:
                entry = candidate_for(str(row["user_id"]))
                entry["sort_at"] = max(int(entry["sort_at"]), int(row["updated_at"]))

            items: list[dict[str, object]] = []
            for entry in candidates.values():
                dashboard_user_id = str(entry["user_id"])
                relationships = self._list_dashboard_relationship_rows(conn, dashboard_user_id, query_limit)
                fact_records = self._list_dashboard_user_fact_records(conn, dashboard_user_id, 20)
                member_profile = self._dashboard_member_profile(conn, dashboard_user_id)
                user_profile = self._dashboard_user_profile(conn, dashboard_user_id)
                group_ids = entry["group_ids"]
                assert isinstance(group_ids, set)
                for row in relationships:
                    group_ids.add(str(row["group_id"]))
                    entry["sort_at"] = max(int(entry["sort_at"]), int(row["updated_at"]))
                for record in fact_records:
                    if record.source_group_id:
                        group_ids.add(record.source_group_id)
                    entry["sort_at"] = max(int(entry["sort_at"]), record.updated_at)
                if member_profile is not None:
                    entry["sort_at"] = max(int(entry["sort_at"]), member_profile.updated_at)
                sorted_group_ids = sorted(group_ids)
                items.append(
                    {
                        "group_id": ", ".join(sorted_group_ids),
                        "group_ids": sorted_group_ids,
                        "user_id": dashboard_user_id,
                        "nickname": user_profile["nickname"],
                        "display_name": user_profile["display_name"],
                        "relationship": _dashboard_relationship_to_dict(
                            dashboard_user_id,
                            relationships,
                            sorted_group_ids,
                        ),
                        "profile": _user_profile_to_dict(member_profile),
                        "facts": [_fact_to_dict(record) for record in fact_records],
                        "updated_at": int(entry["sort_at"]),
                    }
                )

        items.sort(key=lambda item: int(item.get("updated_at", 0)), reverse=True)
        return items[:limit]

    def _list_dashboard_relationship_rows(
        self,
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
            WHERE user_id IN ({placeholders})
            ORDER BY updated_at DESC
            LIMIT ?
            """,
            [*variants, int(limit)],
        ).fetchall()

    def _list_dashboard_user_fact_records(
        self,
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

    def _dashboard_member_profile(
        self,
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

    def _dashboard_user_profile(
        self,
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

    def list_dashboard_messages(
        self,
        group_id: str = "",
        user_id: str = "",
        start_time: int | None = None,
        end_time: int | None = None,
        limit: int = 200,
    ) -> list[dict[str, object]]:
        where = []
        params: list[object] = []
        if group_id:
            where.append("group_id = ?")
            params.append(str(group_id))
        if user_id:
            where.append("user_id = ?")
            params.append(str(user_id))
        if start_time is not None:
            where.append("time >= ?")
            params.append(int(start_time))
        if end_time is not None:
            where.append("time < ?")
            params.append(int(end_time))
        where_sql = "WHERE " + " AND ".join(where) if where else ""

        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT id, time, group_id, user_id, message_id, raw_message, plain_text,
                       sender_name, sender_role
                FROM messages
                {where_sql}
                ORDER BY time DESC, id DESC
                LIMIT ?
                """,
                [*params, int(limit)],
            ).fetchall()
        messages = [
            {
                "id": int(row["id"]),
                "time": int(row["time"]),
                "group_id": str(row["group_id"]),
                "user_id": str(row["user_id"]),
                "message_id": str(row["message_id"]),
                "raw_message": str(row["raw_message"]),
                "plain_text": str(row["plain_text"]),
                "sender_name": str(row["sender_name"] or ""),
                "sender_role": str(row["sender_role"] or ""),
                "attachments": [],
                "mentions": [],
            }
            for row in rows
        ]
        self._attach_dashboard_attachments(messages)
        self._attach_dashboard_mentions(messages)
        return messages

    def _attach_dashboard_attachments(self, messages: list[dict[str, object]]) -> None:
        if not messages:
            return
        pairs = [(str(item["group_id"]), str(item["message_id"])) for item in messages]
        clauses = " OR ".join("(group_id = ? AND message_id = ?)" for _ in pairs)
        params: list[object] = []
        for group_id, message_id in pairs:
            params.extend([group_id, message_id])
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT group_id, message_id, attachment_type, file, url, summary, raw_data
                FROM message_attachments
                WHERE {clauses}
                ORDER BY id
                """,
                params,
            ).fetchall()
        by_key: dict[tuple[str, str], list[dict[str, object]]] = {}
        for row in rows:
            key = (str(row["group_id"]), str(row["message_id"]))
            by_key.setdefault(key, []).append(
                {
                    "attachment_type": str(row["attachment_type"]),
                    "file": str(row["file"] or ""),
                    "url": str(row["url"] or ""),
                    "summary": str(row["summary"] or ""),
                    "raw_data": str(row["raw_data"] or ""),
                }
            )
        for item in messages:
            key = (str(item["group_id"]), str(item["message_id"]))
            item["attachments"] = by_key.get(key, [])

    def _attach_dashboard_mentions(self, messages: list[dict[str, object]]) -> None:
        if not messages:
            return
        pairs = [(str(item["group_id"]), str(item["message_id"])) for item in messages]
        clauses = " OR ".join("(group_id = ? AND message_id = ?)" for _ in pairs)
        params: list[object] = []
        for group_id, message_id in pairs:
            params.extend([group_id, message_id])
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT group_id, message_id, mentioned_user_id, display_name, is_bot, raw_data
                FROM message_mentions
                WHERE {clauses}
                ORDER BY id
                """,
                params,
            ).fetchall()
        by_key: dict[tuple[str, str], list[dict[str, object]]] = {}
        for row in rows:
            key = (str(row["group_id"]), str(row["message_id"]))
            by_key.setdefault(key, []).append(
                {
                    "user_id": str(row["mentioned_user_id"] or ""),
                    "display_name": str(row["display_name"] or ""),
                    "is_bot": bool(row["is_bot"]),
                    "raw_data": str(row["raw_data"] or ""),
                }
            )
        for item in messages:
            key = (str(item["group_id"]), str(item["message_id"]))
            item["mentions"] = by_key.get(key, [])

    def list_dashboard_stickers(
        self,
        group_id: str = "",
        limit: int = 200,
    ) -> list[dict[str, object]]:
        where = ["enabled = 1", "local_path != ''"]
        params: list[object] = []
        if group_id:
            where.append("group_id = ?")
            params.append(str(group_id))
        params.append(int(limit))
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT id, group_id, source_user_id, source_message_id, url, file,
                       local_path, sha256, description, ocr_text, mood, usage,
                       tags, confidence, enabled, created_at, updated_at,
                       last_seen_at, hit_count, send_count, last_sent_at
                FROM sticker_assets
                WHERE {' AND '.join(where)}
                ORDER BY group_id, confidence DESC, hit_count DESC, updated_at DESC, id DESC
                LIMIT ?
                """,
                params,
            ).fetchall()
        return [_sticker_asset_to_dict(_sticker_asset_record(row)) for row in rows]

    def list_dashboard_pending(self, limit: int = 100) -> list[dict[str, object]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, owner_type, owner_id, kind, content, confidence, importance, status,
                       updated_at, source_user_id, source_group_id, subject_user_id,
                       claim_scope, verification_status
                FROM memory_items
                WHERE status IN ('pending_confirmation', 'conflict')
                ORDER BY updated_at DESC, id DESC
                LIMIT ?
                """,
                (int(limit),),
            ).fetchall()
            fact_rows = conn.execute(
                """
                SELECT id, subject_user_id, fact_type, claim_text, topic, stance,
                       confidence, status, claim_scope, source_user_id, source_group_id,
                       evidence_message_id, evidence_text, created_at, updated_at
                FROM member_facts
                WHERE status = 'pending_confirmation'
                ORDER BY updated_at DESC, id DESC
                LIMIT ?
                """,
                (int(limit),),
            ).fetchall()

        items = []
        for row in rows:
            record = _memory_record(row)
            data = _memory_to_dict(record)
            data["item_type"] = "memory"
            prefix = "#bot persona self" if record.owner_type == "self" else "#bot memory"
            data["approve_command"] = f"{prefix} approve {record.id}"
            data["reject_command"] = f"{prefix} reject {record.id}"
            items.append(data)
        for row in fact_rows:
            record = _fact_record(row)
            data = _fact_to_dict(record)
            data["item_type"] = "fact"
            data["approve_command"] = f"#bot facts approve {record.id}"
            data["reject_command"] = f"#bot facts reject {record.id}"
            items.append(data)
        items.sort(key=lambda item: int(item.get("updated_at", 0)), reverse=True)
        return items[:limit]

    def build_snapshot(self, context: MessageContext) -> ConversationSnapshot:
        human_count, bot_count = self.get_recent_activity_counts(context.group_id)
        target_users, unknown_refs, ambiguous_refs = self._resolve_target_user_contexts(context)
        speaker_messages, other_messages = self.get_focused_recent_messages(
            context.group_id,
            context.user_id,
        )
        recent_bot_reply, recent_bot_reply_seconds = self.get_recent_bot_reply_to_user(
            context.group_id,
            context.user_id,
            self.interaction_followup_seconds,
        )
        return ConversationSnapshot(
            recent_messages=self.get_recent_messages(context.group_id, limit=12),
            speaker_recent_messages=speaker_messages,
            other_recent_messages=other_messages,
            recent_bot_reply_to_user=recent_bot_reply,
            recent_bot_reply_to_user_seconds=recent_bot_reply_seconds,
            recent_human_messages_60s=human_count,
            recent_bot_messages_120s=bot_count,
            recent_image_descriptions=self.get_recent_image_descriptions(context.group_id, limit=8),
            sticker_assets=self.list_sticker_assets(
                context.group_id,
                limit=self.sticker_context_limit,
            ),
            user_memories=[],
            user_facts=self.list_user_facts(context.user_id, limit=20),
            user_profile=self.get_user_profile(context.user_id),
            self_memories=self.list_memories("self", "bot", limit=12),
            group_reflections=self.list_memories("group", context.group_id, limit=3),
            group_lexicon=self.list_group_lexicon_records(context.group_id, limit=10),
            relationship=self.get_relationship(context.group_id, context.user_id),
            persona_lines=self.get_persona_lines(),
            target_users=target_users,
            unknown_name_refs=unknown_refs,
            ambiguous_name_refs=ambiguous_refs,
        )

    def _resolve_target_user_contexts(
        self,
        context: MessageContext,
    ) -> tuple[list[TargetUserContext], list[str], dict[str, list[str]]]:
        context = replace(context, plain_text=strip_quoted_messages(context.plain_text))
        target_reasons: dict[str, str] = {}
        for user_id in _extract_explicit_target_user_ids(context):
            target_reasons.setdefault(user_id, "explicit_qq")

        name_refs = _extract_identity_name_refs(context.plain_text)
        unknown_refs: list[str] = []
        ambiguous_refs: dict[str, list[str]] = {}
        with self._connect() as conn:
            for ref in name_refs:
                matches = self._lookup_alias_users(conn, ref)
                if not matches:
                    unknown_refs.append(ref)
                    continue
                if len(matches) > 1:
                    ambiguous_refs[ref] = matches[: self.target_user_limit]
                    continue
                target_reasons.setdefault(matches[0], f"alias:{ref}")

            contexts: list[TargetUserContext] = []
            for user_id, reason in list(target_reasons.items())[: self.target_user_limit]:
                aliases = self._list_active_aliases(conn, user_id, limit=12)
                facts = self.list_user_facts(
                    user_id,
                    limit=self.context_fact_limit,
                    status="accepted",
                )
                contexts.append(
                    TargetUserContext(
                        user_id=user_id,
                        resolution_status="resolved",
                        match_reason=reason,
                        aliases=aliases,
                        facts=facts,
                        profile=self.get_user_profile(user_id),
                    )
                )

        return contexts, _dedupe_short_strings(unknown_refs), ambiguous_refs

    def _lookup_alias_users(self, conn: sqlite3.Connection, name: str) -> list[str]:
        alias = _clean_alias(name)
        if not alias or not _is_reasonable_member_alias(alias):
            return []
        rows = conn.execute(
            """
            SELECT user_id, MAX(confidence) AS confidence, MAX(updated_at) AS updated_at
            FROM member_aliases
            WHERE alias = ?
              AND status = 'active'
            GROUP BY user_id
            ORDER BY confidence DESC, updated_at DESC
            LIMIT ?
            """,
            (alias, self.target_user_limit + 1),
        ).fetchall()
        return [str(row["user_id"]) for row in rows if str(row["user_id"] or "").strip()]

    def _list_active_aliases(
        self,
        conn: sqlite3.Connection,
        user_id: str,
        limit: int = 12,
    ) -> list[str]:
        subject = _dashboard_user_id(user_id)
        if not subject:
            return []
        rows = conn.execute(
            """
            SELECT alias
            FROM member_aliases
            WHERE user_id = ?
              AND status = 'active'
            ORDER BY confidence DESC, updated_at DESC, id DESC
            LIMIT ?
            """,
            (subject, int(limit)),
        ).fetchall()
        return _dedupe_short_strings(
            str(row["alias"] or "")
            for row in rows
            if _is_reasonable_member_alias(str(row["alias"] or ""))
        )

    def list_group_lexicon_records(
        self,
        group_id: str,
        term: str = "",
        limit: int = 12,
        status: str = "active",
    ) -> list[MemoryRecord]:
        normalized_term = _normalize_lexicon_term(term)
        params: list[object] = [str(group_id), str(status)]
        where = [
            "owner_type = 'group'",
            "owner_id = ?",
            "kind = 'lexicon'",
            "status = ?",
        ]
        if normalized_term:
            where.append("(subject_user_id = ? OR content LIKE ?)")
            params.extend([_lexicon_subject(normalized_term), f"%{term}%"])
        params.append(int(limit))

        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT id, owner_type, owner_id, kind, content, confidence, importance, status,
                       updated_at, source_user_id, source_group_id, subject_user_id,
                       claim_scope, verification_status
                FROM memory_items
                WHERE {' AND '.join(where)}
                ORDER BY importance DESC, updated_at DESC, id DESC
                LIMIT ?
                """,
                params,
            ).fetchall()
        return [_memory_record(row) for row in rows]

    def list_group_lexicon(self, group_id: str, term: str = "", limit: int = 12) -> list[str]:
        return [
            format_memory_record(record)
            for record in self.list_group_lexicon_records(group_id, term=term, limit=limit)
        ]

    def has_group_lexicon(
        self,
        group_id: str,
        term: str,
        statuses: Iterable[str] = ("active", "pending_confirmation", "conflict"),
    ) -> bool:
        normalized_term = _normalize_lexicon_term(term)
        if not normalized_term:
            return False
        status_list = [str(status) for status in statuses]
        placeholders = ",".join("?" for _ in status_list)
        with self._connect() as conn:
            row = conn.execute(
                f"""
                SELECT 1
                FROM memory_items
                WHERE owner_type = 'group'
                  AND owner_id = ?
                  AND kind = 'lexicon'
                  AND subject_user_id = ?
                  AND status IN ({placeholders})
                LIMIT 1
                """,
                [str(group_id), _lexicon_subject(normalized_term), *status_list],
            ).fetchone()
        return row is not None

    def get_persona_lines(self) -> list[str]:
        with self._connect() as conn:
            rows = conn.execute("SELECT key, value FROM persona_state ORDER BY key").fetchall()
        lines = [f"{row['key']}: {row['value']}" for row in rows]
        self_memories = self.list_memories("self", "bot", limit=8)
        lines.extend(f"self_memory#{record.id}: {record.content}" for record in self_memories)
        return lines

    def format_persona(self) -> str:
        return "\n".join(self.get_persona_lines()) or "(empty)"

    def should_reflect(self, group_id: str, threshold: int, min_interval_seconds: int) -> bool:
        now = int(time.time())
        with self._connect() as conn:
            latest = conn.execute(
                """
                SELECT created_at
                FROM memory_items
                WHERE owner_type = 'group'
                  AND owner_id = ?
                  AND kind = 'reflection'
                  AND status = 'active'
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (str(group_id),),
            ).fetchone()
            latest_time = int(latest["created_at"]) if latest else 0
            if latest_time and now - latest_time < min_interval_seconds:
                return False
            count = conn.execute(
                "SELECT COUNT(1) AS count FROM messages WHERE group_id = ? AND time > ?",
                (str(group_id), latest_time),
            ).fetchone()
        return int(count["count"]) >= threshold

    def record_decision(
        self,
        context: MessageContext,
        decision: ParticipationDecision,
        reply: str | None,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO bot_decisions (
                    time, group_id, user_id, message_id, mode, action, reason, score,
                    value_type, value_score, traffic_level, reply
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    int(time.time()),
                    context.group_id,
                    context.user_id,
                    context.message_id,
                    decision.mode,
                    decision.action,
                    decision.reason,
                    decision.score,
                    decision.value_type,
                    decision.value_score,
                    decision.traffic_level,
                    reply or "",
                ),
            )

    def get_last_decision(self, group_id: str) -> str:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT time, user_id, message_id, mode, action, reason, score,
                       value_type, value_score, traffic_level, reply
                FROM bot_decisions
                WHERE group_id = ?
                ORDER BY id DESC
                LIMIT 1
                """,
                (str(group_id),),
            ).fetchone()
        if row is None:
            return "暂无决策记录。"
        return (
            f"time={row['time']}\n"
            f"user={row['user_id']}\n"
            f"message={row['message_id']}\n"
            f"mode={row['mode']}\n"
            f"action={row['action']}\n"
            f"score={row['score']:.2f}\n"
            f"value={row['value_type'] or 'none'}:{row['value_score']:.2f}\n"
            f"traffic={row['traffic_level'] or 'normal'}\n"
            f"reason={row['reason']}\n"
            f"reply={row['reply'] or '(none)'}"
        )

    def build_conflict_confirmation(
        self,
        conflicts: Iterable[MemoryCandidate],
        context: MessageContext,
        mode: ParticipationMode,
    ) -> str | None:
        if mode == "silent" or not context.is_direct:
            return None
        for item in conflicts:
            if item.claim_scope not in {"self_report", "bot_directed"}:
                continue
            if item.owner_type != "user" or item.subject_user_id != context.user_id:
                continue
            if item.kind in SENSITIVE_CONFIRMATION_KINDS:
                continue
            if not item.conflict_of:
                continue
            old = self.get_memory_record(item.conflict_of)
            if old is None:
                continue
            return f"我之前记得你说过「{old.content}」，现在是改成「{item.content}」了吗？"
        return None

    def get_memory_record(self, memory_id: int) -> MemoryRecord | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT id, owner_type, owner_id, kind, content, confidence, importance, status,
                       updated_at, source_user_id, source_group_id, subject_user_id,
                       claim_scope, verification_status
                FROM memory_items
                WHERE id = ?
                """,
                (int(memory_id),),
            ).fetchone()
        return _memory_record(row) if row else None

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

    def is_user_ignored(self, user_id: str) -> bool:
        user_id = _dashboard_user_id(user_id)
        if not user_id:
            return False
        with self._connect() as conn:
            row = conn.execute(
                "SELECT ignored FROM ignored_users WHERE user_id = ?",
                (user_id,),
            ).fetchone()
        return bool(row["ignored"]) if row is not None else False

    def add_ignored_user(self, user_id: str) -> None:
        user_id = _dashboard_user_id(user_id)
        if not user_id:
            return
        now = int(time.time())
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO ignored_users (user_id, ignored, updated_at)
                VALUES (?, 1, ?)
                ON CONFLICT(user_id) DO UPDATE SET
                    ignored = 1,
                    updated_at = excluded.updated_at
                """,
                (user_id, now),
            )

    def remove_ignored_user(self, user_id: str) -> None:
        user_id = _dashboard_user_id(user_id)
        if not user_id:
            return
        now = int(time.time())
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO ignored_users (user_id, ignored, updated_at)
                VALUES (?, 0, ?)
                ON CONFLICT(user_id) DO UPDATE SET
                    ignored = 0,
                    updated_at = excluded.updated_at
                """,
                (user_id, now),
            )

    def list_ignored_users(self) -> list[str]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT user_id FROM ignored_users WHERE ignored = 1 ORDER BY user_id"
            ).fetchall()
        return [str(row["user_id"]) for row in rows]

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            try:
                yield conn
                conn.commit()
            finally:
                conn.close()

    def _seed_config(self, conn: sqlite3.Connection) -> None:
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
        conn.executemany(
            """
            INSERT INTO ignored_users (user_id, ignored, updated_at)
            VALUES (?, 1, ?)
            ON CONFLICT(user_id) DO UPDATE SET ignored = 1, updated_at = excluded.updated_at
            """,
            [(user_id, now) for user_id in self.initial_ignored_users],
        )
        conn.executemany(
            """
            INSERT INTO persona_state (key, value, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(key) DO NOTHING
            """,
            [(key, value, now) for key, value in self.initial_persona.items()],
        )
    def _normalize_fact_candidate(self, item: FactCandidate) -> FactCandidate:
        source_user_id = _dashboard_user_id(item.source_user_id)
        subject_user_id = _dashboard_user_id(item.subject_user_id)
        claim_scope = _safe_claim_scope(item.claim_scope)
        if claim_scope == "self_report" and subject_user_id and source_user_id and subject_user_id != source_user_id:
            claim_scope = "third_party"
        importance = _fact_importance(item)
        return replace(
            item,
            subject_user_id=subject_user_id,
            fact_type=_clean_fact_field(item.fact_type, 40),
            claim_text=_clean_fact_field(item.claim_text, 300),
            topic=_clean_fact_field(item.topic, 120),
            stance=_clean_fact_field(item.stance, 60),
            confidence=_clamp_float(item.confidence),
            importance=importance,
            status=_safe_fact_status(item.status),
            claim_scope=claim_scope,
            source_user_id=source_user_id,
            source_group_id=str(item.source_group_id).strip(),
            evidence_message_id=str(item.evidence_message_id).strip(),
            evidence_text=_clean_fact_field(item.evidence_text, 1000),
        )

    def _fact_acceptance_status(
        self,
        conn: sqlite3.Connection,
        item: FactCandidate,
    ) -> str:
        if not _is_complete_fact(item):
            return "rejected"
        if item.confidence < self.fact_confidence_threshold:
            return "rejected"
        if _looks_low_value_fact(item):
            return "rejected"
        if _is_unreasonable_alias_fact(item):
            return "rejected"
        if item.claim_scope == "third_party":
            source_trust = self._source_trust(conn, item.source_group_id, item.source_user_id)
            if (
                source_trust >= self.third_party_trust_threshold
                and item.confidence >= self.third_party_confidence_threshold
            ):
                return "accepted"
            return "pending_confirmation"
        return "accepted"

    def _find_duplicate_fact(
        self,
        conn: sqlite3.Connection,
        item: FactCandidate,
        acceptance_status: str,
    ) -> FactRecord | None:
        target_status = "accepted" if acceptance_status == "accepted" else "pending_confirmation"
        row = conn.execute(
            """
            SELECT id, subject_user_id, fact_type, claim_text, topic, stance,
                   confidence, status, claim_scope, source_user_id, source_group_id,
                   evidence_message_id, evidence_text, created_at, updated_at
            FROM member_facts
            WHERE subject_user_id = ?
              AND fact_type = ?
              AND claim_text = ?
              AND status = ?
            ORDER BY confidence DESC, updated_at DESC
            LIMIT 1
            """,
            (item.subject_user_id, item.fact_type, item.claim_text, target_status),
        ).fetchone()
        if row:
            return _fact_record(row)
        if not item.evidence_message_id:
            return None
        row = conn.execute(
            """
            SELECT id, subject_user_id, fact_type, claim_text, topic, stance,
                   confidence, status, claim_scope, source_user_id, source_group_id,
                   evidence_message_id, evidence_text, created_at, updated_at
            FROM member_facts
            WHERE subject_user_id = ?
              AND evidence_message_id = ?
              AND claim_text = ?
              AND status = ?
            ORDER BY confidence DESC, updated_at DESC
            LIMIT 1
            """,
            (item.subject_user_id, item.evidence_message_id, item.claim_text, target_status),
        ).fetchone()
        return _fact_record(row) if row else None

    def _find_conflicting_facts(
        self,
        conn: sqlite3.Connection,
        item: FactCandidate,
    ) -> list[FactRecord]:
        if item.status in FACT_INACTIVE_STATUSES:
            return []
        if item.fact_type in {"identity", "alias"}:
            denied_aliases = _extract_denied_aliases(item.claim_text, item.evidence_text)
            if not denied_aliases:
                return []
            clauses = []
            params: list[object] = [item.subject_user_id]
            for alias in denied_aliases:
                clauses.append("(claim_text LIKE ? OR evidence_text LIKE ? OR topic LIKE ?)")
                like = f"%{alias}%"
                params.extend([like, like, like])
            rows = conn.execute(
                f"""
                SELECT id, subject_user_id, fact_type, claim_text, topic, stance,
                       confidence, status, claim_scope, source_user_id, source_group_id,
                       evidence_message_id, evidence_text, created_at, updated_at,
                       importance, last_seen_at, superseded_by_fact_id, forget_reason
                FROM member_facts
                WHERE subject_user_id = ?
                  AND status = 'accepted'
                  AND fact_type IN ('identity', 'alias')
                  AND ({' OR '.join(clauses)})
                ORDER BY confidence DESC, updated_at DESC
                LIMIT 10
                """,
                params,
            ).fetchall()
            return [_fact_record(row) for row in rows]

        if item.fact_type not in {"preference", "dislike", "opinion", "habit", "boundary", "event_stance"}:
            return []
        rows = conn.execute(
            """
            SELECT id, subject_user_id, fact_type, claim_text, topic, stance,
                   confidence, status, claim_scope, source_user_id, source_group_id,
                   evidence_message_id, evidence_text, created_at, updated_at,
                   importance, last_seen_at, superseded_by_fact_id, forget_reason
            FROM member_facts
            WHERE subject_user_id = ?
              AND fact_type = ?
              AND topic = ?
              AND status = 'accepted'
              AND claim_text != ?
            ORDER BY confidence DESC, updated_at DESC
            LIMIT 5
            """,
            (item.subject_user_id, item.fact_type, item.topic, item.claim_text),
        ).fetchall()
        return [_fact_record(row) for row in rows]

    def _supersede_facts(
        self,
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

    def _sync_aliases_for_fact(self, conn: sqlite3.Connection, fact: FactRecord) -> None:
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

    def _insert_fact(
        self,
        conn: sqlite3.Connection,
        item: FactCandidate,
        now: int,
    ) -> FactRecord:
        cursor = conn.execute(
            """
            INSERT INTO member_facts (
                subject_user_id, fact_type, claim_text, topic, stance,
                confidence, status, claim_scope, source_user_id, source_group_id,
                evidence_message_id, evidence_text, created_at, updated_at,
                importance, last_seen_at, superseded_by_fact_id, forget_reason
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                item.subject_user_id,
                item.fact_type,
                item.claim_text,
                item.topic,
                item.stance,
                item.confidence,
                item.status,
                item.claim_scope,
                item.source_user_id,
                item.source_group_id,
                item.evidence_message_id,
                item.evidence_text,
                now,
                now,
                item.importance,
                now,
                None,
                "",
            ),
        )
        return FactRecord(
            id=int(cursor.lastrowid),
            subject_user_id=item.subject_user_id,
            fact_type=item.fact_type,
            claim_text=item.claim_text,
            topic=item.topic,
            stance=item.stance,
            confidence=item.confidence,
            status=item.status,
            claim_scope=item.claim_scope,
            source_user_id=item.source_user_id,
            source_group_id=item.source_group_id,
            evidence_message_id=item.evidence_message_id,
            evidence_text=item.evidence_text,
            created_at=now,
            updated_at=now,
            importance=item.importance,
            last_seen_at=now,
        )

    def _normalize_memory_candidate(self, item: MemoryCandidate) -> MemoryCandidate:
        return replace(
            item,
            owner_id=str(item.owner_id),
            kind=item.kind.strip(),
            content=" ".join(item.content.split()),
            confidence=_clamp_float(item.confidence),
            importance=_clamp_float(item.importance),
            source_text=item.source_text.strip(),
            source_user_id=str(item.source_user_id or item.owner_id),
            source_group_id=str(item.source_group_id),
            subject_user_id=str(item.subject_user_id or item.owner_id),
            claim_scope=_safe_claim_scope(item.claim_scope),
            verification_status=_safe_verification_status(item.verification_status),
        )

    def _acceptance_status(
        self,
        conn: sqlite3.Connection,
        item: MemoryCandidate,
        confidence_threshold: float,
    ) -> str:
        if not item.content or item.confidence < confidence_threshold:
            return "rejected"

        if item.owner_type == "self" or item.subject_user_id == "bot" or item.claim_scope == "bot_directed":
            if item.source_user_id == "bot" or self._is_admin_conn(conn, item.source_user_id):
                return "accepted"
            return "pending_confirmation"

        if item.claim_scope == "third_party":
            source_trust = self._source_trust(conn, item.source_group_id, item.source_user_id)
            return "accepted" if source_trust >= TRUSTED_THIRD_PARTY_THRESHOLD else "pending_confirmation"

        if item.claim_scope == "group_fact":
            if item.source_user_id == "bot":
                return "accepted" if item.confidence >= confidence_threshold else "rejected"
            source_trust = self._source_trust(conn, item.source_group_id, item.source_user_id)
            if source_trust >= TRUSTED_THIRD_PARTY_THRESHOLD and item.confidence >= 0.85:
                return "accepted"
            return "pending_confirmation"

        return "accepted"

    def _source_trust(self, conn: sqlite3.Connection, group_id: str, user_id: str) -> int:
        row = conn.execute(
            """
            SELECT trust
            FROM relationships
            WHERE group_id = ? AND user_id = ?
            """,
            (str(group_id), str(user_id)),
        ).fetchone()
        return int(row["trust"]) if row else 0

    def _is_admin_conn(self, conn: sqlite3.Connection, user_id: str) -> bool:
        if str(user_id) in self.initial_admins:
            return True
        row = conn.execute("SELECT 1 FROM admins WHERE user_id = ?", (str(user_id),)).fetchone()
        return row is not None

    def _find_duplicate_memory(
        self,
        conn: sqlite3.Connection,
        item: MemoryCandidate,
    ) -> MemoryRecord | None:
        if item.kind == "lexicon" and item.subject_user_id:
            row = conn.execute(
                """
                SELECT id, owner_type, owner_id, kind, content, confidence, importance, status,
                       updated_at, source_user_id, source_group_id, subject_user_id,
                       claim_scope, verification_status
                FROM memory_items
                WHERE owner_type = ?
                  AND owner_id = ?
                  AND kind = 'lexicon'
                  AND subject_user_id = ?
                  AND status = 'active'
                LIMIT 1
                """,
                (item.owner_type, item.owner_id, item.subject_user_id),
            ).fetchone()
            if row:
                return _memory_record(row)

        row = conn.execute(
            """
            SELECT id, owner_type, owner_id, kind, content, confidence, importance, status,
                   updated_at, source_user_id, source_group_id, subject_user_id,
                   claim_scope, verification_status
            FROM memory_items
            WHERE owner_type = ?
              AND owner_id = ?
              AND kind = ?
              AND content = ?
              AND status = 'active'
            LIMIT 1
            """,
            (item.owner_type, item.owner_id, item.kind, item.content),
        ).fetchone()
        return _memory_record(row) if row else None

    def _find_conflicting_memory(
        self,
        conn: sqlite3.Connection,
        item: MemoryCandidate,
    ) -> MemoryRecord | None:
        if item.owner_type == "self" and item.kind in {"self_preference", "self_boundary"}:
            return self._find_conflicting_self_memory(conn, item)

        if item.kind not in CONFLICT_SENSITIVE_KINDS:
            return None
        row = conn.execute(
            """
            SELECT id, owner_type, owner_id, kind, content, confidence, importance, status,
                   updated_at, source_user_id, source_group_id, subject_user_id,
                   claim_scope, verification_status
            FROM memory_items
            WHERE owner_type = ?
              AND owner_id = ?
              AND kind = ?
              AND content != ?
              AND status = 'active'
            ORDER BY confidence DESC, updated_at DESC
            LIMIT 1
            """,
            (item.owner_type, item.owner_id, item.kind, item.content),
        ).fetchone()
        return _memory_record(row) if row else None

    def _find_conflicting_self_memory(
        self,
        conn: sqlite3.Connection,
        item: MemoryCandidate,
    ) -> MemoryRecord | None:
        rows = conn.execute(
            """
            SELECT id, owner_type, owner_id, kind, content, confidence, importance, status,
                   updated_at, source_user_id, source_group_id, subject_user_id,
                   claim_scope, verification_status
            FROM memory_items
            WHERE owner_type = 'self'
              AND owner_id = ?
              AND kind = ?
              AND content != ?
              AND status = 'active'
            ORDER BY confidence DESC, updated_at DESC
            """,
            (item.owner_id, item.kind, item.content),
        ).fetchall()
        for row in rows:
            record = _memory_record(row)
            if _looks_like_self_direct_conflict(item.content, record.content):
                return record
        return None

    def _insert_memory(self, conn: sqlite3.Connection, item: MemoryCandidate, now: int) -> None:
        conn.execute(
            """
            INSERT INTO memory_items (
                owner_type, owner_id, kind, content, confidence, importance, status,
                evidence_message_id, source_text, source_user_id, source_group_id,
                subject_user_id, claim_scope, verification_status,
                conflict_of, created_at, updated_at, last_seen_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                item.owner_type,
                item.owner_id,
                item.kind,
                item.content,
                item.confidence,
                item.importance,
                item.status,
                item.evidence_message_id,
                item.source_text,
                item.source_user_id,
                item.source_group_id,
                item.subject_user_id,
                item.claim_scope,
                item.verification_status,
                item.conflict_of,
                now,
                now,
                now,
            ),
        )


def format_memory_record(record: MemoryRecord) -> str:
    return (
        f"#{record.id} [{record.kind}/{record.status}/{record.claim_scope}] "
        f"{record.content} -> {record.owner_type}:{record.owner_id} "
        f"(src={record.source_user_id}, subject={record.subject_user_id}, "
        f"conf={record.confidence:.2f}, imp={record.importance:.2f})"
    )


def format_fact_record(record: FactRecord) -> str:
    return (
        f"#{record.id} [{record.fact_type}/{record.status}/{record.claim_scope}] "
        f"{record.claim_text} -> user:{record.subject_user_id} "
        f"(topic={record.topic}, stance={record.stance or '-'}, "
        f"src={record.source_user_id}, conf={record.confidence:.2f})"
    )


def format_sticker_asset(asset: StickerAssetRecord) -> str:
    status = "enabled" if asset.enabled else "disabled"
    tags = "、".join(asset.tags[:5]) or "(no tags)"
    usage = asset.usage or asset.description or "(no usage)"
    return (
        f"#{asset.id} [{status}] mood={asset.mood or '(unknown)'} "
        f"tags={tags} conf={asset.confidence:.2f} hits={asset.hit_count}\n"
        f"用途：{usage}\n"
        f"本地：{asset.local_path}"
    )


def _sticker_asset_to_dict(asset: StickerAssetRecord) -> dict[str, object]:
    usage = asset.usage or asset.description or "(no usage)"
    return {
        "id": asset.id,
        "group_id": asset.group_id,
        "source_user_id": asset.source_user_id,
        "source_message_id": asset.source_message_id,
        "url": asset.url,
        "file": asset.file,
        "local_path": asset.local_path,
        "sha256": asset.sha256,
        "description": asset.description,
        "ocr_text": asset.ocr_text,
        "mood": asset.mood,
        "usage": usage,
        "trigger": usage,
        "tags": list(asset.tags),
        "confidence": asset.confidence,
        "enabled": asset.enabled,
        "created_at": asset.created_at,
        "updated_at": asset.updated_at,
        "last_seen_at": asset.last_seen_at,
        "hit_count": asset.hit_count,
        "send_count": asset.send_count,
        "last_sent_at": asset.last_sent_at,
        "delete_command": f"#bot stickers delete {asset.id}",
    }


def _llm_usage_summary_to_dict(row: sqlite3.Row | None, since: int) -> dict[str, object]:
    if row is None:
        return {
            "since": since,
            "calls": 0,
            "prompt_chars": 0,
            "completion_chars": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "first_at": 0,
            "last_at": 0,
        }
    return {
        "since": since,
        "calls": int(row["calls"] or 0),
        "prompt_chars": int(row["prompt_chars"] or 0),
        "completion_chars": int(row["completion_chars"] or 0),
        "prompt_tokens": int(row["prompt_tokens"] or 0),
        "completion_tokens": int(row["completion_tokens"] or 0),
        "total_tokens": int(row["total_tokens"] or 0),
        "first_at": int(row["first_at"] or 0),
        "last_at": int(row["last_at"] or 0),
    }


def _llm_usage_group_to_dict(row: sqlite3.Row) -> dict[str, object]:
    return {
        "purpose": str(row["purpose"] or ""),
        "model": str(row["model"] or ""),
        "calls": int(row["calls"] or 0),
        "prompt_chars": int(row["prompt_chars"] or 0),
        "completion_chars": int(row["completion_chars"] or 0),
        "prompt_tokens": int(row["prompt_tokens"] or 0),
        "completion_tokens": int(row["completion_tokens"] or 0),
        "total_tokens": int(row["total_tokens"] or 0),
        "last_at": int(row["last_at"] or 0),
    }


def _llm_usage_row_to_dict(row: sqlite3.Row) -> dict[str, object]:
    return {
        "id": int(row["id"]),
        "created_at": int(row["created_at"]),
        "purpose": str(row["purpose"] or ""),
        "model": str(row["model"] or ""),
        "prompt_chars": int(row["prompt_chars"] or 0),
        "completion_chars": int(row["completion_chars"] or 0),
        "prompt_tokens": int(row["prompt_tokens"] or 0),
        "completion_tokens": int(row["completion_tokens"] or 0),
        "total_tokens": int(row["total_tokens"] or 0),
    }


def _compact_persona_items(items: dict[str, str]) -> dict[str, str]:
    return {key: value.strip() for key, value in items.items() if value.strip()}


def _memory_to_dict(record: MemoryRecord) -> dict[str, object]:
    return {
        "id": record.id,
        "owner_type": record.owner_type,
        "owner_id": record.owner_id,
        "kind": record.kind,
        "content": record.content,
        "confidence": record.confidence,
        "importance": record.importance,
        "status": record.status,
        "updated_at": record.updated_at,
        "source_user_id": record.source_user_id,
        "source_group_id": record.source_group_id,
        "subject_user_id": record.subject_user_id,
        "claim_scope": record.claim_scope,
        "verification_status": record.verification_status,
    }


def _fact_to_dict(record: FactRecord) -> dict[str, object]:
    return {
        "id": record.id,
        "subject_user_id": record.subject_user_id,
        "fact_type": record.fact_type,
        "claim_text": record.claim_text,
        "topic": record.topic,
        "stance": record.stance,
        "confidence": record.confidence,
        "status": record.status,
        "claim_scope": record.claim_scope,
        "source_user_id": record.source_user_id,
        "source_group_id": record.source_group_id,
        "evidence_message_id": record.evidence_message_id,
        "evidence_text": record.evidence_text,
        "created_at": record.created_at,
        "updated_at": record.updated_at,
        "importance": record.importance,
        "last_seen_at": record.last_seen_at,
        "superseded_by_fact_id": record.superseded_by_fact_id,
        "forget_reason": record.forget_reason,
    }


def _user_profile_to_dict(record: UserProfileRecord | None) -> dict[str, object] | None:
    if record is None:
        return None
    return {
        "user_id": record.user_id,
        "summary": record.summary,
        "traits": record.traits,
        "supporting_fact_ids": list(record.supporting_fact_ids),
        "fact_count": record.fact_count,
        "version": record.version,
        "updated_at": record.updated_at,
    }


def _relationship_to_dict(relation: RelationshipState) -> dict[str, object]:
    return {
        "group_id": relation.group_id,
        "user_id": relation.user_id,
        "closeness": relation.closeness,
        "trust": relation.trust,
        "familiarity": relation.familiarity,
        "tension": relation.tension,
        "summary": relation.summary,
    }


def _relationship_row_to_dict(row: sqlite3.Row) -> dict[str, object]:
    data = _relationship_to_dict(
        RelationshipState(
            group_id=str(row["group_id"]),
            user_id=str(row["user_id"]),
            closeness=int(row["closeness"]),
            trust=int(row["trust"]),
            familiarity=int(row["familiarity"]),
            tension=int(row["tension"]),
            summary=str(row["summary"] or ""),
        )
    )
    data["updated_at"] = int(row["updated_at"])
    return data


def _format_message_context_line(row: sqlite3.Row) -> str:
    name = str(row["sender_name"] or row["user_id"])
    text = str(row["plain_text"] or "").strip()
    if not text:
        return ""
    return f"{name}: {text}"


def _relationship_rank_label(user_id: str, profile: dict[str, object] | None) -> str:
    name = ""
    if profile:
        name = str(profile.get("display_name") or profile.get("nickname") or "")
    name = _compact_display_text(name, 24)
    return f"{name}(QQ:{user_id})" if name else f"QQ:{user_id}"


def _compact_display_text(value: str, limit: int) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return f"{text[: max(0, limit - 3)].rstrip()}..."


def _dashboard_relationship_to_dict(
    user_id: str,
    rows: list[sqlite3.Row],
    group_ids: list[str],
) -> dict[str, object] | None:
    if not rows:
        return None
    summary = ""
    updated_at = 0
    closeness = 0
    trust = 0
    familiarity = 0
    tension = 0
    for row in rows:
        closeness += int(row["closeness"])
        trust += int(row["trust"])
        familiarity += int(row["familiarity"])
        tension += int(row["tension"])
        summary = _merge_summary(summary, str(row["summary"] or ""))
        updated_at = max(updated_at, int(row["updated_at"]))
    return {
        "group_id": ", ".join(group_ids),
        "group_ids": group_ids,
        "user_id": user_id,
        "closeness": _clamp_score(closeness),
        "trust": _clamp_score(trust),
        "familiarity": _clamp_score(familiarity),
        "tension": _clamp_score(tension),
        "summary": summary,
        "updated_at": updated_at,
    }


def _dashboard_user_id(value: str) -> str:
    user_id = str(value or "").strip()
    match = re.fullmatch(r"(?i)qq[:：]\s*(\d+)", user_id)
    return match.group(1) if match else user_id


def _dashboard_user_id_variants(value: str) -> list[str]:
    canonical = _dashboard_user_id(value)
    variants = [canonical]
    if canonical.isdigit():
        variants.extend([f"QQ:{canonical}", f"qq:{canonical}", f"QQ：{canonical}"])
    return list(dict.fromkeys(item for item in variants if item))


def _image_vision_cache_record(row: sqlite3.Row) -> ImageVisionCacheRecord:
    return ImageVisionCacheRecord(
        url=str(row["url"]),
        description=str(row["description"] or ""),
        ocr_text=str(row["ocr_text"] or ""),
        topics=tuple(_decode_string_list(str(row["topics"] or "[]"))),
        memory=str(row["memory"] or ""),
        confidence=float(row["confidence"]),
        importance=float(row["importance"]),
        model=str(row["model"] or ""),
        created_at=int(row["created_at"]),
        updated_at=int(row["updated_at"]),
        last_seen_at=int(row["last_seen_at"]),
        hit_count=int(row["hit_count"]),
    )


def _sticker_asset_record(row: sqlite3.Row) -> StickerAssetRecord:
    return StickerAssetRecord(
        id=int(row["id"]),
        group_id=str(row["group_id"]),
        source_user_id=str(row["source_user_id"] or ""),
        source_message_id=str(row["source_message_id"] or ""),
        url=str(row["url"] or ""),
        file=str(row["file"] or ""),
        local_path=str(row["local_path"] or ""),
        sha256=str(row["sha256"] or ""),
        description=str(row["description"] or ""),
        ocr_text=str(row["ocr_text"] or ""),
        mood=str(row["mood"] or ""),
        usage=str(row["usage"] or ""),
        tags=tuple(_decode_string_list(str(row["tags"] or "[]"))),
        confidence=float(row["confidence"]),
        enabled=bool(int(row["enabled"])),
        created_at=int(row["created_at"]),
        updated_at=int(row["updated_at"]),
        last_seen_at=int(row["last_seen_at"]),
        hit_count=int(row["hit_count"]),
        send_count=_row_int(row, "send_count", 0),
        last_sent_at=_row_int(row, "last_sent_at", 0),
    )


def _memory_record(row: sqlite3.Row) -> MemoryRecord:
    return MemoryRecord(
        id=int(row["id"]),
        owner_type=str(row["owner_type"]),
        owner_id=str(row["owner_id"]),
        kind=str(row["kind"]),
        content=str(row["content"]),
        confidence=float(row["confidence"]),
        importance=float(row["importance"]),
        status=str(row["status"]),
        updated_at=int(row["updated_at"]),
        source_user_id=_row_value(row, "source_user_id", ""),
        source_group_id=_row_value(row, "source_group_id", ""),
        subject_user_id=_row_value(row, "subject_user_id", str(row["owner_id"])),
        claim_scope=_row_value(row, "claim_scope", "self_report"),
        verification_status=_row_value(row, "verification_status", "accepted"),
    )


def _fact_record(row: sqlite3.Row) -> FactRecord:
    superseded_by = _row_value(row, "superseded_by_fact_id", "")
    return FactRecord(
        id=int(row["id"]),
        subject_user_id=str(row["subject_user_id"]),
        fact_type=str(row["fact_type"]),
        claim_text=str(row["claim_text"]),
        topic=str(row["topic"]),
        stance=str(row["stance"] or ""),
        confidence=float(row["confidence"]),
        status=str(row["status"]),
        claim_scope=str(row["claim_scope"]),
        source_user_id=str(row["source_user_id"] or ""),
        source_group_id=str(row["source_group_id"] or ""),
        evidence_message_id=str(row["evidence_message_id"] or ""),
        evidence_text=str(row["evidence_text"] or ""),
        created_at=int(row["created_at"]),
        updated_at=int(row["updated_at"]),
        importance=_row_float(row, "importance", 0.5),
        last_seen_at=_row_int(row, "last_seen_at", int(row["updated_at"])),
        superseded_by_fact_id=int(superseded_by) if superseded_by.strip() else None,
        forget_reason=_row_value(row, "forget_reason", ""),
    )


def _user_profile_record(row: sqlite3.Row) -> UserProfileRecord:
    return UserProfileRecord(
        user_id=str(row["user_id"]),
        summary=str(row["summary"] or ""),
        traits=_decode_traits(str(row["traits_json"] or "{}")),
        supporting_fact_ids=tuple(_decode_int_list(str(row["supporting_fact_ids"] or "[]"))),
        fact_count=int(row["fact_count"]),
        version=int(row["version"]),
        updated_at=int(row["updated_at"]),
    )


def _clamp_score(value: int) -> int:
    return max(0, min(100, int(value)))


def _clamp_float(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _safe_claim_scope(value: str) -> str:
    return value if value in {"self_report", "third_party", "bot_directed", "group_fact"} else "self_report"


def _safe_verification_status(value: str) -> str:
    return value if value in {"accepted", "pending_confirmation", "conflict", "rejected"} else "pending_confirmation"


def _safe_fact_status(value: str) -> str:
    return (
        value
        if value in {"candidate", "accepted", "pending_confirmation", "rejected", "superseded", "forgotten"}
        else "candidate"
    )


def _clean_fact_field(value: str, limit: int) -> str:
    return " ".join(str(value or "").strip().split())[:limit]


def _is_complete_fact(item: FactCandidate) -> bool:
    return bool(
        item.subject_user_id
        and item.fact_type
        and item.claim_text
        and item.topic
        and item.evidence_message_id
        and item.evidence_text
    )


def _looks_low_value_fact(item: FactCandidate) -> bool:
    text = f"{item.claim_text} {item.topic} {item.evidence_text}"
    if len(item.claim_text) < 6 or len(item.topic) < 2:
        return True
    low_value_markers = (
        "继续聊",
        "随口",
        "发了",
        "发送",
        "分享图片",
        "分享截图",
        "空消息",
        "表情包",
        "聊天",
        "参与讨论",
        "表达情绪",
    )
    return any(marker in text for marker in low_value_markers) and not any(
        signal in text for signal in ("认为", "觉得", "喜欢", "讨厌", "支持", "反对", "评价", "倾向")
    )


def _row_value(row: sqlite3.Row, key: str, default: str) -> str:
    try:
        value = row[key]
    except (IndexError, KeyError):
        return default
    return str(value or default)


def _row_int(row: sqlite3.Row, key: str, default: int) -> int:
    try:
        value = row[key]
    except (IndexError, KeyError):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: str, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _local_usage_date(timestamp: int) -> str:
    return time.strftime("%Y-%m-%d", time.localtime(int(timestamp)))


def _row_float(row: sqlite3.Row, key: str, default: float) -> float:
    try:
        value = row[key]
    except (IndexError, KeyError):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _extract_explicit_target_user_ids(context: MessageContext) -> list[str]:
    candidates: list[str] = []
    for mention in context.mentions:
        if mention.is_bot:
            continue
        user_id = _dashboard_user_id(mention.user_id)
        if user_id.isdigit():
            candidates.append(user_id)

    text = context.plain_text
    for match in re.finditer(r"(?i)(?:@?\s*QQ\s*[:：]\s*|qq=)(\d{5,20})", text):
        candidates.append(match.group(1))
    if _looks_like_identity_query(text):
        for match in re.finditer(r"(?<!\d)(\d{5,20})(?!\d)", text):
            candidates.append(match.group(1))
    return _dedupe_short_strings(candidates)


def _extract_identity_name_refs(text: str) -> list[str]:
    if not _looks_like_identity_query(text):
        return []
    direct_qq_pattern = re.compile(r"(?i)QQ\s*[:：]\s*\d{5,20}|\d{5,20}")
    refs: list[str] = []
    patterns = (
        re.compile(r"谁是\s*([^?？,，。!！\s@]{1,30})"),
        re.compile(r"([^?？,，。!！\s@]{1,30})\s*是谁"),
        re.compile(r"([^?？,，。!！\s@]{1,30})\s*(?:叫啥|叫什么|叫什麼)"),
        re.compile(r"(?:怎么|如何|要怎么|该怎么)\s*称呼\s*([^?？,，。!！\s@]{1,30})"),
        re.compile(r"(?:名字|昵称|外号|称呼)\s*(?:是|叫|为)\s*([^?？,，。!！\s@]{1,30})"),
    )
    for pattern in patterns:
        for match in pattern.finditer(text):
            raw = match.group(1)
            if direct_qq_pattern.fullmatch(raw.strip()):
                continue
            cleaned = _clean_alias(raw)
            if cleaned:
                refs.append(cleaned)
    return _dedupe_short_strings(refs)


def _looks_like_identity_query(text: str) -> bool:
    return bool(
        re.search(
            r"(谁是|是谁|叫啥|叫什么|叫什麼|怎么称呼|如何称呼|要怎么称呼|该怎么称呼|名字|昵称|外号|称呼)",
            text,
        )
    )


def _extract_aliases_from_fact(fact: FactRecord) -> list[tuple[str, str]]:
    if fact.fact_type not in {"identity", "alias"}:
        return []
    candidates = _extract_alias_candidates_from_fact(fact)
    return _dedupe_alias_pairs(
        (alias, alias_type)
        for alias, alias_type in candidates
        if _is_reasonable_member_alias(alias)
    )


def _extract_alias_candidates_from_fact(fact: FactCandidate | FactRecord) -> list[tuple[str, str]]:
    if fact.fact_type not in {"identity", "alias"}:
        return []
    texts = (fact.claim_text, fact.evidence_text, fact.topic)
    denied = set(_extract_denied_aliases(fact.claim_text, fact.evidence_text))
    aliases: list[tuple[str, str]] = []
    patterns = (
        ("alias", re.compile(r"(?:称呼|昵称|外号|名字)\s*(?:是|叫|为|：|:)?\s*[“\"']?([^，,。；;、\s”\"']{1,30})")),
        ("alias", re.compile(r"(?<!不)(?<!别)(?<!要)(?<!再)(?:叫做|称作|称为|叫)\s*(?:我|他|她|ta|TA)?\s*[“\"']?([^，,。；;、\s”\"']{1,30})")),
        ("identity", re.compile(r"(?:自称|身份是|是)\s*[“\"']?([^，,。；;、\s”\"']{1,30})")),
        ("alias", re.compile(r"(?:改叫|以后叫)\s*[“\"']?([^，,。；;、\s”\"']{1,30})")),
    )
    for alias_type, pattern in patterns:
        for text in texts:
            for match in pattern.finditer(text):
                alias = _clean_alias(match.group(1))
                if not alias or alias in denied:
                    continue
                aliases.append((alias, alias_type))
    return _dedupe_alias_pairs(aliases)


def _is_unreasonable_alias_fact(fact: FactCandidate | FactRecord) -> bool:
    if fact.fact_type not in {"identity", "alias"}:
        return False
    aliases = _extract_alias_candidates_from_fact(fact)
    return bool(aliases) and all(not _is_reasonable_member_alias(alias) for alias, _ in aliases)


def _is_reasonable_member_alias(value: str) -> bool:
    alias = _clean_alias(value)
    if not alias:
        return False
    compact = re.sub(r"\s+", "", alias)
    if compact in RELATIONAL_ALIAS_TERMS:
        return False
    relation_core = (
        "主人|主子|老板|老板娘|领导|管理员|管理|群主|版主|"
        "爸爸|爸|爹|父亲|妈妈|妈|母亲|哥哥|哥|姐姐|姐|"
        "弟弟|弟|妹妹|妹|儿子|女儿|老婆|老公|媳妇|丈夫|妻子|"
        "对象|男朋友|女朋友"
    )
    if re.fullmatch(rf"(?:小|老|大|阿)?(?:{relation_core})(?:大人)?", compact):
        return False
    if re.fullmatch(r"(?:第?[一二三四五六七八九十\d]+)?(?:管理员|管理|群主|版主)", compact):
        return False
    return True


def _extract_denied_aliases(*texts: str) -> list[str]:
    combined = "\n".join(str(text or "") for text in texts)
    aliases: list[str] = []
    patterns = (
        re.compile(r"(?:不是|不叫|别叫|不要叫|别再叫|不要再叫)\s*(?:我|他|她|ta|TA)?\s*[“\"']?([^，,。；;、\s”\"']{1,30})"),
        re.compile(r"[“\"']?([^，,。；;、\s”\"']{1,30})[”\"']?\s*(?:不是我|不是他|不是她|不是ta|不对)"),
    )
    for pattern in patterns:
        for match in pattern.finditer(combined):
            alias = _clean_alias(match.group(1))
            if alias:
                aliases.append(alias)
    return _dedupe_short_strings(aliases)


def _clean_alias(value: str) -> str:
    alias = " ".join(str(value or "").strip().split())
    alias = alias.strip("「」『』“”\"'`.,，。:：;；!?！？()（）[]【】")
    if not 1 <= len(alias) <= 30:
        return ""
    if re.fullmatch(r"(?i)qq[:：]?\d+", alias):
        return ""
    if alias in {"谁", "你", "我", "他", "她", "它", "ta", "TA", "这个", "那个", "哪位"}:
        return ""
    return alias


def _dedupe_alias_pairs(values: Iterable[tuple[str, str]]) -> list[tuple[str, str]]:
    result: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for alias, alias_type in values:
        key = (alias, alias_type)
        if alias and key not in seen:
            result.append(key)
            seen.add(key)
    return result


def _dedupe_short_strings(values: Iterable[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        item = " ".join(str(value or "").strip().split())
        if not item or item in seen:
            continue
        result.append(item)
        seen.add(item)
    return result


def _fact_importance(item: FactCandidate) -> float:
    base = _clamp_float(getattr(item, "importance", 0.5))
    if item.fact_type in {"identity", "boundary"}:
        base = max(base, 0.9)
    elif item.fact_type in {"skill", "habit"}:
        base = max(base, 0.75)
    elif item.fact_type in {"preference", "dislike"}:
        base = max(base, 0.55)
    if str(item.evidence_text or "").startswith("image_index="):
        base = min(base, 0.3)
    return base


def _compact_string_list(values: Iterable[str], limit: int = 10) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        item = " ".join(str(value).strip().split())[:80]
        if not item or item in seen:
            continue
        result.append(item)
        seen.add(item)
        if len(result) >= limit:
            break
    return result


def _sticker_text_key(value: str) -> str:
    return "".join(ch for ch in str(value).casefold() if ch.isalnum())


def _useful_sticker_text_key(value: str) -> bool:
    return len(value) >= 8 and len(set(value)) >= 3


def _decode_string_list(value: str) -> list[str]:
    try:
        decoded = json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return []
    if not isinstance(decoded, list):
        return []
    return _compact_string_list((str(item) for item in decoded), limit=10)


def _compact_int_list(values: Iterable[int], limit: int = 20) -> list[int]:
    result: list[int] = []
    seen: set[int] = set()
    for value in values:
        try:
            item = int(value)
        except (TypeError, ValueError):
            continue
        if item in seen:
            continue
        result.append(item)
        seen.add(item)
        if len(result) >= limit:
            break
    return result


def _decode_int_list(value: str) -> list[int]:
    try:
        decoded = json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return []
    if not isinstance(decoded, list):
        return []
    return _compact_int_list(decoded, limit=80)


def _decode_traits(value: str) -> dict[str, object]:
    try:
        decoded = json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return {}
    return decoded if isinstance(decoded, dict) else {}


def _normalize_lexicon_term(term: str) -> str:
    return " ".join(str(term).strip().lower().split())


def _lexicon_subject(term: str) -> str:
    return f"term:{_normalize_lexicon_term(term)}"


def _looks_like_self_direct_conflict(new_content: str, old_content: str) -> bool:
    positive_tokens = ("喜欢", "想", "会", "习惯", "可以")
    negative_tokens = ("不喜欢", "讨厌", "怕", "不会", "不太", "不能")
    new_positive = any(token in new_content for token in positive_tokens)
    new_negative = any(token in new_content for token in negative_tokens)
    old_positive = any(token in old_content for token in positive_tokens)
    old_negative = any(token in old_content for token in negative_tokens)
    shared = _self_object_terms(new_content) & _self_object_terms(old_content)
    return bool(shared and ((new_positive and old_negative) or (new_negative and old_positive)))


def _self_object_terms(content: str) -> set[str]:
    cleaned = content
    for token in ("不喜欢", "喜欢", "讨厌", "害怕", "怕", "我", "很", "比较", "一点", "有点"):
        cleaned = cleaned.replace(token, "")
    terms: set[str] = set()
    for phrase in re.findall(r"[\u4e00-\u9fffA-Za-z0-9]{2,}", cleaned):
        terms.add(phrase)
        if len(phrase) <= 12:
            terms.update(phrase[index : index + 2] for index in range(len(phrase) - 1))
    return {term for term in terms if len(term) >= 2}


def _merge_summary(current: str, patch: str) -> str:
    return merge_relationship_summary(current, patch)
