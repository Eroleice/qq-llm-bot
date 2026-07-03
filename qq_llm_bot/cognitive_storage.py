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
    UserProfileRecord,
    UserProfileDraft,
)
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


class BotStorage:
    def __init__(
        self,
        db_path: Path,
        initial_admins: Iterable[str],
        initial_groups: Iterable[str],
        initial_persona: dict[str, str],
        sticker_context_limit: int = 24,
        fact_confidence_threshold: float = 0.75,
        third_party_trust_threshold: int = 70,
        third_party_confidence_threshold: float = 0.85,
        profile_fact_threshold: int = 5,
    ) -> None:
        self.db_path = db_path
        self.initial_admins = {str(item) for item in initial_admins}
        self.initial_groups = {str(item) for item in initial_groups}
        self.initial_persona = initial_persona
        self.sticker_context_limit = max(1, int(sticker_context_limit))
        self.fact_confidence_threshold = _clamp_float(fact_confidence_threshold)
        self.third_party_trust_threshold = _clamp_score(third_party_trust_threshold)
        self.third_party_confidence_threshold = _clamp_float(third_party_confidence_threshold)
        self.profile_fact_threshold = max(1, int(profile_fact_threshold))
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
                }
            )
        )
        return cls(
            config.resolve_path(config.storage.sqlite_path),
            config.bot.admin_ids,
            config.bot.enabled_groups,
            persona,
            config.stickers.max_context_stickers,
            config.facts.fact_confidence_threshold,
            config.facts.third_party_trust_threshold,
            config.facts.third_party_confidence_threshold,
            config.facts.profile_fact_threshold,
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
                    hit_count INTEGER NOT NULL DEFAULT 0
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
                    updated_at INTEGER NOT NULL
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
                """
            )
            self._seed_config(conn)

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
        now = int(time.time())
        self.record_message(
            MessageContext(
                group_id=str(group_id),
                user_id=str(bot_id),
                message_id=f"bot-{now}",
                plain_text=reply,
                raw_message=reply,
                sender_name="bot",
                sender_role="bot",
                timestamp=now,
            )
        )

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
            if existing is not None:
                asset_id = int(existing["id"])
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
                        local_path,
                        local_path,
                        sha256,
                        sha256,
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
                       last_seen_at, hit_count
                FROM sticker_assets
                WHERE id = ?
                """,
                (asset_id,),
            ).fetchone()
        return _sticker_asset_record(row) if row is not None else None

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
                       last_seen_at, hit_count
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
                       last_seen_at, hit_count
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
        return asset

    def record_sticker_sent(self, sticker_id: int) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE sticker_assets
                SET last_seen_at = ?,
                    hit_count = hit_count + 1
                WHERE id = ?
                """,
                (int(time.time()), int(sticker_id)),
            )

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

                duplicate = self._find_duplicate_fact(conn, normalized, acceptance_status)
                if duplicate:
                    conn.execute(
                        """
                        UPDATE member_facts
                        SET confidence = MAX(confidence, ?),
                            updated_at = ?
                        WHERE id = ?
                        """,
                        (normalized.confidence, now, duplicate.id),
                    )
                    updated = replace(
                        duplicate,
                        confidence=max(duplicate.confidence, normalized.confidence),
                        updated_at=now,
                    )
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
    ) -> list[FactRecord]:
        subject = _dashboard_user_id(user_id)
        if not subject:
            return []
        where = ["subject_user_id = ?", "status = ?"]
        params: list[object] = [subject, _safe_fact_status(status)]
        if group_id:
            where.append("source_group_id = ?")
            params.append(str(group_id))
        limit_value = int(limit)
        limit_sql = "LIMIT ?" if limit_value > 0 else ""
        if limit_value > 0:
            params.append(limit_value)
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT id, subject_user_id, fact_type, claim_text, topic, stance,
                       confidence, status, claim_scope, source_user_id, source_group_id,
                       evidence_message_id, evidence_text, created_at, updated_at
                FROM member_facts
                WHERE {' AND '.join(where)}
                ORDER BY updated_at DESC, id DESC
                {limit_sql}
                """,
                params,
            ).fetchall()
        return [_fact_record(row) for row in rows]

    def list_user_facts_text(self, user_id: str, limit: int = 20, status: str = "accepted") -> list[str]:
        return [format_fact_record(record) for record in self.list_user_facts(user_id, limit, status)]

    def list_pending_facts(self, limit: int = 20) -> list[FactRecord]:
        with self._connect() as conn:
            rows = conn.execute(
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
        return [_fact_record(row) for row in rows]

    def get_fact_record(self, fact_id: int) -> FactRecord | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT id, subject_user_id, fact_type, claim_text, topic, stance,
                       confidence, status, claim_scope, source_user_id, source_group_id,
                       evidence_message_id, evidence_text, created_at, updated_at
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
                SET status = 'accepted', updated_at = ?
                WHERE id = ? AND status = 'pending_confirmation'
                """,
                (now, int(fact_id)),
            )
            if cursor.rowcount <= 0:
                return None
            row = conn.execute(
                """
                SELECT id, subject_user_id, fact_type, claim_text, topic, stance,
                       confidence, status, claim_scope, source_user_id, source_group_id,
                       evidence_message_id, evidence_text, created_at, updated_at
                FROM member_facts
                WHERE id = ?
                """,
                (int(fact_id),),
            ).fetchone()
        return _fact_record(row) if row else None

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
            name = str(row["sender_name"] or row["user_id"])
            text = str(row["plain_text"] or "").strip()
            if text:
                lines.append(f"{name}: {text}")
        return lines

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
            }
            for row in rows
        ]
        self._attach_dashboard_attachments(messages)
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
                       last_seen_at, hit_count
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
        return ConversationSnapshot(
            recent_messages=self.get_recent_messages(context.group_id, limit=12),
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
        return replace(
            item,
            subject_user_id=subject_user_id,
            fact_type=_clean_fact_field(item.fact_type, 40),
            claim_text=_clean_fact_field(item.claim_text, 300),
            topic=_clean_fact_field(item.topic, 120),
            stance=_clean_fact_field(item.stance, 60),
            confidence=_clamp_float(item.confidence),
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
                evidence_message_id, evidence_text, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
        "delete_command": f"#bot stickers delete {asset.id}",
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
    return value if value in {"candidate", "accepted", "pending_confirmation", "rejected"} else "candidate"


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
