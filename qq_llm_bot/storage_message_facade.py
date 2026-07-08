from __future__ import annotations

import sqlite3
from typing import Iterable

import qq_llm_bot.storage_messages as _storage_messages
from qq_llm_bot.models import ImageVisionCacheRecord, MessageContext


class StorageMessageFacadeMixin:
    def record_message(self, context: MessageContext) -> None:
        _storage_messages.record_message(self, context)

    def _upsert_user_profile(
        self,
        conn: sqlite3.Connection,
        context: MessageContext,
        now: int,
    ) -> None:
        _storage_messages.upsert_user_profile(conn, context, now)

    def record_bot_reply(self, group_id: str, bot_id: str, reply: str) -> None:
        _storage_messages.record_bot_reply(self, group_id, bot_id, reply)

    def record_bot_reply_parts(self, group_id: str, bot_id: str, replies: Iterable[str]) -> None:
        _storage_messages.record_bot_reply_parts(self, group_id, bot_id, replies)

    def get_recent_bot_reply_texts(self, group_id: str, limit: int = 10) -> list[str]:
        return _storage_messages.get_recent_bot_reply_texts(self, group_id, limit=limit)

    def _record_attachments(self, conn: sqlite3.Connection, context: MessageContext) -> None:
        _storage_messages.record_attachments(conn, context)

    def _record_mentions(self, conn: sqlite3.Connection, context: MessageContext) -> None:
        _storage_messages.record_mentions(conn, context)

    def update_image_descriptions(
        self,
        group_id: str,
        message_id: str,
        descriptions: list[str],
    ) -> None:
        _storage_messages.update_image_descriptions(self, group_id, message_id, descriptions)

    def get_image_vision_cache(self, url: str) -> ImageVisionCacheRecord | None:
        return _storage_messages.get_image_vision_cache(self, url)

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
        _storage_messages.upsert_image_vision_cache(
            self,
            url=url,
            description=description,
            ocr_text=ocr_text,
            topics=topics,
            memory=memory,
            confidence=confidence,
            importance=importance,
            model=model,
        )

    def get_recent_messages(self, group_id: str, limit: int = 12) -> list[str]:
        return _storage_messages.get_recent_messages(self, group_id, limit=limit)

    def get_focused_recent_messages(
        self,
        group_id: str,
        user_id: str,
        speaker_limit: int = 5,
        other_limit: int = 10,
    ) -> tuple[list[str], list[str]]:
        return _storage_messages.get_focused_recent_messages(
            self,
            group_id,
            user_id,
            speaker_limit=speaker_limit,
            other_limit=other_limit,
        )

    def get_recent_bot_reply_to_user(
        self,
        group_id: str,
        user_id: str,
        window_seconds: int,
    ) -> tuple[str, int]:
        return _storage_messages.get_recent_bot_reply_to_user(
            self,
            group_id,
            user_id,
            window_seconds,
        )

    def get_recent_activity_counts(
        self,
        group_id: str,
        human_window_seconds: int = 60,
        bot_window_seconds: int = 120,
    ) -> tuple[int, int]:
        return _storage_messages.get_recent_activity_counts(
            self,
            group_id,
            human_window_seconds=human_window_seconds,
            bot_window_seconds=bot_window_seconds,
        )

    def get_recent_image_descriptions(self, group_id: str, limit: int = 8) -> list[str]:
        return _storage_messages.get_recent_image_descriptions(self, group_id, limit=limit)

    def count_image_generation_usage(self, user_id: str, usage_date: str) -> int:
        return _storage_messages.count_image_generation_usage(self, user_id, usage_date)

    def record_image_generation_usage(
        self,
        group_id: str,
        user_id: str,
        usage_date: str,
        prompt: str,
        image_ref: str,
        created_at: int | None = None,
    ) -> None:
        _storage_messages.record_image_generation_usage(
            self,
            group_id,
            user_id,
            usage_date,
            prompt,
            image_ref,
            created_at=created_at,
        )
