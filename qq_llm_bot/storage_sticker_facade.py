from __future__ import annotations

import sqlite3

import qq_llm_bot.storage_stickers as _storage_stickers
from qq_llm_bot.models import MessageContext, StickerAssetRecord, StickerCandidate


class StorageStickerFacadeMixin:
    def upsert_sticker_asset(
        self,
        context: MessageContext,
        candidate: StickerCandidate,
        local_path: str,
        sha256: str = "",
    ) -> StickerAssetRecord | None:
        return _storage_stickers.upsert_sticker_asset(
            self,
            context,
            candidate,
            local_path,
            sha256=sha256,
        )

    def find_existing_sticker_asset(
        self,
        group_id: str,
        candidate: StickerCandidate,
    ) -> StickerAssetRecord | None:
        return _storage_stickers.find_existing_sticker_asset(self, group_id, candidate)

    def list_sticker_assets(
        self,
        group_id: str,
        limit: int = 24,
        enabled_only: bool = True,
    ) -> list[StickerAssetRecord]:
        return _storage_stickers.list_sticker_assets(
            self,
            group_id,
            limit=limit,
            enabled_only=enabled_only,
        )

    def list_stickers(self, group_id: str, limit: int = 20) -> list[str]:
        return _storage_stickers.list_stickers(self, group_id, limit=limit)

    def get_sticker_asset(self, sticker_id: int) -> StickerAssetRecord | None:
        return _storage_stickers.get_sticker_asset(self, sticker_id)

    def set_sticker_enabled(self, sticker_id: int, enabled: bool) -> bool:
        return _storage_stickers.set_sticker_enabled(self, sticker_id, enabled)

    def delete_sticker_asset(self, sticker_id: int) -> StickerAssetRecord | None:
        return _storage_stickers.delete_sticker_asset(self, sticker_id)

    def record_sticker_sent(
        self,
        sticker_id: int,
        usage_date: str = "",
        sent_at: int | None = None,
    ) -> None:
        _storage_stickers.record_sticker_sent(
            self,
            sticker_id,
            usage_date=usage_date,
            sent_at=sent_at,
        )

    def count_sticker_usage(self, sticker_id: int, usage_date: str = "") -> int:
        return _storage_stickers.count_sticker_usage(self, sticker_id, usage_date=usage_date)

    def list_sticker_usage_daily(
        self,
        group_id: str = "",
        usage_date: str = "",
        limit: int = 200,
    ) -> list[dict[str, object]]:
        return _storage_stickers.list_sticker_usage_daily(
            self,
            group_id=group_id,
            usage_date=usage_date,
            limit=limit,
        )

    def claim_sticker_cleanup(self, interval_seconds: int, now: int | None = None) -> bool:
        return _storage_stickers.claim_sticker_cleanup(self, interval_seconds, now=now)

    def delete_unused_sticker_assets(
        self,
        unused_seconds: int,
        now: int | None = None,
    ) -> list[StickerAssetRecord]:
        return _storage_stickers.delete_unused_sticker_assets(
            self,
            unused_seconds,
            now=now,
        )

    def _find_sticker_asset(
        self,
        conn: sqlite3.Connection,
        group_id: str,
        sha256: str = "",
        url: str = "",
    ) -> sqlite3.Row | None:
        return _storage_stickers.find_sticker_asset_row(conn, group_id, sha256=sha256, url=url)

    def _find_similar_sticker_asset(
        self,
        conn: sqlite3.Connection,
        group_id: str,
        candidate: StickerCandidate,
    ) -> sqlite3.Row | None:
        return _storage_stickers.find_similar_sticker_asset_row(conn, group_id, candidate)
