from __future__ import annotations

import time
from datetime import datetime

from loguru import logger

from plugins.llm_group_bot.reply_sending import same_local_path
from qq_llm_bot.cognitive_agents import AgentPipeline
from qq_llm_bot.cognitive_storage import BotStorage
from qq_llm_bot.config import AppConfig
from qq_llm_bot.models import MessageContext, StickerAssetRecord, StickerCandidate
from qq_llm_bot.stickers import StickerLocalStore


class PluginMaintenance:
    def __init__(
        self,
        *,
        config: AppConfig,
        storage: BotStorage,
        pipeline: AgentPipeline,
        sticker_store: StickerLocalStore,
    ) -> None:
        self._config = config
        self._storage = storage
        self._pipeline = pipeline
        self._sticker_store = sticker_store

    def usage_date(self, now: int | None = None) -> str:
        timestamp = int(time.time() if now is None else now)
        return datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d")

    def cleanup_unused_stickers(self) -> None:
        if not self._config.stickers.enabled:
            return
        now = int(time.time())
        interval_seconds = self._config.stickers.cleanup_interval_hours * 60 * 60
        if not self._storage.claim_sticker_cleanup(interval_seconds, now=now):
            return
        unused_seconds = self._config.stickers.unused_ttl_hours * 60 * 60
        deleted_assets = self._storage.delete_unused_sticker_assets(unused_seconds, now=now)
        deleted_files = 0
        for asset in deleted_assets:
            if self._sticker_store.delete_saved_file(asset.local_path):
                deleted_files += 1
        if deleted_assets:
            logger.info(
                "Sticker cleanup deleted {} assets and {} files after {} unused hours",
                len(deleted_assets),
                deleted_files,
                self._config.stickers.unused_ttl_hours,
            )

    def record_outbound_sticker_sent(self, sticker: StickerAssetRecord) -> None:
        self._storage.record_sticker_sent(sticker.id, usage_date=self.usage_date())
        self.cleanup_unused_stickers()

    async def reflect_group(self, group_id: str) -> None:
        if not self._config.reflection.enabled:
            return
        if not self._storage.should_reflect(
            group_id,
            self._config.reflection.message_threshold,
            self._config.reflection.min_interval_seconds,
        ):
            return
        recent_messages = self._storage.get_recent_messages(group_id, self._config.reflection.recent_limit)
        prior_reflections = self._storage.list_memories("group", group_id, limit=3)
        reflection = await self._pipeline.reflect(group_id, recent_messages, prior_reflections)
        if reflection:
            self._storage.record_memory_candidates([reflection])

    async def update_profiles(self, user_ids: list[str], force: bool = False) -> None:
        seen: set[str] = set()
        for raw_user_id in user_ids:
            user_id = str(raw_user_id).strip()
            if not user_id or user_id in seen:
                continue
            seen.add(user_id)
            if not force and not self._storage.should_update_user_profile(
                user_id,
                self._config.facts.profile_fact_threshold,
            ):
                continue
            facts = self._storage.list_user_facts(user_id, limit=0)
            if force and not facts:
                self._storage.clear_user_profile(user_id)
                continue
            draft = await self._pipeline.profile(user_id, facts, self._storage.get_user_profile(user_id))
            if draft is None:
                continue
            self._storage.maybe_update_user_profile(user_id, draft, facts, force=force)

    async def record_sticker_candidates(
        self,
        context: MessageContext,
        candidates: list[StickerCandidate],
    ) -> None:
        if not self._config.stickers.enabled or not candidates:
            return
        for candidate in candidates:
            existing = self._storage.find_existing_sticker_asset(context.group_id, candidate)
            if existing is not None:
                self._storage.upsert_sticker_asset(
                    context,
                    candidate,
                    local_path=existing.local_path,
                    sha256=existing.sha256,
                )
                continue
            saved = await self._sticker_store.save_candidate(context, candidate)
            if saved is None:
                continue
            asset = self._storage.upsert_sticker_asset(
                context,
                candidate,
                local_path=saved.local_path,
                sha256=saved.sha256,
            )
            if asset is not None and not same_local_path(asset.local_path, saved.local_path):
                self._sticker_store.delete_saved_file(saved.local_path)
