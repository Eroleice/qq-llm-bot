from __future__ import annotations

import asyncio
import time
from dataclasses import replace
from typing import Any

from loguru import logger

from plugins.llm_group_bot import deferred_vision as _deferred_vision
from qq_llm_bot.config import ParticipationMode
from qq_llm_bot.models import MessageContext, ParticipationDecision, RelationDelta


class ObservationBatchCoordinator:
    def __init__(
        self,
        *,
        config: Any,
        storage: Any,
        pipeline: Any,
        maintenance: Any,
    ) -> None:
        self.config = config
        self.storage = storage
        self.pipeline = pipeline
        self.maintenance = maintenance
        self.observation_buffers: dict[str, list[MessageContext]] = {}
        self.observation_last_flush_at: dict[str, int] = {}

    def register_pending_vision(self, context: MessageContext) -> Any:
        return _deferred_vision.register_pending_vision(context)

    def ensure_deferred_vision_task(self, context: MessageContext) -> asyncio.Task[list[str]] | None:
        return _deferred_vision.ensure_deferred_vision_task(context)

    def context_with_relevant_pending_images(self, context: MessageContext) -> MessageContext:
        return _deferred_vision.context_with_relevant_pending_images(context)

    async def wait_for_relevant_pending_vision(self, context: MessageContext) -> bool:
        return await _deferred_vision.wait_for_relevant_pending_vision(context)

    def should_defer_realtime_pipeline(self, context: MessageContext, mode: ParticipationMode) -> bool:
        if not self.config.observation_batch.enabled:
            return False
        if context.is_direct:
            return False
        if context.bot_mentioned:
            return False
        if self._has_recent_bot_reply_context(context):
            return False
        if mode == "silent":
            return True
        if mode == "passive":
            return True
        if mode == "active":
            return not self._looks_like_active_realtime_candidate(context)
        return False

    def _has_recent_bot_reply_context(self, context: MessageContext) -> bool:
        text = context.plain_text.strip()
        if not text or len(text) > 160:
            return False
        recent_reply, _ = self.storage.get_recent_bot_reply_to_user(
            context.group_id,
            context.user_id,
            self.config.bot.interaction_followup_seconds,
        )
        return bool(recent_reply)

    def _looks_like_active_realtime_candidate(self, context: MessageContext) -> bool:
        text = " ".join(context.plain_text.split())
        compact = "".join(text.split())
        if len(compact) < 6:
            return False
        realtime_cues = (
            "?",
            "？",
            "吗",
            "么",
            "怎么",
            "如何",
            "为什么",
            "咋",
            "要不要",
            "该不该",
            "有没有",
            "谁",
            "哪",
            "啥",
            "求",
            "建议",
            "推荐",
            "帮",
            "分析",
            "看法",
            "方案",
        )
        return any(cue in compact for cue in realtime_cues)
    async def defer_observation(self, context: MessageContext, mode: ParticipationMode) -> None:
        self.ensure_deferred_vision_task(context)
        image_descriptions = [
            attachment.summary
            for attachment in context.attachments
            if attachment.attachment_type == "image" and attachment.summary
        ]
        buffer = self.observation_buffers.setdefault(context.group_id, [])
        if context.group_id not in self.observation_last_flush_at:
            self.observation_last_flush_at[context.group_id] = int(time.time())
        buffer.append(self._context_with_deferred_image_descriptions(context, image_descriptions))
        self.storage.apply_relationship_delta(
            context.group_id,
            context.user_id,
            RelationDelta(familiarity=1, reason="deferred batch observation"),
        )
        self.storage.record_decision(
            context,
            ParticipationDecision("observe", self._deferred_observation_reason(mode), mode, 0.0),
            "",
        )
        await self.flush_observation_batch(context.group_id)

    async def record_deferred_vision(self, context: MessageContext) -> list[str]:
        if not context.attachments:
            return []
        try:
            vision = await self.pipeline.observe_vision(context)
        except Exception as exc:  # pragma: no cover - image observation must never break chat handling
            logger.warning(
                "Deferred image observation failed for group {} message {}: {}",
                context.group_id,
                context.message_id,
                exc,
            )
            return []
    
        image_descriptions = list(vision.attachment_descriptions or tuple(vision.descriptions))
        self.storage.update_image_descriptions(context.group_id, context.message_id, image_descriptions)
        await self.maintenance.record_sticker_candidates(context, list(vision.sticker_candidates))
        fact_write = self.storage.record_fact_candidates(list(vision.fact_candidates))
        memory_write = self.storage.record_memory_candidates(list(vision.memory_candidates))
        if memory_write.conflicts:
            logger.info(
                "Deferred image observation recorded {} memory conflicts in group {}; waiting for manual review",
                len(memory_write.conflicts),
                context.group_id,
            )
        await self.maintenance.update_profiles(
            [fact.subject_user_id for fact in fact_write.accepted],
            force=False,
        )
        return image_descriptions

    def _context_with_deferred_image_descriptions(
        self,
        context: MessageContext,
        image_descriptions: list[str],
    ) -> MessageContext:
        descriptions = [description.strip() for description in image_descriptions if description.strip()]
        if not descriptions:
            return context
        lines = []
        if context.plain_text:
            lines.append(context.plain_text)
        lines.extend(f"[图片解读] {description}" for description in descriptions)
        return replace(context, plain_text="\n".join(lines))

    def refresh_observation_buffer_image_descriptions(
        self,
        context: MessageContext,
        descriptions: list[str],
    ) -> None:
        if not descriptions:
            return
        buffer = self.observation_buffers.get(context.group_id)
        if not buffer:
            return
        for index, buffered in enumerate(buffer):
            if buffered.message_id == context.message_id:
                buffer[index] = self._context_with_deferred_image_descriptions(buffered, descriptions)
                return
    def _deferred_observation_reason(self, mode: ParticipationMode) -> str:
        if mode == "silent":
            return "silent mode deferred to batch observation"
        if mode == "passive":
            return "passive non-direct message deferred to batch observation"
        return "active low-priority message deferred to batch observation"

    async def flush_observation_batch(self, group_id: str, force: bool = False) -> None:
        if not self.config.observation_batch.enabled:
            return
        buffer = self.observation_buffers.get(group_id, [])
        if not buffer:
            return
    
        now = int(time.time())
        last_flush = self.observation_last_flush_at.get(group_id, now)
        due_by_size = len(buffer) >= self.config.observation_batch.batch_size
        due_by_time = now - last_flush >= self.config.observation_batch.max_interval_seconds
        if not force and not due_by_size and not due_by_time:
            return
    
        batch_size = min(len(buffer), self.config.observation_batch.max_messages_per_batch)
        batch = list(buffer[:batch_size])
        try:
            result = await self.pipeline.observe_batch(
                group_id,
                batch,
                self.storage.list_memories("group", group_id, limit=3),
                self.storage.list_group_lexicon_records(group_id, limit=10),
            )
        except Exception as exc:  # pragma: no cover - batch observation must never break chat handling
            self.observation_last_flush_at[group_id] = now
            logger.warning("Batch observation failed for group {}: {}", group_id, exc)
            return
    
        del buffer[:batch_size]
        if not buffer:
            self.observation_buffers.pop(group_id, None)
        self.observation_last_flush_at[group_id] = now
    
        memories = list(result.memories)
        if result.reflection is not None:
            memories.append(result.reflection)
        fact_write = self.storage.record_fact_candidates(result.facts)
        memory_write = self.storage.record_memory_candidates(memories)
        if memory_write.conflicts:
            logger.info(
                "Batch observation recorded {} memory conflicts in group {}; waiting for manual review",
                len(memory_write.conflicts),
                group_id,
            )
        await self.maintenance.update_profiles(
            [fact.subject_user_id for fact in fact_write.accepted],
            force=False,
        )
