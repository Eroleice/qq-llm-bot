from __future__ import annotations

import asyncio
import contextlib
import time
from itertools import count

from loguru import logger
from nonebot.adapters.onebot.v11 import Bot
from nonebot.adapters.onebot.v11.exception import ActionFailed

from qq_llm_bot.config import BotConfig
from qq_llm_bot.models import StickerAssetRecord
from qq_llm_bot.outbound_queue_errors import (
    is_retryable_send_error,
    onebot_group_id,
    send_error_detail,
)
from qq_llm_bot.outbound_queue_models import (
    BotsProvider,
    QueuedSendAttempt,
    QueuedSendStatus,
    StickerSentCallback,
    _QueuedOutboundMessage,
)


class OutboundGroupSendQueue:
    def __init__(
        self,
        bot_config: BotConfig,
        *,
        on_sticker_sent: StickerSentCallback | None = None,
        retry_interval_seconds: float = 3.0,
    ) -> None:
        self._config = bot_config
        self._on_sticker_sent = on_sticker_sent
        self._retry_interval_seconds = retry_interval_seconds
        self._queue_ids = count(1)
        self._queue: list[_QueuedOutboundMessage] = []
        self._queue_lock = asyncio.Lock()
        self._flush_lock = asyncio.Lock()
        self._retry_worker: asyncio.Task[None] | None = None

    def start_retry_worker(self, bots_provider: BotsProvider) -> None:
        if not self._config.send_retry_enabled:
            return
        if self._retry_worker is not None and not self._retry_worker.done():
            return
        self._retry_worker = asyncio.create_task(self._retry_loop(bots_provider))

    async def stop_retry_worker(self) -> None:
        task = self._retry_worker
        self._retry_worker = None
        if task is None or task.done():
            return
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    async def queue_size(self, bot_self_id: str | None = None) -> int:
        async with self._queue_lock:
            if bot_self_id is None:
                return len(self._queue)
            return sum(1 for item in self._queue if item.bot_self_id == str(bot_self_id))

    async def queue_group_attempts(
        self,
        bot: Bot,
        group_id: str,
        send_attempts: tuple[QueuedSendAttempt, ...],
        *,
        source: str,
        reason: str,
    ) -> bool:
        if not self._config.send_retry_enabled or not send_attempts:
            return False
        now = time.time()
        async with self._queue_lock:
            self._drop_expired_locked(now)
            while len(self._queue) >= self._config.send_retry_queue_limit:
                dropped = self._queue.pop(0)
                logger.warning(
                    "Dropping queued outbound message #{} for bot {} group {}: queue limit reached",
                    dropped.id,
                    dropped.bot_self_id,
                    dropped.group_id,
                )
            queued = _QueuedOutboundMessage(
                id=next(self._queue_ids),
                bot_self_id=str(bot.self_id),
                group_id=str(group_id),
                send_attempts=send_attempts,
                created_at=now,
                next_attempt_at=now,
                source=source,
                reason=reason,
            )
            self._queue.append(queued)
            logger.warning(
                "Queued outbound message #{} for bot {} group {} from {} after {} (queue_size={})",
                queued.id,
                queued.bot_self_id,
                queued.group_id,
                source,
                reason,
                len(self._queue),
            )
        return True

    async def flush(self, bot: Bot, reason: str) -> None:
        if not self._config.send_retry_enabled:
            return
        async with self._flush_lock:
            while True:
                queued = await self._pop_next_due(bot.self_id)
                if queued is None:
                    return
                try:
                    status, sticker, error = await self._try_send_group_message(bot, queued)
                except Exception as exc:
                    logger.warning(
                        "Dropping queued outbound message #{} after unexpected send error: {}",
                        queued.id,
                        exc,
                    )
                    continue
                if status == "sent":
                    logger.info(
                        "Flushed queued outbound message #{} for bot {} group {} via {}",
                        queued.id,
                        queued.bot_self_id,
                        queued.group_id,
                        reason,
                    )
                    if sticker is not None and self._on_sticker_sent is not None:
                        self._on_sticker_sent(sticker)
                    continue
                if status == "retry":
                    await self._requeue(queued, error)
                    return

    async def _retry_loop(self, bots_provider: BotsProvider) -> None:
        while True:
            await asyncio.sleep(self._retry_interval_seconds)
            for bot in list(bots_provider()):
                if isinstance(bot, Bot):
                    await self.flush(bot, "retry worker")

    async def _pop_next_due(self, bot_self_id: str) -> _QueuedOutboundMessage | None:
        now = time.time()
        async with self._queue_lock:
            self._drop_expired_locked(now)
            for index, queued in enumerate(self._queue):
                if queued.bot_self_id == str(bot_self_id) and queued.next_attempt_at <= now:
                    return self._queue.pop(index)
        return None

    async def _try_send_group_message(
        self,
        bot: Bot,
        queued: _QueuedOutboundMessage,
    ) -> tuple[QueuedSendStatus, StickerAssetRecord | None, BaseException | None]:
        group_id = onebot_group_id(queued.group_id)
        if group_id is None:
            logger.warning(
                "Dropping queued outbound message #{}: invalid group id {}",
                queued.id,
                queued.group_id,
            )
            return ("drop", None, None)

        first_action_error: ActionFailed | None = None
        for attempt in queued.send_attempts:
            try:
                await bot.send_group_msg(group_id=group_id, message=attempt.message)
            except ActionFailed as exc:
                if first_action_error is None:
                    first_action_error = exc
                logger.warning(
                    "Queued outbound message #{} action attempt failed for group {}: {}",
                    queued.id,
                    queued.group_id,
                    exc,
                )
                continue
            except Exception as exc:
                if should_queue_send_error(exc):
                    return ("retry", None, exc)
                raise
            return ("sent", attempt.sticker, None)

        logger.warning(
            "Dropping queued outbound message #{} after all action attempts failed: {}",
            queued.id,
            first_action_error,
        )
        return ("drop", None, first_action_error)

    async def _requeue(
        self,
        queued: _QueuedOutboundMessage,
        error: BaseException | None,
    ) -> None:
        now = time.time()
        queued.attempts += 1
        queued.reason = send_error_detail(error) if error is not None else queued.reason
        if (
            queued.attempts >= self._config.send_retry_max_attempts
            or now - queued.created_at > self._config.send_retry_max_age_seconds
        ):
            logger.warning(
                "Dropping queued outbound message #{} for bot {} group {} after {} attempts: {}",
                queued.id,
                queued.bot_self_id,
                queued.group_id,
                queued.attempts,
                queued.reason,
            )
            return
        queued.next_attempt_at = now + self._retry_delay(queued.attempts)
        async with self._queue_lock:
            self._queue.append(queued)
        logger.warning(
            "Queued outbound message #{} retry {}/{} in {:.1f}s: {}",
            queued.id,
            queued.attempts,
            self._config.send_retry_max_attempts,
            queued.next_attempt_at - now,
            queued.reason,
        )

    def _drop_expired_locked(self, now: float) -> None:
        kept: list[_QueuedOutboundMessage] = []
        for queued in self._queue:
            expired = now - queued.created_at > self._config.send_retry_max_age_seconds
            exhausted = queued.attempts >= self._config.send_retry_max_attempts
            if expired or exhausted:
                logger.warning(
                    "Dropping queued outbound message #{} for bot {} group {}: {}",
                    queued.id,
                    queued.bot_self_id,
                    queued.group_id,
                    "expired" if expired else "attempts exhausted",
                )
                continue
            kept.append(queued)
        self._queue[:] = kept

    def _retry_delay(self, attempts: int) -> float:
        base = self._config.send_retry_base_delay_seconds
        maximum = self._config.send_retry_max_delay_seconds
        return min(maximum, base * (2 ** max(0, attempts - 1)))


def should_queue_send_error(exc: BaseException) -> bool:
    return is_retryable_send_error(exc)
