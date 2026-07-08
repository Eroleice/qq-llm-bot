from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass
from typing import Any

from loguru import logger
from nonebot.adapters.onebot.v11 import Bot

from qq_llm_bot.config import ParticipationMode
from qq_llm_bot.models import MessageContext
from qq_llm_bot.realtime_merge import merge_realtime_contexts


@dataclass
class _PendingRealtimeReply:
    group_id: str
    user_id: str
    mode: ParticipationMode
    contexts: list[MessageContext]
    first_message_at: float
    updated_at: float
    generation: int = 1
    committing: bool = False
    task: asyncio.Task[None] | None = None


config: Any = None
_process_group_context_callback: Callable[..., Awaitable[bool]] | None = None
_should_defer_realtime_pipeline_callback: Callable[[MessageContext, ParticipationMode], bool] | None = None
_pending_realtime_replies: dict[tuple[str, str], _PendingRealtimeReply] = {}
_pending_realtime_lock = asyncio.Lock()


def configure(
    *,
    config_: Any,
    process_group_context: Callable[..., Awaitable[bool]],
    should_defer_realtime_pipeline: Callable[[MessageContext, ParticipationMode], bool],
) -> None:
    global config, _process_group_context_callback, _should_defer_realtime_pipeline_callback
    config = config_
    _process_group_context_callback = process_group_context
    _should_defer_realtime_pipeline_callback = should_defer_realtime_pipeline


async def maybe_enqueue_realtime_reply(
    bot: Bot,
    context: MessageContext,
    mode: ParticipationMode,
) -> bool:
    if config is None:  # pragma: no cover - plugin setup invariant
        raise RuntimeError("realtime reply handlers are not configured")
    if not config.bot.realtime_merge_enabled:
        return False

    key = _realtime_reply_key(context)
    now = time.time()
    async with _pending_realtime_lock:
        pending = _pending_realtime_replies.get(key)
        if pending is not None and _can_append_realtime_context(pending, context, now):
            pending.contexts.append(context)
            pending.mode = mode
            pending.updated_at = now
            pending.generation += 1
            _cancel_realtime_task(pending)
            pending.task = asyncio.create_task(
                _run_pending_realtime_reply(bot, key, pending.generation)
            )
            logger.info(
                "Merged realtime reply context group={} user={} messages={} generation={}",
                context.group_id,
                context.user_id,
                len(pending.contexts),
                pending.generation,
            )
            return True

        if pending is not None:
            return False

        if _should_defer_realtime_pipeline(context, mode):
            return False

        pending = _PendingRealtimeReply(
            group_id=context.group_id,
            user_id=context.user_id,
            mode=mode,
            contexts=[context],
            first_message_at=now,
            updated_at=now,
        )
        pending.task = asyncio.create_task(_run_pending_realtime_reply(bot, key, pending.generation))
        _pending_realtime_replies[key] = pending
        return True


async def _run_pending_realtime_reply(
    bot: Bot,
    key: tuple[str, str],
    generation: int,
) -> None:
    try:
        pending = await _get_pending_realtime_reply(key, generation)
        if pending is None:
            return
        contexts = tuple(pending.contexts)
        context = merge_realtime_contexts(contexts)

        async def _commit_guard() -> bool:
            await _wait_realtime_grace(key, generation)
            return await _claim_pending_realtime_reply(key, generation)

        await _process_group_context(
            bot,
            context,
            pending.mode,
            source_contexts=contexts,
            commit_guard=_commit_guard,
        )
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # pragma: no cover - realtime task must not break the matcher loop
        logger.exception("Realtime reply task failed for {}: {}", key, exc)
    finally:
        await _discard_pending_realtime_reply(key, generation)


async def _get_pending_realtime_reply(
    key: tuple[str, str],
    generation: int,
) -> _PendingRealtimeReply | None:
    async with _pending_realtime_lock:
        pending = _pending_realtime_replies.get(key)
        if pending is None or pending.generation != generation:
            return None
        return pending


async def _wait_realtime_grace(key: tuple[str, str], generation: int) -> None:
    grace_seconds = max(0.0, float(config.bot.realtime_merge_grace_seconds))
    while grace_seconds > 0:
        pending = await _get_pending_realtime_reply(key, generation)
        if pending is None:
            return
        max_window = max(0.0, float(config.bot.realtime_merge_max_window_seconds))
        deadline = min(pending.updated_at + grace_seconds, pending.first_message_at + max_window)
        remaining = deadline - time.time()
        if remaining <= 0:
            return
        await asyncio.sleep(min(remaining, 0.25))


async def _claim_pending_realtime_reply(key: tuple[str, str], generation: int) -> bool:
    async with _pending_realtime_lock:
        pending = _pending_realtime_replies.get(key)
        if pending is None or pending.generation != generation or pending.committing:
            return False
        pending.committing = True
        return True


async def _discard_pending_realtime_reply(key: tuple[str, str], generation: int) -> None:
    async with _pending_realtime_lock:
        pending = _pending_realtime_replies.get(key)
        if pending is not None and pending.generation == generation:
            _pending_realtime_replies.pop(key, None)


async def stop_realtime_reply_workers() -> None:
    async with _pending_realtime_lock:
        tasks = [
            pending.task
            for pending in _pending_realtime_replies.values()
            if pending.task is not None and not pending.task.done()
        ]
        _pending_realtime_replies.clear()
    for task in tasks:
        task.cancel()
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)


def _can_append_realtime_context(
    pending: _PendingRealtimeReply,
    context: MessageContext,
    now: float,
) -> bool:
    if pending.committing:
        return False
    if len(pending.contexts) >= config.bot.realtime_merge_max_messages:
        return False
    if now - pending.first_message_at > config.bot.realtime_merge_max_window_seconds:
        return False
    return (
        pending.group_id == context.group_id
        and pending.user_id == context.user_id
    )


def _cancel_realtime_task(pending: _PendingRealtimeReply) -> None:
    if pending.task is not None and not pending.task.done():
        pending.task.cancel()


def _realtime_reply_key(context: MessageContext) -> tuple[str, str]:
    return (context.group_id, context.user_id)


async def _process_group_context(
    bot: Bot,
    context: MessageContext,
    mode: ParticipationMode,
    *,
    source_contexts: Iterable[MessageContext] | None = None,
    commit_guard: Any | None = None,
) -> bool:
    if _process_group_context_callback is None:  # pragma: no cover - plugin setup invariant
        raise RuntimeError("realtime reply handlers are not configured")
    return await _process_group_context_callback(
        bot,
        context,
        mode,
        source_contexts=source_contexts,
        commit_guard=commit_guard,
    )


def _should_defer_realtime_pipeline(
    context: MessageContext,
    mode: ParticipationMode,
) -> bool:
    if _should_defer_realtime_pipeline_callback is None:  # pragma: no cover - plugin setup invariant
        raise RuntimeError("realtime reply handlers are not configured")
    return _should_defer_realtime_pipeline_callback(context, mode)
