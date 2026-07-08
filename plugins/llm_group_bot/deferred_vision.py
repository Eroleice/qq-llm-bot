from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from dataclasses import replace
from typing import Any

from loguru import logger

from qq_llm_bot.models import MessageAttachment, MessageContext


@dataclass
class _PendingVision:
    future: asyncio.Future[list[str]]
    created_at: float
    context: MessageContext
    worker: asyncio.Task[list[str]] | None = None


pending_vision_tasks: dict[tuple[str, str], _PendingVision] = {}
_PENDING_VISION_MAX_MESSAGES = 3
_PENDING_VISION_MAX_AGE_SECONDS = 120.0
_deferred_vision_lock: asyncio.Lock | None = None


config: Any = None
_record_deferred_vision_callback: Callable[[MessageContext], Awaitable[list[str]]] | None = None
_refresh_observation_buffer_image_descriptions_callback: Callable[[MessageContext, list[str]], None] | None = None


def configure(
    *,
    config_: Any,
    record_deferred_vision: Callable[[MessageContext], Awaitable[list[str]]],
    refresh_observation_buffer_image_descriptions: Callable[[MessageContext, list[str]], None],
) -> None:
    global config, _record_deferred_vision_callback
    global _refresh_observation_buffer_image_descriptions_callback
    config = config_
    _record_deferred_vision_callback = record_deferred_vision
    _refresh_observation_buffer_image_descriptions_callback = refresh_observation_buffer_image_descriptions


async def _record_deferred_vision(context: MessageContext) -> list[str]:
    if _record_deferred_vision_callback is None:  # pragma: no cover - setup invariant
        raise RuntimeError("deferred vision handlers are not configured")
    return await _record_deferred_vision_callback(context)


def _register_pending_vision(context: MessageContext) -> _PendingVision | None:
    if not _has_image_attachments(context):
        return None
    _cleanup_pending_vision_tasks()
    key = _pending_vision_key(context)
    existing = pending_vision_tasks.get(key)
    if existing is not None:
        return existing
    future: asyncio.Future[list[str]] = asyncio.get_running_loop().create_future()
    pending = _PendingVision(future=future, created_at=time.time(), context=context)
    pending_vision_tasks[key] = pending
    return pending


def _ensure_deferred_vision_task(context: MessageContext) -> asyncio.Task[list[str]] | None:
    pending = _register_pending_vision(context)
    if pending is None:
        return None
    if pending.worker is None or pending.worker.done():
        pending.worker = asyncio.create_task(_run_pending_deferred_vision(context))
    return pending.worker


async def _run_pending_deferred_vision(context: MessageContext) -> list[str]:
    descriptions: list[str] = []
    try:
        async with _get_deferred_vision_lock():
            descriptions = await _record_deferred_vision(context)
        _refresh_observation_buffer_image_descriptions(context, descriptions)
    except Exception as exc:  # pragma: no cover - pending vision must never block the group
        logger.warning(
            "Pending deferred image observation failed for group {} message {}: {}",
            context.group_id,
            context.message_id,
            exc,
        )
    finally:
        _finish_pending_vision(context, descriptions)
    return descriptions


def _get_deferred_vision_lock() -> asyncio.Lock:
    global _deferred_vision_lock
    if _deferred_vision_lock is None:
        _deferred_vision_lock = asyncio.Lock()
    return _deferred_vision_lock


def _finish_pending_vision(context: MessageContext, descriptions: list[str]) -> None:
    key = _pending_vision_key(context)
    pending = pending_vision_tasks.pop(key, None)
    if pending is None or pending.future.done():
        return
    pending.future.set_result(list(descriptions))


def _context_with_relevant_pending_images(context: MessageContext) -> MessageContext:
    if _has_image_attachments(context) or not _looks_like_image_context_followup(context):
        return context
    attachments: list[MessageAttachment] = []
    seen_urls = {
        attachment.url
        for attachment in context.attachments
        if attachment.attachment_type == "image" and attachment.url
    }
    for pending in _recent_pending_vision_tasks(context):
        for attachment in pending.context.attachments:
            if attachment.attachment_type != "image" or not attachment.url:
                continue
            if attachment.url in seen_urls:
                continue
            seen_urls.add(attachment.url)
            attachments.append(attachment)
            if len(attachments) >= config.vision.max_images_per_message:
                break
        if len(attachments) >= config.vision.max_images_per_message:
            break
    if not attachments:
        return context
    note = f"[相关未解析图片 x{len(attachments)}：来自最近群聊，当前消息可能在询问这些图片]"
    plain_text = "\n".join(part for part in (context.plain_text, note) if part).strip()
    return replace(context, plain_text=plain_text, attachments=[*context.attachments, *attachments])


def _refresh_observation_buffer_image_descriptions(
    context: MessageContext,
    descriptions: list[str],
) -> None:
    if _refresh_observation_buffer_image_descriptions_callback is None:  # pragma: no cover - setup invariant
        raise RuntimeError("deferred vision handlers are not configured")
    _refresh_observation_buffer_image_descriptions_callback(context, descriptions)


async def _wait_for_relevant_pending_vision(context: MessageContext) -> bool:
    if not _looks_like_image_context_followup(context):
        _cleanup_pending_vision_tasks()
        return True

    pending = _recent_pending_vision_tasks(context)
    if not pending:
        return True

    logger.info(
        "Using {} pending image analyses as inline multimodal context for group {} message {}",
        len(pending),
        context.group_id,
        context.message_id,
    )
    return True


def _recent_pending_vision_tasks(context: MessageContext) -> list[_PendingVision]:
    _cleanup_pending_vision_tasks()
    current_key = _pending_vision_key(context)
    candidates = [
        pending
        for key, pending in pending_vision_tasks.items()
        if key != current_key and key[0] == context.group_id and not pending.future.done()
    ]
    candidates.sort(
        key=lambda item: (
            item.context.user_id == context.user_id,
            item.created_at,
        ),
        reverse=True,
    )
    return candidates[:_PENDING_VISION_MAX_MESSAGES]


def _cleanup_pending_vision_tasks() -> None:
    now = time.time()
    for key, pending in list(pending_vision_tasks.items()):
        if pending.future.done() or now - pending.created_at > _PENDING_VISION_MAX_AGE_SECONDS:
            pending_vision_tasks.pop(key, None)


def _pending_vision_key(context: MessageContext) -> tuple[str, str]:
    return (context.group_id, context.message_id)


def _has_image_attachments(context: MessageContext) -> bool:
    return any(attachment.attachment_type == "image" for attachment in context.attachments)


def _looks_like_image_context_followup(context: MessageContext) -> bool:
    if _has_image_attachments(context):
        return False
    compact = "".join(context.plain_text.split()).lower()
    if not compact:
        return False
    explicit_image_terms = (
        "图",
        "图片",
        "截图",
        "照片",
        "新闻",
        "image",
        "screenshot",
    )
    generic_reference_terms = (
        "这事",
        "这个事",
        "这件事",
        "这个事情",
        "这条",
        "这个",
        "这",
    )
    question_terms = (
        "发生什么",
        "发生啥",
        "啥情况",
        "什么情况",
        "怎么回事",
        "怎么看",
        "看法",
        "评价",
        "解释",
        "总结",
        "真的假的",
        "是真的吗",
    )
    return any(term in compact for term in explicit_image_terms) or (
        any(term in compact for term in generic_reference_terms)
        and any(term in compact for term in question_terms)
    )


register_pending_vision = _register_pending_vision
ensure_deferred_vision_task = _ensure_deferred_vision_task
context_with_relevant_pending_images = _context_with_relevant_pending_images
wait_for_relevant_pending_vision = _wait_for_relevant_pending_vision
