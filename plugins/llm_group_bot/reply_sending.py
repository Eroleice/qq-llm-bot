from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Iterable

from loguru import logger
from nonebot.adapters.onebot.v11 import Bot, Message, MessageSegment
from nonebot.adapters.onebot.v11.exception import ActionFailed

from qq_llm_bot.config import BotConfig
from qq_llm_bot.models import MessageContext, ParticipationDecision, StickerAssetRecord
from qq_llm_bot.onebot_messages import parse_outgoing_mention_parts
from qq_llm_bot.outbound_queue import (
    OutboundGroupSendQueue,
    QueuedSendAttempt,
    onebot_group_id,
    send_error_detail,
    should_queue_send_error,
)
from qq_llm_bot.reply_style import settings_from_bot_config, split_reply_bubbles, style_reply_text
from qq_llm_bot.stickers import sticker_file_ref, sticker_file_refs


@dataclass(frozen=True)
class ReplySendResult:
    parts: tuple[str, ...]
    sticker: StickerAssetRecord | None = None
    queued: bool = False


@dataclass(frozen=True)
class SingleReplySendResult:
    sticker: StickerAssetRecord | None = None
    queued: bool = False


class ReplySender:
    def __init__(
        self,
        *,
        bot_config: BotConfig,
        outbound_queue: OutboundGroupSendQueue,
        fallback_sender: Callable[[Message | str], Awaitable[Any]],
        recent_bot_reply_texts: Callable[[str, int], list[str]],
    ) -> None:
        self._bot_config = bot_config
        self._outbound_queue = outbound_queue
        self._fallback_sender = fallback_sender
        self._recent_bot_reply_texts = recent_bot_reply_texts

    async def send_group_reply(
        self,
        reply: str,
        sticker: StickerAssetRecord | None,
        reply_to_message_id: str | None,
        *,
        bot: Bot | None = None,
        context: MessageContext | None = None,
        decision: ParticipationDecision | None = None,
        allow_bubbles: bool = True,
    ) -> ReplySendResult:
        parts = self._prepare_reply_parts(reply, context, decision, allow_bubbles)
        if not parts and sticker is None:
            return ReplySendResult(())

        delay_seconds = self._bot_config.reply_bubble_delay_seconds if allow_bubbles else 0
        queued = False
        sent_sticker: StickerAssetRecord | None = None
        if len(parts) > 1:
            for index, part in enumerate(parts[:-1]):
                first_reply_to = reply_to_message_id if index == 0 else None
                result = await self._send_single_reply(
                    part,
                    None,
                    first_reply_to,
                    bot=bot,
                    context=context,
                )
                queued = queued or result.queued
                if result.queued:
                    queued_sticker = await self._queue_remaining_reply_parts(
                        parts[index + 1 :],
                        sticker,
                        bot=bot,
                        context=context,
                    )
                    return ReplySendResult(parts, result.sticker or queued_sticker, queued=True)
                if delay_seconds > 0:
                    await asyncio.sleep(delay_seconds)
            last_reply_to = None
            last_part = parts[-1]
        else:
            last_reply_to = reply_to_message_id
            last_part = parts[0] if parts else ""

        result = await self._send_single_reply(last_part, None, last_reply_to, bot=bot, context=context)
        queued = queued or result.queued
        if result.queued:
            queued_sticker = await self._queue_reply_sticker(
                sticker,
                bot=bot,
                context=context,
                source="group reply sticker",
                reason="previous reply text queued",
            )
            return ReplySendResult(parts, queued_sticker, queued=True)

        sticker_result = await self._send_reply_sticker(sticker, bot=bot, context=context)
        sent_sticker = sticker_result.sticker
        queued = queued or sticker_result.queued
        return ReplySendResult(parts, sent_sticker, queued)

    async def _send_single_reply(
        self,
        reply: str,
        sticker: StickerAssetRecord | None,
        reply_to_message_id: str | None,
        *,
        bot: Bot | None,
        context: MessageContext | None,
    ) -> SingleReplySendResult:
        first_error: ActionFailed | None = None
        attempts = _reply_send_attempts(
            reply,
            sticker,
            reply_to_message_id,
        )
        for index, (message, attempted_sticker, used_reply) in enumerate(attempts):
            sticker_included = message_contains_image(message)
            try:
                await self._send_group_message(message, bot=bot, context=context)
            except ActionFailed as exc:
                if first_error is None:
                    first_error = exc
                _log_reply_send_failure(
                    exc,
                    attempted_sticker if sticker_included else None,
                    used_reply,
                    reply_to_message_id,
                )
                continue
            except Exception as exc:
                if not should_queue_send_error(exc):
                    raise
                queued_sticker = attempted_sticker if sticker_included else None
                queued = await self._queue_reply_attempts(
                    attempts[index:],
                    bot=bot,
                    context=context,
                    source="group reply",
                    reason=send_error_detail(exc),
                )
                if queued:
                    logger.warning(
                        "Group reply send queued for group {} message {} after transient failure: {}",
                        context.group_id if context else "",
                        context.message_id if context else "",
                        exc,
                    )
                    return SingleReplySendResult(queued_sticker, queued=True)
                raise
            return SingleReplySendResult(attempted_sticker if sticker_included else None)
        if first_error is not None:
            raise first_error
        return SingleReplySendResult()

    async def _send_group_message(
        self,
        message: Message | str,
        *,
        bot: Bot | None,
        context: MessageContext | None,
    ) -> None:
        group_id = onebot_group_id(context.group_id) if context is not None else None
        if bot is not None and group_id is not None:
            await bot.send_group_msg(group_id=group_id, message=message)
            return
        await self._fallback_sender(message)

    async def _queue_remaining_reply_parts(
        self,
        remaining_parts: tuple[str, ...],
        sticker: StickerAssetRecord | None,
        *,
        bot: Bot | None,
        context: MessageContext | None,
    ) -> StickerAssetRecord | None:
        if not remaining_parts:
            return None
        queued_sticker: StickerAssetRecord | None = None
        for part in remaining_parts:
            await self._queue_reply_attempts(
                _reply_send_attempts(part, None, None),
                bot=bot,
                context=context,
                source="group reply bubble",
                reason="previous bubble queued",
            )
        if sticker is not None:
            queued_sticker = await self._queue_reply_sticker(
                sticker,
                bot=bot,
                context=context,
                source="group reply sticker",
                reason="previous bubble queued",
            )
        return queued_sticker

    async def _send_reply_sticker(
        self,
        sticker: StickerAssetRecord | None,
        *,
        bot: Bot | None,
        context: MessageContext | None,
    ) -> SingleReplySendResult:
        if sticker is None:
            return SingleReplySendResult()
        try:
            return await self._send_single_reply("", sticker, None, bot=bot, context=context)
        except ActionFailed as exc:
            _log_reply_send_failure(exc, sticker, False, None)
            return SingleReplySendResult()

    async def _queue_reply_sticker(
        self,
        sticker: StickerAssetRecord | None,
        *,
        bot: Bot | None,
        context: MessageContext | None,
        source: str,
        reason: str,
    ) -> StickerAssetRecord | None:
        if sticker is None:
            return None
        queued = await self._queue_reply_attempts(
            _reply_send_attempts("", sticker, None),
            bot=bot,
            context=context,
            source=source,
            reason=reason,
        )
        return sticker if queued else None

    async def _queue_reply_attempts(
        self,
        attempts: list[tuple[Message | str, StickerAssetRecord | None, bool]],
        *,
        bot: Bot | None,
        context: MessageContext | None,
        source: str,
        reason: str,
    ) -> bool:
        if bot is None or context is None:
            return False
        queued_attempts = tuple(
            QueuedSendAttempt(message, attempted_sticker if message_contains_image(message) else None)
            for message, attempted_sticker, _ in attempts
        )
        return await self._outbound_queue.queue_group_attempts(
            bot,
            context.group_id,
            queued_attempts,
            source=source,
            reason=reason,
        )

    def _prepare_reply_parts(
        self,
        reply: str,
        context: MessageContext | None,
        decision: ParticipationDecision | None,
        allow_bubbles: bool,
    ) -> tuple[str, ...]:
        text = str(reply or "").strip()
        if not text:
            return ()
        if not allow_bubbles:
            return (text,)

        settings = settings_from_bot_config(self._bot_config)
        if context is not None and decision is not None:
            recent_replies = self._recent_bot_reply_texts(
                context.group_id,
                self._bot_config.reply_emoji_cooldown_messages,
            )
            text = style_reply_text(
                text,
                settings,
                action=decision.action,
                value_type=decision.value_type,
                trigger_text=context.plain_text,
                recent_bot_replies=recent_replies,
            )
        return split_reply_bubbles(text, settings)


def _reply_send_attempts(
    reply: str,
    sticker: StickerAssetRecord | None,
    reply_to_message_id: str | None,
) -> list[tuple[Message | str, StickerAssetRecord | None, bool]]:
    reply_to = str(reply_to_message_id or "").strip() or None
    sticker_refs = _sticker_file_refs(sticker) if sticker is not None else ()
    requested: list[tuple[str | None, StickerAssetRecord | None, str]] = []
    if reply_to:
        if sticker is not None:
            requested.extend((reply_to, sticker, file_ref) for file_ref in sticker_refs)
            requested.append((reply_to, None, ""))
            requested.extend((None, sticker, file_ref) for file_ref in sticker_refs)
        else:
            requested.append((reply_to, None, ""))
        requested.append((None, None, ""))
    else:
        if sticker is not None:
            requested.extend((None, sticker, file_ref) for file_ref in sticker_refs)
            requested.append((None, None, ""))
        else:
            requested.append((None, None, ""))

    attempts: list[tuple[Message | str, StickerAssetRecord | None, bool]] = []
    seen: set[tuple[bool, int | None, str]] = set()
    for attempt_reply_to, attempt_sticker, attempt_file_ref in requested:
        if not reply and attempt_sticker is None:
            continue
        key = (
            bool(attempt_reply_to),
            attempt_sticker.id if attempt_sticker is not None else None,
            attempt_file_ref if attempt_sticker is not None else "",
        )
        if key in seen:
            continue
        seen.add(key)
        attempts.append(
            (
                _reply_message(reply, attempt_sticker, attempt_reply_to, attempt_file_ref),
                attempt_sticker,
                bool(attempt_reply_to),
            )
        )
    return attempts


def _reply_message(
    reply: str,
    sticker: StickerAssetRecord | None,
    reply_to_message_id: str | None = None,
    file_ref: str = "",
) -> Message | str:
    text_message = _reply_text_message(reply)
    if sticker is not None and not file_ref:
        file_ref = _sticker_file_ref(sticker)
    reply_to = str(reply_to_message_id or "").strip()
    if not file_ref and not reply_to:
        return text_message
    message = Message()
    if reply_to:
        message += MessageSegment.reply(reply_to)
    if reply:
        if isinstance(text_message, Message):
            message += text_message
        else:
            message += MessageSegment.text(reply)
        if file_ref:
            message += MessageSegment.text("\n")
    if file_ref:
        message += MessageSegment.image(file=file_ref)
    return message


def _log_reply_send_failure(
    exc: ActionFailed,
    sticker: StickerAssetRecord | None,
    used_reply: bool,
    reply_to_message_id: str | None,
) -> None:
    if sticker is not None:
        logger.warning(
            "Group reply send failed for asset #{} ({}) reply_to={}: {}",
            sticker.id,
            sticker.url or sticker.local_path,
            reply_to_message_id if used_reply else "",
            exc,
        )
        return
    if used_reply:
        logger.warning("Group reply send failed reply_to={}: {}", reply_to_message_id, exc)


def _reply_text_message(reply: str) -> Message | str:
    parts = parse_outgoing_mention_parts(reply)
    if not any(part.kind == "at" for part in parts):
        return reply
    message = Message()
    for part in parts:
        if part.kind == "at":
            message += MessageSegment.at(part.user_id)
        elif part.text:
            message += MessageSegment.text(part.text)
    return message


def message_contains_image(message: Message | str) -> bool:
    return isinstance(message, Message) and any(segment.type == "image" for segment in message)


def message_has_reply(message: Message) -> bool:
    return any(segment.type == "reply" for segment in message)


def _sticker_file_ref(sticker: StickerAssetRecord) -> str:
    return sticker_file_ref(sticker)


def _sticker_file_refs(sticker: StickerAssetRecord) -> tuple[str, ...]:
    return sticker_file_refs(sticker)


def same_local_path(left: str, right: str) -> bool:
    try:
        return Path(left).resolve() == Path(right).resolve()
    except OSError:
        return str(left).strip() == str(right).strip()


def reply_record_text(
    reply: str | Iterable[str] | None,
    sticker: StickerAssetRecord | None,
) -> str:
    if isinstance(reply, str) or reply is None:
        text = reply or ""
    else:
        text = "\n".join(part for part in reply if str(part or "").strip())
    if sticker is None:
        return text
    label = sticker.usage or sticker.description or sticker.local_path or sticker.url
    marker = f"[\u8868\u60c5 #{sticker.id}: {label}]"
    return f"{text}\n{marker}".strip()
