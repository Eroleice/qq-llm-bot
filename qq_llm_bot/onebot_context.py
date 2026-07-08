from __future__ import annotations

import time
from collections.abc import Iterable
from typing import Any

from loguru import logger
from nonebot.adapters.onebot.v11 import Bot, GroupMessageEvent

from qq_llm_bot.directness import looks_like_bot_address, text_mentions_bot_name
from qq_llm_bot.models import MessageContext
from qq_llm_bot.onebot_messages import (
    image_attachments_from_message,
    render_message_text_and_mentions,
    render_message_text_and_mentions_with_forwards,
)


async def build_message_context(
    bot: Bot,
    event: GroupMessageEvent,
    *,
    bot_names: Iterable[str],
) -> MessageContext:
    top_level_text, _ = render_message_text_and_mentions(event.message, str(bot.self_id))
    plain_text, mentions = await render_message_text_and_mentions_with_forwards(
        event.message,
        str(bot.self_id),
        forward_message_fetcher(bot),
        reply_fetcher=reply_message_fetcher(bot, event),
    )
    sender = getattr(event, "sender", None)
    sender_nickname = _sender_field(sender, "nickname")
    sender_name = _sender_field(sender, "card") or sender_nickname
    sender_role = _sender_field(sender, "role")

    return MessageContext(
        group_id=str(event.group_id),
        user_id=str(event.user_id),
        message_id=str(event.message_id),
        plain_text=plain_text,
        raw_message=str(event.message),
        sender_name=sender_name,
        sender_nickname=sender_nickname,
        sender_role=sender_role,
        is_direct=_is_direct_message(bot, event, top_level_text, bot_names),
        bot_mentioned=_is_bot_mentioned(bot, event, top_level_text, bot_names),
        timestamp=int(getattr(event, "time", 0) or time.time()),
        attachments=image_attachments_from_message(event.message),
        mentions=mentions,
    )


def forward_message_fetcher(bot: Bot):
    async def _fetch(forward_id: str) -> Any:
        try:
            return await bot.get_forward_msg(id=forward_id)
        except Exception as exc:
            logger.warning("Failed to fetch forwarded message {}: {}", forward_id, exc)
            return None

    return _fetch


def reply_message_fetcher(bot: Bot, event: GroupMessageEvent):
    async def _fetch(message_id: str) -> Any:
        event_reply = event_reply_payload(event, message_id)
        if event_reply is not None:
            return event_reply
        target_message_id = onebot_message_id(message_id)
        if target_message_id is None:
            return None
        try:
            return await bot.get_msg(message_id=target_message_id)
        except Exception as exc:
            logger.warning("Failed to fetch quoted message {}: {}", message_id, exc)
            return None

    return _fetch


def event_reply_payload(event: GroupMessageEvent, message_id: str) -> dict[str, Any] | None:
    reply = getattr(event, "reply", None)
    if reply is None:
        return None
    reply_message_id = str(
        getattr(reply, "message_id", "") or getattr(reply, "id", "") or ""
    ).strip()
    if reply_message_id and message_id and reply_message_id != str(message_id):
        return None
    message = getattr(reply, "message", None)
    if message is None:
        message = getattr(reply, "content", None)
    raw_message = str(getattr(reply, "raw_message", "") or "")
    sender = getattr(reply, "sender", None)
    return {
        "message_id": reply_message_id or str(message_id),
        "message": message if message is not None else raw_message,
        "raw_message": raw_message,
        "sender": {
            "user_id": _sender_field(sender, "user_id"),
            "nickname": _sender_field(sender, "nickname") or _sender_field(sender, "card"),
            "card": _sender_field(sender, "card"),
        },
    }


def onebot_message_id(message_id: object) -> int | str | None:
    text = str(message_id or "").strip()
    if not text:
        return None
    try:
        return int(text)
    except ValueError:
        return text


def _is_direct_message(
    bot: Bot,
    event: GroupMessageEvent,
    plain_text: str,
    bot_names: Iterable[str],
) -> bool:
    if _has_explicit_bot_at(bot, event):
        return True
    if _is_reply_to_bot(bot, event):
        return True
    return looks_like_bot_address(plain_text, bot_names)


def _is_bot_mentioned(
    bot: Bot,
    event: GroupMessageEvent,
    plain_text: str,
    bot_names: Iterable[str],
) -> bool:
    return (
        _has_explicit_bot_at(bot, event)
        or _is_reply_to_bot(bot, event)
        or text_mentions_bot_name(plain_text, bot_names)
    )


def _has_explicit_bot_at(bot: Bot, event: GroupMessageEvent) -> bool:
    for segment in event.message:
        if segment.type == "at" and str(segment.data.get("qq")) == str(bot.self_id):
            return True
    return False


def _is_reply_to_bot(bot: Bot, event: GroupMessageEvent) -> bool:
    reply = getattr(event, "reply", None)
    if reply is None:
        return False
    sender = getattr(reply, "sender", None)
    return str(_sender_field(sender, "user_id")) == str(bot.self_id)


def _sender_field(sender: Any, field: str) -> str:
    if sender is None:
        return ""
    if isinstance(sender, dict):
        return str(sender.get(field, "") or "")
    return str(getattr(sender, field, "") or "")
