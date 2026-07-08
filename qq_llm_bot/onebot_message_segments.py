from __future__ import annotations

import json
import re
from typing import Any, Iterable

from qq_llm_bot.models import MessageAttachment, MessageMention
from qq_llm_bot.onebot_message_types import OutgoingMessagePart

OUTGOING_MENTION_PATTERN = re.compile(
    r"@(?:"
    r"(?P<label>[^@\r\n()]{1,80}?)\s*\(\s*[Qq][Qq]\s*[:：]\s*(?P<label_qq>\d{1,20})\s*\)"
    r"|[Qq][Qq]\s*[:：]\s*(?P<qq>\d{1,20})"
    r")"
)


def render_message_text_and_mentions(
    segments: Iterable[Any],
    bot_id: str = "",
) -> tuple[str, list[MessageMention]]:
    parts: list[str] = []
    mentions: list[MessageMention] = []
    bot_id = str(bot_id)

    for segment in segments:
        segment_type = str(getattr(segment, "type", ""))
        data = dict(getattr(segment, "data", {}) or {})
        if segment_type == "text":
            parts.append(str(data.get("text", "") or ""))
            continue
        if segment_type != "at":
            continue

        user_id = str(data.get("qq", "") or "").strip()
        display_name = _mention_display_name(data, user_id)
        mention = MessageMention(
            user_id=user_id,
            display_name=display_name,
            is_bot=bool(bot_id and user_id == bot_id),
            raw_data=json.dumps(data, ensure_ascii=False),
        )
        mentions.append(mention)
        parts.append(format_mention_label(mention))

    return "".join(parts).strip(), mentions

def parse_outgoing_mention_parts(text: str) -> list[OutgoingMessagePart]:
    parts: list[OutgoingMessagePart] = []
    cursor = 0
    for match in OUTGOING_MENTION_PATTERN.finditer(text):
        user_id = (match.group("label_qq") or match.group("qq") or "").strip()
        if not user_id:
            continue
        if match.start() > cursor:
            parts.append(OutgoingMessagePart("text", text=text[cursor : match.start()]))
        parts.append(OutgoingMessagePart("at", user_id=user_id))
        cursor = match.end()
    if cursor < len(text):
        parts.append(OutgoingMessagePart("text", text=text[cursor:]))
    return parts or [OutgoingMessagePart("text", text=text)]

def format_mention_label(mention: MessageMention) -> str:
    user_id = mention.user_id.strip()
    display_name = " ".join(mention.display_name.split())
    if _is_all_mention(user_id):
        return "@all"
    if display_name and display_name != user_id:
        return f"@{display_name}(QQ:{user_id})"
    return f"@QQ:{user_id}" if user_id else "@unknown"

def coerce_message_segments(message: Any) -> list[Any]:
    return _coerce_segments(message)

def segment_type(segment: Any) -> str:
    return _segment_type(segment)

def segment_data(segment: Any) -> dict[str, Any]:
    return _segment_data(segment)

def reply_segment_ids(message: Any) -> list[str]:
    ids: list[str] = []
    for segment in coerce_message_segments(message):
        if segment_type(segment) != "reply":
            continue
        data = segment_data(segment)
        reply_id = str(data.get("id") or data.get("message_id") or "").strip()
        if reply_id:
            ids.append(reply_id)
    return ids

def image_attachments_from_payload(payload: Any) -> list[MessageAttachment]:
    if payload is None:
        return []
    if isinstance(payload, dict):
        message = payload.get("message")
        if message is None:
            message = payload.get("content")
        if message is None:
            message = payload.get("raw_message")
        return image_attachments_from_message(message)
    return image_attachments_from_message(payload)

def image_attachments_from_message(message: Any) -> list[MessageAttachment]:
    attachments: list[MessageAttachment] = []
    for segment in coerce_message_segments(message):
        current_type = segment_type(segment)
        data = segment_data(segment)
        if current_type == "image":
            attachments.append(_image_attachment_from_data(data))
            continue
        if current_type in {"node", "forward"}:
            for key in ("content", "message", "raw_message"):
                if key in data:
                    attachments.extend(image_attachments_from_message(data.get(key)))
    return attachments

def _mention_display_name(data: dict[str, Any], user_id: str) -> str:
    for key in ("name", "nickname", "card"):
        value = str(data.get(key, "") or "").strip()
        if value:
            return value
    return "all" if _is_all_mention(user_id) else user_id

def _is_all_mention(user_id: str) -> bool:
    return user_id.lower() in {"all", "everyone"}


def _mention_from_data(data: dict[str, Any], bot_id: str) -> MessageMention:
    user_id = str(data.get("qq", "") or "").strip()
    display_name = _mention_display_name(data, user_id)
    return MessageMention(
        user_id=user_id,
        display_name=display_name,
        is_bot=bool(bot_id and user_id == bot_id),
        raw_data=json.dumps(data, ensure_ascii=False),
    )

def _coerce_segments(message: Any) -> list[Any]:
    if message is None:
        return []
    if isinstance(message, str):
        return _parse_message_string(message)
    if isinstance(message, dict):
        if "type" in message:
            return [message]
        for key in ("message", "content", "raw_message"):
            if key in message:
                return _coerce_segments(message.get(key))
        return []
    try:
        return list(message)
    except TypeError:
        return [message]

def _parse_message_string(message: str) -> list[Any]:
    try:
        from nonebot.adapters.onebot.v11 import Message

        return list(Message(message))
    except Exception:
        return [{"type": "text", "data": {"text": message}}]

def _segment_type(segment: Any) -> str:
    if isinstance(segment, dict):
        return str(segment.get("type", "") or "")
    return str(getattr(segment, "type", "") or "")

def _segment_data(segment: Any) -> dict[str, Any]:
    raw_data = segment.get("data", {}) if isinstance(segment, dict) else getattr(segment, "data", {})
    return dict(raw_data) if isinstance(raw_data, dict) else {}

def _image_attachment_from_data(data: dict[str, Any]) -> MessageAttachment:
    return MessageAttachment(
        attachment_type="image",
        file=str(data.get("file", "") or ""),
        url=str(data.get("url", "") or data.get("file_url", "") or ""),
        summary=str(data.get("summary", "") or ""),
        raw_data=json.dumps(data, ensure_ascii=False),
    )
