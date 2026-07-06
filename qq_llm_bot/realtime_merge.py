from __future__ import annotations

from dataclasses import replace
from typing import Iterable

from qq_llm_bot.models import MessageContext


def merge_realtime_contexts(contexts: Iterable[MessageContext]) -> MessageContext:
    items = [context for context in contexts if context is not None]
    if not items:
        raise ValueError("contexts must not be empty")
    if len(items) == 1:
        return items[0]

    latest = items[-1]
    return replace(
        latest,
        plain_text=_merged_plain_text(items),
        raw_message=_merged_raw_message(items),
        is_direct=any(context.is_direct for context in items),
        timestamp=latest.timestamp,
        attachments=[
            attachment
            for context in items
            for attachment in context.attachments
        ],
        mentions=[
            mention
            for context in items
            for mention in context.mentions
        ],
    )


def split_image_descriptions_by_context(
    contexts: Iterable[MessageContext],
    descriptions: Iterable[str],
) -> list[tuple[MessageContext, list[str]]]:
    remaining = [str(description or "").strip() for description in descriptions]
    result: list[tuple[MessageContext, list[str]]] = []
    offset = 0
    for context in contexts:
        image_count = sum(
            1
            for attachment in context.attachments
            if attachment.attachment_type == "image"
        )
        if image_count <= 0:
            result.append((context, []))
            continue
        result.append((context, remaining[offset : offset + image_count]))
        offset += image_count
    return result


def _merged_plain_text(contexts: list[MessageContext]) -> str:
    lines = [f"用户连续发了 {len(contexts)} 条："]
    for index, context in enumerate(contexts, start=1):
        text = _message_text_for_merge(context)
        lines.append(f"{index}. {text}")
    return "\n".join(lines).strip()


def _message_text_for_merge(context: MessageContext) -> str:
    text = str(context.plain_text or "").strip()
    if text:
        return text
    if context.attachments:
        return "[图片]"
    return str(context.raw_message or "").strip() or "[空消息]"


def _merged_raw_message(contexts: list[MessageContext]) -> str:
    messages = [str(context.raw_message or "").strip() for context in contexts]
    return "\n".join(message for message in messages if message).strip() or contexts[-1].raw_message
