from __future__ import annotations

from typing import Any, Iterable

from qq_llm_bot.models import MessageMention
from qq_llm_bot.onebot_message_nested_rendering import (
    ForwardRenderState,
    render_forward_segment,
    render_reply_segment,
)
from qq_llm_bot.onebot_message_segments import (
    _coerce_segments,
    _mention_from_data,
    _segment_data,
    _segment_type,
    format_mention_label,
)
from qq_llm_bot.onebot_message_types import ForwardFetcher, ReplyFetcher
from qq_llm_bot.onebot_render_utils import append_block, truncate_text
from qq_llm_bot.onebot_segment_placeholders import segment_placeholder


async def render_message_text_and_mentions_with_forwards(
    segments: Iterable[Any],
    bot_id: str = "",
    forward_fetcher: ForwardFetcher | None = None,
    *,
    reply_fetcher: ReplyFetcher | None = None,
    max_forward_depth: int = 2,
    max_forward_nodes: int = 80,
    max_forward_chars: int = 6000,
) -> tuple[str, list[MessageMention]]:
    state = ForwardRenderState(max_nodes=max(0, max_forward_nodes))
    text, mentions = await _render_segments(
        segments,
        bot_id=str(bot_id),
        forward_fetcher=forward_fetcher,
        reply_fetcher=reply_fetcher,
        depth=0,
        max_forward_depth=max(0, max_forward_depth),
        state=state,
        include_placeholders=False,
    )
    return truncate_text(text, max_forward_chars), mentions


async def _render_segments(
    segments: Any,
    *,
    bot_id: str,
    forward_fetcher: ForwardFetcher | None,
    reply_fetcher: ReplyFetcher | None,
    depth: int,
    max_forward_depth: int,
    state: ForwardRenderState,
    include_placeholders: bool,
) -> tuple[str, list[MessageMention]]:
    parts: list[str] = []
    mentions: list[MessageMention] = []

    for segment in _coerce_segments(segments):
        segment_type = _segment_type(segment)
        data = _segment_data(segment)
        if segment_type == "text":
            parts.append(str(data.get("text", "") or ""))
            continue
        if segment_type == "at":
            mention = _mention_from_data(data, bot_id)
            mentions.append(mention)
            parts.append(format_mention_label(mention))
            continue
        if segment_type == "forward":
            forward_text, forward_mentions = await render_forward_segment(
                data,
                bot_id=bot_id,
                forward_fetcher=forward_fetcher,
                reply_fetcher=reply_fetcher,
                depth=depth,
                max_forward_depth=max_forward_depth,
                state=state,
                render_segments=_render_segments,
            )
            append_block(parts, forward_text)
            mentions.extend(forward_mentions)
            continue
        if segment_type == "reply":
            reply_text, reply_mentions = await render_reply_segment(
                data,
                bot_id=bot_id,
                forward_fetcher=forward_fetcher,
                reply_fetcher=reply_fetcher,
                depth=depth,
                max_forward_depth=max_forward_depth,
                state=state,
                render_segments=_render_segments,
            )
            append_block(parts, reply_text)
            mentions.extend(reply_mentions)
            continue
        if include_placeholders:
            parts.append(segment_placeholder(segment_type, data))

    return "".join(parts).strip(), mentions
