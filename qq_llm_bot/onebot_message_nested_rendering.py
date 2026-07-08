from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from qq_llm_bot.models import MessageMention
from qq_llm_bot.onebot_forward_nodes import (
    extract_forward_nodes,
    extract_quoted_message,
    format_forward_node,
    format_quoted_message,
    forward_fallback,
    quoted_fallback,
)
from qq_llm_bot.onebot_message_types import (
    FORWARDED_RECORD_END,
    FORWARDED_RECORD_START,
    ForwardFetcher,
    ReplyFetcher,
)
from qq_llm_bot.onebot_render_utils import first_text


@dataclass
class ForwardRenderState:
    max_nodes: int
    rendered_nodes: int = 0


async def render_forward_segment(
    data: dict[str, Any],
    *,
    bot_id: str,
    forward_fetcher: ForwardFetcher | None,
    reply_fetcher: ReplyFetcher | None,
    depth: int,
    max_forward_depth: int,
    state: ForwardRenderState,
    render_segments: Any,
) -> tuple[str, list[MessageMention]]:
    forward_id = first_text(data, "id", "forward_id", "resid", "message_id")
    if not forward_fetcher or not forward_id:
        return forward_fallback(forward_id), []
    if depth >= max_forward_depth:
        return f"[鍚堝苟杞彂鑱婂ぉ璁板綍: {forward_id}锛屽凡鍒板睍寮€娣卞害涓婇檺]", []

    payload = await forward_fetcher(forward_id)
    nodes = extract_forward_nodes(payload)
    if not nodes:
        return forward_fallback(forward_id), []

    parts = [FORWARDED_RECORD_START]
    mentions: list[MessageMention] = []
    for index, node in enumerate(nodes):
        if state.rendered_nodes >= state.max_nodes:
            omitted = len(nodes) - index
            parts.append(f"[杩樻湁 {omitted} 鏉¤浆鍙戞秷鎭湭灞曞紑]")
            break
        state.rendered_nodes += 1
        node_text, node_mentions = await render_segments(
            node.content,
            bot_id=bot_id,
            forward_fetcher=forward_fetcher,
            reply_fetcher=reply_fetcher,
            depth=depth + 1,
            max_forward_depth=max_forward_depth,
            state=state,
            include_placeholders=True,
        )
        mentions.extend(node_mentions)
        parts.append(format_forward_node(node, node_text))
    parts.append(FORWARDED_RECORD_END)
    return "\n".join(part for part in parts if part), mentions


async def render_reply_segment(
    data: dict[str, Any],
    *,
    bot_id: str,
    forward_fetcher: ForwardFetcher | None,
    reply_fetcher: ReplyFetcher | None,
    depth: int,
    max_forward_depth: int,
    state: ForwardRenderState,
    render_segments: Any,
) -> tuple[str, list[MessageMention]]:
    reply_id = first_text(data, "id", "message_id")
    if not reply_fetcher or not reply_id:
        return quoted_fallback(reply_id), []
    if depth >= max_forward_depth:
        return f"[琚紩鐢ㄦ秷鎭? {reply_id}锛屽凡鍒板睍寮€娣卞害涓婇檺]", []

    payload = await reply_fetcher(reply_id)
    node = extract_quoted_message(payload)
    if node is None:
        return quoted_fallback(reply_id), []

    node_text, _ = await render_segments(
        node.content,
        bot_id=bot_id,
        forward_fetcher=forward_fetcher,
        reply_fetcher=reply_fetcher,
        depth=depth + 1,
        max_forward_depth=max_forward_depth,
        state=state,
        include_placeholders=True,
    )
    return format_quoted_message(node, node_text, reply_id), []
