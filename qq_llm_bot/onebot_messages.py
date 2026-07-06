from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Iterable, NamedTuple

from qq_llm_bot.models import MessageMention

FORWARDED_RECORD_START = "[合并转发聊天记录开始]"
FORWARDED_RECORD_END = "[合并转发聊天记录结束]"
QUOTED_MESSAGE_START = "[被引用消息开始]"
QUOTED_MESSAGE_END = "[被引用消息结束]"
OUTGOING_MENTION_PATTERN = re.compile(
    r"@(?:"
    r"(?P<label>[^@\r\n()]{1,80}?)\s*\(\s*[Qq][Qq]\s*[:：]\s*(?P<label_qq>\d{1,20})\s*\)"
    r"|[Qq][Qq]\s*[:：]\s*(?P<qq>\d{1,20})"
    r")"
)

ForwardFetcher = Callable[[str], Awaitable[Any]]
ReplyFetcher = Callable[[str], Awaitable[Any]]


@dataclass(frozen=True)
class OutgoingMessagePart:
    kind: str
    text: str = ""
    user_id: str = ""


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
    state = _ForwardRenderState(max_nodes=max(0, max_forward_nodes))
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
    return _truncate_text(text, max_forward_chars), mentions


def strip_forwarded_records(text: str) -> str:
    return _strip_marked_block(text, FORWARDED_RECORD_START, FORWARDED_RECORD_END)


def strip_quoted_messages(text: str) -> str:
    return _strip_marked_block(text, QUOTED_MESSAGE_START, QUOTED_MESSAGE_END)


def _strip_marked_block(text: str, start_marker: str, end_marker: str) -> str:
    pattern = re.compile(
        rf"{re.escape(start_marker)}.*?{re.escape(end_marker)}",
        re.S,
    )
    return "\n".join(line for line in pattern.sub("", text).splitlines() if line.strip()).strip()


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


def _mention_display_name(data: dict[str, Any], user_id: str) -> str:
    for key in ("name", "nickname", "card"):
        value = str(data.get(key, "") or "").strip()
        if value:
            return value
    return "all" if _is_all_mention(user_id) else user_id


def _is_all_mention(user_id: str) -> bool:
    return user_id.lower() in {"all", "everyone"}


@dataclass
class _ForwardRenderState:
    max_nodes: int
    rendered_nodes: int = 0


class _ForwardNode(NamedTuple):
    user_id: str
    nickname: str
    content: Any


async def _render_segments(
    segments: Any,
    *,
    bot_id: str,
    forward_fetcher: ForwardFetcher | None,
    reply_fetcher: ReplyFetcher | None,
    depth: int,
    max_forward_depth: int,
    state: _ForwardRenderState,
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
            forward_text, forward_mentions = await _render_forward_segment(
                data,
                bot_id=bot_id,
                forward_fetcher=forward_fetcher,
                reply_fetcher=reply_fetcher,
                depth=depth,
                max_forward_depth=max_forward_depth,
                state=state,
            )
            _append_block(parts, forward_text)
            mentions.extend(forward_mentions)
            continue
        if segment_type == "reply":
            reply_text, reply_mentions = await _render_reply_segment(
                data,
                bot_id=bot_id,
                forward_fetcher=forward_fetcher,
                reply_fetcher=reply_fetcher,
                depth=depth,
                max_forward_depth=max_forward_depth,
                state=state,
            )
            _append_block(parts, reply_text)
            mentions.extend(reply_mentions)
            continue
        if include_placeholders:
            parts.append(_segment_placeholder(segment_type, data))

    return "".join(parts).strip(), mentions


async def _render_forward_segment(
    data: dict[str, Any],
    *,
    bot_id: str,
    forward_fetcher: ForwardFetcher | None,
    reply_fetcher: ReplyFetcher | None,
    depth: int,
    max_forward_depth: int,
    state: _ForwardRenderState,
) -> tuple[str, list[MessageMention]]:
    forward_id = _first_text(data, "id", "forward_id", "resid", "message_id")
    if not forward_fetcher or not forward_id:
        return _forward_fallback(forward_id), []
    if depth >= max_forward_depth:
        return f"[合并转发聊天记录: {forward_id}，已到展开深度上限]", []

    payload = await forward_fetcher(forward_id)
    nodes = _extract_forward_nodes(payload)
    if not nodes:
        return _forward_fallback(forward_id), []

    parts = [FORWARDED_RECORD_START]
    mentions: list[MessageMention] = []
    for index, node in enumerate(nodes):
        if state.rendered_nodes >= state.max_nodes:
            omitted = len(nodes) - index
            parts.append(f"[还有 {omitted} 条转发消息未展开]")
            break
        state.rendered_nodes += 1
        node_text, node_mentions = await _render_segments(
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
        parts.append(_format_forward_node(node, node_text))
    parts.append(FORWARDED_RECORD_END)
    return "\n".join(part for part in parts if part), mentions


async def _render_reply_segment(
    data: dict[str, Any],
    *,
    bot_id: str,
    forward_fetcher: ForwardFetcher | None,
    reply_fetcher: ReplyFetcher | None,
    depth: int,
    max_forward_depth: int,
    state: _ForwardRenderState,
) -> tuple[str, list[MessageMention]]:
    reply_id = _first_text(data, "id", "message_id")
    if not reply_fetcher or not reply_id:
        return _quoted_fallback(reply_id), []
    if depth >= max_forward_depth:
        return f"[被引用消息: {reply_id}，已到展开深度上限]", []

    payload = await reply_fetcher(reply_id)
    node = _extract_quoted_message(payload)
    if node is None:
        return _quoted_fallback(reply_id), []

    node_text, _ = await _render_segments(
        node.content,
        bot_id=bot_id,
        forward_fetcher=forward_fetcher,
        reply_fetcher=reply_fetcher,
        depth=depth + 1,
        max_forward_depth=max_forward_depth,
        state=state,
        include_placeholders=True,
    )
    return _format_quoted_message(node, node_text, reply_id), []


def _extract_quoted_message(payload: Any) -> _ForwardNode | None:
    if payload is None:
        return None
    if isinstance(payload, dict):
        node = _node_from_item(payload)
        if node is not None:
            return node
    nodes = _nodes_from_value(payload)
    return nodes[0] if nodes else None


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
        for key in ("message", "content"):
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


def _extract_forward_nodes(payload: Any) -> list[_ForwardNode]:
    if payload is None:
        return []
    if isinstance(payload, dict):
        for key in ("messages", "message", "content"):
            value = payload.get(key)
            nodes = _nodes_from_value(value)
            if nodes:
                return nodes
        node = _node_from_item(payload)
        return [node] if node else []
    return _nodes_from_value(payload)


def _nodes_from_value(value: Any) -> list[_ForwardNode]:
    if value is None:
        return []
    if isinstance(value, list):
        nodes = [node for item in value if (node := _node_from_item(item)) is not None]
        if nodes:
            return nodes
        if _looks_like_segment_list(value):
            return [_ForwardNode("", "", value)]
        return []
    node = _node_from_item(value)
    if node is not None:
        return [node]
    if isinstance(value, (str, dict)):
        return [_ForwardNode("", "", value)]
    return []


def _node_from_item(item: Any) -> _ForwardNode | None:
    if not isinstance(item, dict):
        return None
    data = item.get("data") if item.get("type") == "node" else item
    if not isinstance(data, dict):
        return None
    sender = data.get("sender")
    sender_data = sender if isinstance(sender, dict) else {}
    content = _first_existing(data, "content", "message", "raw_message")
    if content is None:
        return None
    user_id = _first_text(data, "user_id", "uin", "qq") or _first_text(
        sender_data, "user_id", "uin", "qq"
    )
    nickname = _first_text(data, "nickname", "name") or _first_text(sender_data, "nickname", "card")
    return _ForwardNode(user_id=user_id, nickname=nickname, content=content)


def _looks_like_segment_list(items: list[Any]) -> bool:
    return any(isinstance(item, dict) and "type" in item for item in items)


def _format_forward_node(node: _ForwardNode, text: str) -> str:
    sender = _format_forward_sender(node)
    body = text.strip() or "[空消息]"
    body = "\n  ".join(body.splitlines())
    return f"{sender}: {body}"


def _format_forward_sender(node: _ForwardNode) -> str:
    nickname = " ".join(node.nickname.split())
    user_id = node.user_id.strip()
    if nickname and user_id:
        return f"{nickname}(QQ:{user_id})"
    if nickname:
        return nickname
    if user_id:
        return f"QQ:{user_id}"
    return "unknown"


def _format_quoted_message(node: _ForwardNode, text: str, message_id: str) -> str:
    sender = _format_forward_sender(node)
    title = "[被引用消息"
    if message_id:
        title += f" #{message_id}"
    if sender != "unknown":
        title += f" | {sender}"
    title += "]"
    body = text.strip() or "[空消息]"
    body = "\n  ".join(body.splitlines())
    return f"{QUOTED_MESSAGE_START}\n{title}\n  {body}\n{QUOTED_MESSAGE_END}"


def _segment_placeholder(segment_type: str, data: dict[str, Any]) -> str:
    if not segment_type:
        return ""
    if segment_type == "image":
        label = _first_text(data, "summary", "file", "url", "file_url")
        return f"[图片: {label}]" if label else "[图片]"
    if segment_type in {"record", "voice"}:
        return "[语音]"
    if segment_type == "video":
        return "[视频]"
    if segment_type == "face":
        return "[表情]"
    if segment_type == "json":
        return _json_placeholder(data)
    if segment_type == "xml":
        return "[XML消息]"
    if segment_type == "reply":
        return ""
    return f"[{segment_type}消息]"


def _json_placeholder(data: dict[str, Any]) -> str:
    raw = str(data.get("data", "") or "").strip()
    if not raw:
        return "[JSON消息]"
    try:
        decoded = json.loads(raw)
    except json.JSONDecodeError:
        return "[JSON消息]"
    title = _find_nested_text(decoded, {"title", "desc", "summary", "prompt"})
    return f"[JSON消息: {title}]" if title else "[JSON消息]"


def _find_nested_text(value: Any, keys: set[str]) -> str:
    if isinstance(value, dict):
        for key, item in value.items():
            if str(key).lower() in keys:
                text = str(item or "").strip()
                if text:
                    return text
        for item in value.values():
            text = _find_nested_text(item, keys)
            if text:
                return text
    if isinstance(value, list):
        for item in value:
            text = _find_nested_text(item, keys)
            if text:
                return text
    return ""


def _forward_fallback(forward_id: str) -> str:
    return f"[合并转发聊天记录: {forward_id or '无法展开'}]"


def _quoted_fallback(message_id: str) -> str:
    return f"[被引用消息: {message_id or '无法展开'}]"


def _append_block(parts: list[str], block: str) -> None:
    if not block:
        return
    if parts and parts[-1] and not parts[-1].endswith(("\n", " ")):
        parts.append("\n")
    parts.append(block)
    if not block.endswith("\n"):
        parts.append("\n")


def _first_existing(data: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in data:
            return data[key]
    return None


def _first_text(data: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = str(data.get(key, "") or "").strip()
        if value:
            return value
    return ""


def _truncate_text(text: str, limit: int) -> str:
    if limit <= 0 or len(text) <= limit:
        return text
    if limit <= 3:
        return text[:limit]
    return text[: limit - 3].rstrip() + "..."
