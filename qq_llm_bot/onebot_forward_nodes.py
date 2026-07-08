from __future__ import annotations

from typing import Any, NamedTuple

from qq_llm_bot.onebot_message_types import QUOTED_MESSAGE_END, QUOTED_MESSAGE_START
from qq_llm_bot.onebot_render_utils import first_existing, first_text


class ForwardNode(NamedTuple):
    user_id: str
    nickname: str
    content: Any


def extract_quoted_message(payload: Any) -> ForwardNode | None:
    if payload is None:
        return None
    if isinstance(payload, dict):
        node = _node_from_item(payload)
        if node is not None:
            return node
    nodes = _nodes_from_value(payload)
    return nodes[0] if nodes else None


def extract_forward_nodes(payload: Any) -> list[ForwardNode]:
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


def format_forward_node(node: ForwardNode, text: str) -> str:
    sender = format_forward_sender(node)
    body = text.strip() or "[空消息]"
    body = "\n  ".join(body.splitlines())
    return f"{sender}: {body}"


def format_forward_sender(node: ForwardNode) -> str:
    nickname = " ".join(node.nickname.split())
    user_id = node.user_id.strip()
    if nickname and user_id:
        return f"{nickname}(QQ:{user_id})"
    if nickname:
        return nickname
    if user_id:
        return f"QQ:{user_id}"
    return "unknown"


def format_quoted_message(node: ForwardNode, text: str, message_id: str) -> str:
    sender = format_forward_sender(node)
    title = "[被引用消息"
    if message_id:
        title += f" #{message_id}"
    if sender != "unknown":
        title += f" | {sender}"
    title += "]"
    body = text.strip() or "[空消息]"
    body = "\n  ".join(body.splitlines())
    return f"{QUOTED_MESSAGE_START}\n{title}\n  {body}\n{QUOTED_MESSAGE_END}"


def forward_fallback(forward_id: str) -> str:
    return f"[合并转发聊天记录: {forward_id or '无法展开'}]"


def quoted_fallback(message_id: str) -> str:
    return f"[被引用消息: {message_id or '无法展开'}]"


def _nodes_from_value(value: Any) -> list[ForwardNode]:
    if value is None:
        return []
    if isinstance(value, list):
        nodes = [node for item in value if (node := _node_from_item(item)) is not None]
        if nodes:
            return nodes
        if _looks_like_segment_list(value):
            return [ForwardNode("", "", value)]
        return []
    node = _node_from_item(value)
    if node is not None:
        return [node]
    if isinstance(value, (str, dict)):
        return [ForwardNode("", "", value)]
    return []


def _node_from_item(item: Any) -> ForwardNode | None:
    if not isinstance(item, dict):
        return None
    data = item.get("data") if item.get("type") == "node" else item
    if not isinstance(data, dict):
        return None
    sender = data.get("sender")
    sender_data = sender if isinstance(sender, dict) else {}
    content = first_existing(data, "content", "message", "raw_message")
    if content is None:
        return None
    user_id = first_text(data, "user_id", "uin", "qq") or first_text(
        sender_data, "user_id", "uin", "qq"
    )
    nickname = first_text(data, "nickname", "name") or first_text(sender_data, "nickname", "card")
    return ForwardNode(user_id=user_id, nickname=nickname, content=content)


def _looks_like_segment_list(items: list[Any]) -> bool:
    return any(isinstance(item, dict) and "type" in item for item in items)
