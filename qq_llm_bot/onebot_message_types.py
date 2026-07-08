from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

FORWARDED_RECORD_START = "[合并转发聊天记录开始]"
FORWARDED_RECORD_END = "[合并转发聊天记录结束]"
QUOTED_MESSAGE_START = "[被引用消息开始]"
QUOTED_MESSAGE_END = "[被引用消息结束]"

ForwardFetcher = Callable[[str], Awaitable[Any]]
ReplyFetcher = Callable[[str], Awaitable[Any]]


@dataclass(frozen=True)
class OutgoingMessagePart:
    kind: str
    text: str = ""
    user_id: str = ""
