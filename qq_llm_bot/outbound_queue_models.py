from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any, Literal

from nonebot.adapters.onebot.v11 import Message

from qq_llm_bot.models import StickerAssetRecord


@dataclass(frozen=True)
class QueuedSendAttempt:
    message: Message | str
    sticker: StickerAssetRecord | None = None


@dataclass
class _QueuedOutboundMessage:
    id: int
    bot_self_id: str
    group_id: str
    send_attempts: tuple[QueuedSendAttempt, ...]
    created_at: float
    next_attempt_at: float
    attempts: int = 0
    source: str = ""
    reason: str = ""


QueuedSendStatus = Literal["sent", "retry", "drop"]
BotsProvider = Callable[[], Iterable[Any]]
StickerSentCallback = Callable[[StickerAssetRecord], None]
