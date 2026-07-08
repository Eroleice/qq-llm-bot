from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True)
class MessageAttachment:
    attachment_type: Literal["image"]
    file: str = ""
    url: str = ""
    summary: str = ""
    raw_data: str = ""


@dataclass(frozen=True)
class MessageMention:
    user_id: str
    display_name: str = ""
    is_bot: bool = False
    raw_data: str = ""


@dataclass(frozen=True)
class ImageVisionCacheRecord:
    url: str
    description: str
    ocr_text: str = ""
    topics: tuple[str, ...] = ()
    memory: str = ""
    confidence: float = 0.0
    importance: float = 0.5
    model: str = ""
    created_at: int = 0
    updated_at: int = 0
    last_seen_at: int = 0
    hit_count: int = 0


@dataclass(frozen=True)
class StickerCandidate:
    url: str
    file: str = ""
    description: str = ""
    ocr_text: str = ""
    mood: str = ""
    usage: str = ""
    tags: tuple[str, ...] = ()
    confidence: float = 0.0


@dataclass(frozen=True)
class StickerAssetRecord:
    id: int
    group_id: str
    source_user_id: str
    source_message_id: str
    url: str
    file: str
    local_path: str
    sha256: str
    description: str
    ocr_text: str
    mood: str
    usage: str
    tags: tuple[str, ...]
    confidence: float
    enabled: bool
    created_at: int
    updated_at: int
    last_seen_at: int
    hit_count: int = 0
    send_count: int = 0
    last_sent_at: int = 0


@dataclass(frozen=True)
class MessageContext:
    group_id: str
    user_id: str
    message_id: str
    plain_text: str
    raw_message: str
    sender_name: str = ""
    sender_nickname: str = ""
    sender_role: str = ""
    is_direct: bool = False
    bot_mentioned: bool = False
    timestamp: int = 0
    attachments: list[MessageAttachment] = field(default_factory=list)
    mentions: list[MessageMention] = field(default_factory=list)
