from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, replace
from typing import Protocol

from qq_llm_bot.models import (
    FactCandidate,
    ImageVisionCacheRecord,
    MemoryCandidate,
    MessageAttachment,
    MessageContext,
    StickerCandidate,
)

UNRESOLVED_IMAGE_DESCRIPTION = "收到一张图片，但当前没有可用的视觉解读结果"


@dataclass(frozen=True)
class VisionAnalysis:
    descriptions: list[str]
    ocr_text: str = ""
    topics: tuple[str, ...] = ()
    memory_candidates: tuple[MemoryCandidate, ...] = ()
    fact_candidates: tuple[FactCandidate, ...] = ()
    sticker_candidates: tuple[StickerCandidate, ...] = ()
    attachment_descriptions: tuple[str, ...] = ()
    resolved_image_urls: tuple[str, ...] = ()


@dataclass(frozen=True)
class VisionImageResult:
    url: str
    description: str = ""
    ocr_text: str = ""
    topics: tuple[str, ...] = ()
    image_type: str = "unknown"
    memory: str = ""
    confidence: float = 0.0
    importance: float = 0.5
    is_sticker: bool = False
    sticker_mood: str = ""
    sticker_usage: str = ""
    sticker_tags: tuple[str, ...] = ()
    sticker_confidence: float = 0.0


class VisionCacheStore(Protocol):
    def get_image_vision_cache(self, url: str) -> ImageVisionCacheRecord | None:
        ...

    def upsert_image_vision_cache(
        self,
        *,
        url: str,
        description: str,
        ocr_text: str = "",
        topics: Iterable[str] = (),
        memory: str = "",
        confidence: float = 0.0,
        importance: float = 0.5,
        model: str = "",
    ) -> None:
        ...


def context_with_vision(context: MessageContext, vision: VisionAnalysis) -> MessageContext:
    if not vision.descriptions and not vision.ocr_text:
        return context
    lines = []
    if context.plain_text:
        lines.append(context.plain_text)
    for description in vision.descriptions:
        lines.append(f"[图片解读] {description}")
    if vision.ocr_text:
        lines.append(f"[图片文字] {vision.ocr_text}")
    return replace(context, plain_text="\n".join(lines))


def unresolved_context_image_urls(
    context: MessageContext,
    vision: VisionAnalysis,
    limit: int,
) -> list[str]:
    if not context.attachments:
        return []
    resolved_urls = set(vision.resolved_image_urls)
    urls = [
        attachment.url
        for attachment in select_image_attachments(context.attachments, limit)
        if attachment.url and attachment.url not in resolved_urls
    ]
    return dedupe_strings(urls)


def recordable_image_descriptions(
    context: MessageContext,
    vision: VisionAnalysis,
) -> list[str]:
    if not context.attachments:
        return list(vision.descriptions)
    resolved_urls = set(vision.resolved_image_urls)
    attachment_descriptions = list(vision.attachment_descriptions)
    descriptions: list[str] = []
    image_index = 0
    for attachment in context.attachments:
        if attachment.attachment_type != "image":
            continue
        description = (
            attachment_descriptions[image_index]
            if image_index < len(attachment_descriptions)
            else ""
        )
        image_index += 1
        if (
            attachment.url
            and attachment.url in resolved_urls
            and description
            and description != UNRESOLVED_IMAGE_DESCRIPTION
        ):
            descriptions.append(description)
        else:
            descriptions.append("")
    return descriptions


def select_image_attachments(
    attachments: list[MessageAttachment],
    limit: int,
) -> list[MessageAttachment]:
    images = [attachment for attachment in attachments if attachment.attachment_type == "image" and attachment.url]
    max_count = max(1, int(limit))
    if len(images) <= max_count:
        return images
    if max_count == 1:
        return [images[0]]
    if max_count == 2:
        return [images[0], images[-1]]

    step = (len(images) - 1) / (max_count - 1)
    indices: list[int] = []
    for slot in range(max_count):
        index = int(round(slot * step))
        if index not in indices:
            indices.append(index)
    if indices[-1] != len(images) - 1:
        indices[-1] = len(images) - 1
    return [images[index] for index in indices]


def safe_image_type(value: str) -> str:
    return value if value in {"sticker", "content_image", "pure_image", "unknown"} else "unknown"


def infer_image_type(
    description: str,
    ocr_text: str,
    topics: tuple[str, ...],
    is_sticker: bool,
) -> str:
    if is_sticker or looks_like_sticker_image(description, ocr_text, topics):
        return "sticker"
    if ocr_text.strip():
        return "content_image"
    if description.strip():
        return "pure_image"
    return "unknown"


def attachment_descriptions(
    context: MessageContext,
    results_by_url: dict[str, VisionImageResult],
) -> tuple[str, ...]:
    descriptions: list[str] = []
    for attachment in context.attachments:
        if attachment.attachment_type != "image":
            continue
        result = results_by_url.get(attachment.url)
        if result and result.description:
            descriptions.append(result.description)
        else:
            descriptions.append("")
    return tuple(descriptions)


def image_interest_topic(result: VisionImageResult) -> str:
    for topic in result.topics:
        cleaned = _clean_fact_text(topic, 40)
        if cleaned and cleaned not in {"图片", "截图", "照片", "内容", "文字"}:
            return cleaned
    if result.ocr_text:
        text = _clean_fact_text(result.ocr_text, 80)
        return text[:30].strip()
    description = result.description
    description = re.sub(r"^(一张|这张|图片中|图中|照片中|画面中)", "", description).strip()
    description = re.sub(r"(图片|照片|截图|插画|海报)$", "", description).strip()
    return _clean_fact_text(description, 40)


def clean_image_text(value: str) -> str:
    text = " ".join(str(value).strip().split())
    return text[:300].strip()


def clean_sticker_text(value: str, limit: int = 160) -> str:
    text = " ".join(str(value).strip().split())
    return text[:limit].strip()


def looks_like_sticker_image(description: str, ocr_text: str, topics: tuple[str, ...]) -> bool:
    haystack = " ".join([description, ocr_text, *topics]).lower()
    return any(
        token in haystack
        for token in (
            "表情包",
            "梗图",
            "meme",
            "反应图",
            "配字",
            "熊猫头",
            "猫猫表情",
            "狗头",
            "无语",
            "笑死",
        )
    )


def infer_sticker_mood(result: VisionImageResult) -> str:
    text = " ".join([result.description, result.ocr_text, *result.topics])
    if any(token in text for token in ("笑", "哈哈", "开心", "乐")):
        return "好笑"
    if any(token in text for token in ("无语", "沉默", "尴尬")):
        return "无语"
    if any(token in text for token in ("震惊", "惊讶", "瞪大")):
        return "震惊"
    if any(token in text for token in ("困", "累", "下班")):
        return "疲惫"
    return "接梗"


def fallback_sticker_usage(result: VisionImageResult) -> str:
    mood = infer_sticker_mood(result)
    if mood == "好笑":
        return "适合在群里有人开玩笑、接梗或大家都在笑时使用。"
    if mood == "无语":
        return "适合在轻度吐槽、无奈或尴尬但不严肃的场合使用。"
    if mood == "震惊":
        return "适合在看到意外消息、离谱展开或反转时使用。"
    if mood == "疲惫":
        return "适合在聊到犯困、下班、累了或想摆一下时使用。"
    return "适合在轻松聊天里接梗或表达反应时使用。"


def looks_sensitive_image_memory(value: str) -> bool:
    return bool(
        re.search(
            r"(身份证|手机号|住址|家庭住址|银行卡|密码|真实姓名|人脸识别|长得像|某某本人)",
            value,
        )
    )


def dedupe_strings(values: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        text = str(value).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        deduped.append(text)
    return deduped


def _clean_fact_text(value: str, limit: int) -> str:
    text = " ".join(str(value or "").strip().split())
    return text[:limit].strip()
