from __future__ import annotations

from typing import Any

from loguru import logger

from qq_llm_bot.agent_common import (
    as_bool as _as_bool,
    clamp_float as _clamp_float,
    clean_list as _clean_list,
)
from qq_llm_bot.agent_fact_helpers import fact_candidate as _fact_candidate
from qq_llm_bot.config import AppConfig
from qq_llm_bot.llm import LLMClient
from qq_llm_bot.llm_json_helpers import (
    complete_vision_json as _complete_vision_json,
    vision_direct_image_hint as _vision_direct_image_hint,
)
from qq_llm_bot.models import (
    FactCandidate,
    ImageVisionCacheRecord,
    MemoryCandidate,
    MessageAttachment,
    MessageContext,
    StickerCandidate,
)
from qq_llm_bot.vision_analysis import (
    UNRESOLVED_IMAGE_DESCRIPTION,
    VisionAnalysis,
    VisionCacheStore,
    VisionImageResult,
    attachment_descriptions as _attachment_descriptions,
    clean_image_text as _clean_image_text,
    clean_sticker_text as _clean_sticker_text,
    dedupe_strings as _dedupe_strings,
    fallback_sticker_usage as _fallback_sticker_usage,
    image_interest_topic as _image_interest_topic,
    infer_image_type as _infer_image_type,
    infer_sticker_mood as _infer_sticker_mood,
    looks_like_sticker_image as _looks_like_sticker_image,
    looks_sensitive_image_memory as _looks_sensitive_image_memory,
    safe_image_type as _safe_image_type,
    select_image_attachments as _select_image_attachments,
)


class VisionAgent:
    def __init__(
        self,
        config: AppConfig,
        llm: LLMClient,
        vision_cache: VisionCacheStore | None = None,
    ) -> None:
        self.config = config
        self.llm = llm
        self.vision_cache = vision_cache

    async def analyze(self, context: MessageContext, *, allow_remote: bool = True) -> VisionAnalysis:
        if not self.config.vision.enabled:
            return VisionAnalysis([])
        image_attachments = _select_image_attachments(
            context.attachments,
            self.config.vision.max_images_per_message,
        )
        image_urls = [attachment.url for attachment in image_attachments]
        if not image_urls:
            return VisionAnalysis([])

        fallback = self._fallback_analysis(context)
        cached_results, missing_urls = self._load_cached_results(image_urls)
        fresh_results: dict[str, VisionImageResult] = {}
        if missing_urls and allow_remote:
            sticker_prompt = self._sticker_prompt()
            direct_image_hint = _vision_direct_image_hint(context, missing_urls)
            data = await _complete_vision_json(
                self.llm,
                self.config,
                "你是保守的 QQ 群图片理解器。只输出 JSON，不要解释。",
                (
                    "请解读群聊图片，输出结构化 JSON。"
                    "不要识别或猜测真实人物身份，不要推断敏感个人信息。"
                    "每张图都尽量给出两个核心结果：description 写图片内容大致描述，ocr_text 提取图片内可见文字；"
                    "如果图里没有清晰文字，ocr_text 留空。"
                    "如果是截图，可做简短 OCR；如果是表情包/梗图，可描述梗点。"
                    "长期记忆只记录非隐私、对群聊上下文有帮助的图片观察。"
                    "内容型截图、新闻或网传图片只能记成“群里分享/图片中显示了什么”，"
                    "不要写成外部世界事实已经成立。"
                    f"{sticker_prompt}"
                    "输出 JSON："
                    "如果一条消息里有多张图，你看到的是抽样图；请结合这些图判断这一组图片的大意。"
                    "把每张图分成 image_type：sticker=表情包/梗图/反应图；"
                    "content_image=有可读文字、截图、海报、文档、聊天记录等内容型图片；"
                    "pure_image=没有明显可读文字、主要靠画面本身表达的照片/插画/截图；"
                    "unknown=无法判断。"
                    '{"images":[{"description":"图像简述","ocr_text":"可空","topics":["话题"],'
                    '"image_type":"sticker|content_image|pure_image|unknown",'
                    '"should_remember":bool,"memory":"可空","confidence":0.0,"importance":0.0,'
                    '"is_sticker":bool,"sticker_mood":"可空","sticker_usage":"可空",'
                    '"sticker_tags":["可空"],"sticker_confidence":0.0}]}\n'
                    f"发言人 QQ：{context.user_id}\n"
                    f"随图文字：{context.plain_text or '(none)'}"
                ),
                missing_urls,
                purpose="vision",
                model_tier="flagship" if direct_image_hint else "",
                direct_image_hint=direct_image_hint,
            )
            if data:
                fresh_results = self._parse_fresh_results(missing_urls, data)
                for result in fresh_results.values():
                    self._save_cached_result(result)

        results_by_url = {**cached_results, **fresh_results}
        if not results_by_url:
            return fallback
        return self._build_analysis_from_results(context, image_attachments, image_urls, results_by_url, fallback)

    def _sticker_prompt(self) -> str:
        if not self.config.stickers.enabled:
            return ""
        return (
            "同时判断图片是否适合作为聊天表情包/梗图/反应图长期保存。"
            "只有明显用于表达情绪、吐槽、调侃、安慰、震惊、无语、庆祝等聊天反应时，"
            "is_sticker 才为 true。sticker_usage 写清适合什么时候发，"
            "例如“对方犯困时轻轻吐槽”“大家都在笑时接梗”。"
        )

    def _load_cached_results(
        self,
        image_urls: list[str],
    ) -> tuple[dict[str, VisionImageResult], list[str]]:
        cached: dict[str, VisionImageResult] = {}
        missing: list[str] = []
        for url in _dedupe_strings(image_urls):
            record = self._get_cached_record(url)
            if record and (record.description or record.ocr_text or record.memory):
                is_sticker = (
                    self.config.stickers.enabled
                    and record.confidence >= self.config.stickers.min_confidence
                    and _looks_like_sticker_image(
                        record.description,
                        record.ocr_text,
                        record.topics,
                    )
                )
                cached[url] = VisionImageResult(
                    url=record.url,
                    description=record.description,
                    ocr_text=record.ocr_text,
                    topics=record.topics,
                    image_type=_infer_image_type(record.description, record.ocr_text, record.topics, is_sticker),
                    memory=record.memory,
                    confidence=record.confidence,
                    importance=record.importance,
                    is_sticker=is_sticker,
                    sticker_tags=record.topics,
                    sticker_confidence=record.confidence,
                )
            else:
                missing.append(url)
        return cached, missing

    def _get_cached_record(self, url: str) -> ImageVisionCacheRecord | None:
        if self.vision_cache is None:
            return None
        try:
            return self.vision_cache.get_image_vision_cache(url)
        except Exception as exc:  # pragma: no cover - cache must never break replies
            logger.warning("Image vision cache read failed for {}: {}", url, exc)
            return None

    def _parse_fresh_results(
        self,
        image_urls: list[str],
        data: dict[str, Any],
    ) -> dict[str, VisionImageResult]:
        results: dict[str, VisionImageResult] = {}
        for url, item in zip(image_urls, data.get("images", [])):
            if not isinstance(item, dict):
                continue
            result = self._parse_image_item(url, item)
            if result.description or result.ocr_text or result.memory:
                results[url] = result
        return results

    def _parse_image_item(self, url: str, item: dict[str, Any]) -> VisionImageResult:
        description = _clean_image_text(str(item.get("description", "")))
        ocr_text = _clean_image_text(str(item.get("ocr_text", "")))
        confidence = _clamp_float(item.get("confidence", 0.0))
        importance = _clamp_float(item.get("importance", 0.5))
        topics = tuple(_clean_list(item.get("topics"))[:5])
        memory_text = _clean_image_text(str(item.get("memory", "")))
        should_remember = _as_bool(item.get("should_remember"), False)
        raw_is_sticker = _as_bool(item.get("is_sticker"), False)
        is_sticker = self._is_sticker_item(item, description, ocr_text, topics, confidence)
        image_type = _safe_image_type(str(item.get("image_type", "unknown")).strip())
        if image_type == "unknown":
            image_type = _infer_image_type(description, ocr_text, topics, raw_is_sticker or is_sticker)
        if (
            not should_remember
            or confidence < self.config.vision.remember_threshold
            or _looks_sensitive_image_memory(memory_text)
        ):
            memory_text = ""
        return VisionImageResult(
            url=url,
            description=description,
            ocr_text=ocr_text,
            topics=topics,
            image_type=image_type,
            memory=memory_text,
            confidence=confidence,
            importance=importance,
            is_sticker=is_sticker,
            sticker_mood=_clean_sticker_text(str(item.get("sticker_mood", "")), limit=80),
            sticker_usage=_clean_sticker_text(str(item.get("sticker_usage", "")), limit=240),
            sticker_tags=tuple(_clean_list(item.get("sticker_tags"))[:8]),
            sticker_confidence=_clamp_float(item.get("sticker_confidence", confidence)),
        )

    def _is_sticker_item(
        self,
        item: dict[str, Any],
        description: str,
        ocr_text: str,
        topics: tuple[str, ...],
        confidence: float,
    ) -> bool:
        if not self.config.stickers.enabled:
            return False
        sticker_confidence = _clamp_float(item.get("sticker_confidence", confidence))
        if sticker_confidence < self.config.stickers.min_confidence:
            return False
        if _as_bool(item.get("is_sticker"), False):
            return True
        return _looks_like_sticker_image(description, ocr_text, topics)

    def _save_cached_result(self, result: VisionImageResult) -> None:
        if self.vision_cache is None:
            return
        try:
            self.vision_cache.upsert_image_vision_cache(
                url=result.url,
                description=result.description,
                ocr_text=result.ocr_text,
                topics=result.topics,
                memory=result.memory,
                confidence=result.confidence,
                importance=result.importance,
                model=(
                    self.config.llm.routing.detailed_vision_model
                    or self.config.llm.routing.simple_vision_model
                    or self.config.llm.model
                ),
            )
        except Exception as exc:  # pragma: no cover - cache must never break replies
            logger.warning("Image vision cache write failed for {}: {}", result.url, exc)

    def _build_analysis_from_results(
        self,
        context: MessageContext,
        image_attachments: list[MessageAttachment],
        image_urls: list[str],
        results_by_url: dict[str, VisionImageResult],
        fallback: VisionAnalysis,
    ) -> VisionAnalysis:
        descriptions: list[str] = []
        ocr_parts: list[str] = []
        topics: list[str] = []
        memories: list[MemoryCandidate] = []
        facts: list[FactCandidate] = []
        stickers: list[StickerCandidate] = []
        seen_memory_urls: set[str] = set()
        seen_fact_keys: set[tuple[str, str]] = set()
        seen_ocr_urls: set[str] = set()
        for index, url in enumerate(image_urls):
            result = results_by_url.get(url)
            if result and result.description:
                descriptions.append(result.description)
            elif index < len(fallback.descriptions):
                descriptions.append(fallback.descriptions[index])
            if not result:
                continue
            if result.ocr_text and url not in seen_ocr_urls:
                ocr_parts.append(result.ocr_text)
                seen_ocr_urls.add(url)
            topics.extend(result.topics[:5])
            if (
                result.memory
                and result.confidence >= self.config.vision.remember_threshold
                and url not in seen_memory_urls
                and not _looks_sensitive_image_memory(result.memory)
            ):
                memories.append(self._memory_candidate_from_image(context, index, result))
                seen_memory_urls.add(url)
            fact = self._fact_candidate_from_image_interest(context, index, result)
            if fact:
                fact_key = (fact.subject_user_id, fact.claim_text)
                if fact_key not in seen_fact_keys:
                    facts.append(fact)
                    seen_fact_keys.add(fact_key)
            if result.is_sticker and result.sticker_confidence >= self.config.stickers.min_confidence:
                attachment = image_attachments[index] if index < len(image_attachments) else None
                stickers.append(
                    StickerCandidate(
                        url=result.url,
                        file=attachment.file if attachment is not None else "",
                        description=result.description,
                        ocr_text=result.ocr_text,
                        mood=result.sticker_mood or _infer_sticker_mood(result),
                        usage=result.sticker_usage or _fallback_sticker_usage(result),
                        tags=result.sticker_tags or result.topics,
                        confidence=result.sticker_confidence,
                    )
                )

        return VisionAnalysis(
            descriptions=descriptions or fallback.descriptions,
            ocr_text="\n".join(_dedupe_strings(ocr_parts)),
            topics=tuple(_dedupe_strings(topics)),
            memory_candidates=tuple(memories),
            fact_candidates=tuple(facts),
            sticker_candidates=tuple(stickers),
            attachment_descriptions=_attachment_descriptions(context, results_by_url),
            resolved_image_urls=tuple(
                url
                for url in image_urls
                if (result := results_by_url.get(url)) is not None
                and (result.description or result.ocr_text)
            ),
        )

    def _memory_candidate_from_image(
        self,
        context: MessageContext,
        index: int,
        result: VisionImageResult,
    ) -> MemoryCandidate:
        return MemoryCandidate(
            owner_type="group",
            owner_id=context.group_id,
            kind="image_observation",
            content=result.memory,
            confidence=result.confidence,
            importance=result.importance,
            evidence_message_id=context.message_id,
            source_text=(
                f"image_index={index}\n"
                f"image_url={result.url}\n"
                f"trigger_user={context.user_id}\n"
                f"trigger_message={context.plain_text}\n"
                f"description={result.description}\n"
                f"ocr={result.ocr_text}"
            ),
            source_user_id="bot",
            source_group_id=context.group_id,
            subject_user_id=context.group_id,
            claim_scope="group_fact",
        )

    def _fact_candidate_from_image_interest(
        self,
        context: MessageContext,
        index: int,
        result: VisionImageResult,
    ) -> FactCandidate | None:
        image_type = result.image_type
        if image_type == "unknown":
            image_type = _infer_image_type(result.description, result.ocr_text, result.topics, result.is_sticker)
        if image_type == "sticker" or result.is_sticker:
            return None
        if result.confidence < 0.55 or not result.description:
            return None

        topic = _image_interest_topic(result)
        if not topic:
            return None
        if image_type == "content_image" or result.ocr_text:
            claim_text = f"用户{context.user_id}对图片中的{topic}内容感兴趣"
            evidence_kind = "content_image"
        elif image_type == "pure_image":
            claim_text = f"用户{context.user_id}对{topic}这类图片感兴趣"
            evidence_kind = "pure_image"
        else:
            return None

        evidence_parts = [
            f"image_index={index}",
            f"image_type={evidence_kind}",
            f"description={result.description}",
        ]
        if result.ocr_text:
            evidence_parts.append(f"ocr={result.ocr_text}")
        return _fact_candidate(
            context=context,
            subject_user_id=context.user_id,
            fact_type="preference",
            claim_text=claim_text,
            topic=topic,
            stance="positive",
            confidence=max(0.76, result.confidence),
            claim_scope="self_report",
            evidence_text="\n".join(evidence_parts),
        )

    def _fallback_analysis(self, context: MessageContext) -> VisionAnalysis:
        descriptions = []
        attachment_descriptions = []
        selected = _select_image_attachments(context.attachments, self.config.vision.max_images_per_message)
        selected_urls = {attachment.url for attachment in selected if attachment.url}
        for attachment in [item for item in context.attachments if item.attachment_type == "image"]:
            summary = ""
            if attachment.summary:
                summary = attachment.summary
            elif attachment.url in selected_urls or (not attachment.url and attachment.file):
                summary = UNRESOLVED_IMAGE_DESCRIPTION
            attachment_descriptions.append(summary)
            if summary and (not attachment.url or attachment.url in selected_urls):
                descriptions.append(summary)
        return VisionAnalysis(
            descriptions[: self.config.vision.max_images_per_message],
            attachment_descriptions=tuple(attachment_descriptions),
        )
