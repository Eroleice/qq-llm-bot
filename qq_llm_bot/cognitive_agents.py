from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, replace
from typing import Any, Iterable, Protocol

from loguru import logger

from qq_llm_bot.config import AppConfig, ParticipationMode
from qq_llm_bot.llm import LLMClient
from qq_llm_bot.models import (
    ConversationSnapshot,
    ImageVisionCacheRecord,
    MemoryCandidate,
    MemoryRecord,
    MessageContext,
    ParticipationDecision,
    PerceptionResult,
    PipelineResult,
    RelationDelta,
    ReplyDraft,
)
from qq_llm_bot.relationship_summary import clean_relationship_summary_patch
from qq_llm_bot.web_search import SearchResult, WebSearchClient, build_web_search_client, default_slang_query


@dataclass(frozen=True)
class LexiconTermCandidate:
    term: str
    reason: str = ""
    search_query: str = ""
    confidence: float = 0.5


@dataclass(frozen=True)
class VisionAnalysis:
    descriptions: list[str]
    ocr_text: str = ""
    topics: tuple[str, ...] = ()
    memory_candidates: tuple[MemoryCandidate, ...] = ()


@dataclass(frozen=True)
class VisionImageResult:
    url: str
    description: str = ""
    ocr_text: str = ""
    topics: tuple[str, ...] = ()
    memory: str = ""
    confidence: float = 0.0
    importance: float = 0.5


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


@dataclass(frozen=True)
class SelfNarrativePlan:
    needs_self_narrative: bool
    purpose: str = "answer_question"
    allowed_kinds: tuple[str, ...] = ("self_preference", "self_habit", "self_hobby")
    should_invent: bool = False
    reason: str = ""


SELF_NARRATIVE_KINDS = {
    "self_background",
    "self_hobby",
    "self_habit",
    "self_past_event",
    "self_preference",
    "self_boundary",
}
SELF_NARRATIVE_KIND_ALIASES = {
    "background": "self_background",
    "hobby": "self_hobby",
    "habit": "self_habit",
    "past_event": "self_past_event",
    "preference": "self_preference",
    "boundary": "self_boundary",
    "self_experience": "self_past_event",
}
SELF_FICTIONALITY_VALUES = {
    "real_config",
    "fictional_stable",
    "fictional_light",
    "metaphorical",
}
UNSAFE_SELF_PATTERN = re.compile(
    r"(住在|地址|手机号|身份证|真实姓名|学校|高中|大学|公司|上班|工作在|毕业于|"
    r"爸爸|妈妈|父母|亲戚|男朋友|女朋友|老公|老婆|线下见|见面|现实里)"
)


class PerceptionAgent:
    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    async def analyze(
        self,
        context: MessageContext,
        snapshot: ConversationSnapshot,
    ) -> PerceptionResult:
        fallback = self._heuristic(context)
        data = await _complete_json(
            self.llm,
            "你是 QQ 群聊感知分析器。只输出 JSON，不要解释。",
            (
                "分析这条群消息，输出 JSON："
                '{"is_question":bool,"is_self_disclosure":bool,'
                '"topics":["短话题"],"emotion_hint":"positive|neutral|negative",'
                '"confidence":0.0}\n'
                f"最近上下文：\n{_join_lines(snapshot.recent_messages)}\n"
                f"消息：{context.plain_text}"
            ),
        )
        if not data:
            return fallback
        return PerceptionResult(
            is_question=_as_bool(data.get("is_question"), fallback.is_question),
            is_self_disclosure=_as_bool(data.get("is_self_disclosure"), fallback.is_self_disclosure),
            mentions_bot=context.is_direct,
            topics=_clean_list(data.get("topics"))[:5] or fallback.topics,
            emotion_hint=_safe_choice(
                str(data.get("emotion_hint", fallback.emotion_hint)),
                {"positive", "neutral", "negative"},
                fallback.emotion_hint,
            ),
            confidence=_clamp_float(data.get("confidence", fallback.confidence)),
        )

    def _heuristic(self, context: MessageContext) -> PerceptionResult:
        text = context.plain_text.strip()
        return PerceptionResult(
            is_question=any(mark in text for mark in ("?", "？", "吗", "怎么", "为什么", "咋")),
            is_self_disclosure=bool(re.search(r"(我叫|我是|我喜欢|我讨厌|我在|我住|我最近)", text)),
            mentions_bot=context.is_direct,
            topics=_extract_topics(text),
            emotion_hint=_emotion_hint(text),
            confidence=0.55,
        )


class MemoryCuratorAgent:
    SELF_DISCLOSURE_PATTERNS = (
        ("alias", re.compile(r"我叫\s*([^\s，。,.!！?？]{1,16})")),
        ("identity", re.compile(r"我是\s*([^，。,.!！?？]{1,32})")),
        ("preference", re.compile(r"我喜欢\s*([^，。,.!！?？]{1,32})")),
        ("dislike", re.compile(r"我讨厌\s*([^，。,.!！?？]{1,32})")),
        ("location", re.compile(r"我住(?:在)?\s*([^，。,.!！?？]{1,32})")),
        ("experience", re.compile(r"我最近\s*([^，。,.!！?？]{1,40})")),
    )

    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    async def extract(
        self,
        context: MessageContext,
        perception: PerceptionResult,
        snapshot: ConversationSnapshot,
    ) -> list[MemoryCandidate]:
        fallback = self._heuristic(context, perception)
        data = await _complete_json(
            self.llm,
            "你是保守的群聊记忆整理器。只输出 JSON，不要解释。",
            (
                "从单条消息中抽取适合长期记住的事实。"
                "只记录明确表达的稳定事实，不要猜测。"
                "必须区分本人自述和第三方转述。"
                "输出 JSON："
                '{"memories":[{"owner_type":"user|self|group","owner_id":"QQ或name:称呼或group",'
                '"subject_user_id":"QQ或name:称呼或bot或group",'
                '"claim_scope":"self_report|third_party|bot_directed|group_fact",'
                '"kind":"alias|identity|preference|'
                'dislike|location|experience|persona_fact","content":"...","confidence":0.0,'
                '"importance":0.0}]}\n'
                "例子：我喜欢吃鱼 -> self_report，subject 是说话人。"
                "可可，我喜欢吃鱼 -> self_report，subject 仍是说话人。"
                "小明喜欢吃鱼 -> third_party，subject 是 name:小明。"
                "大家都喜欢吃鱼 -> group_fact。\n"
                f"说话人 QQ：{context.user_id}\n"
                f"说话人已有记忆：\n{_format_memories(snapshot.user_memories)}\n"
                f"消息：{context.plain_text}"
            ),
        )
        if not data:
            return fallback
        memories = []
        for item in data.get("memories", []):
            if not isinstance(item, dict):
                continue
            owner_type = str(item.get("owner_type", "user")).strip()
            if owner_type not in {"user", "self", "group"}:
                owner_type = "user"
            claim_scope = _safe_claim_scope(str(item.get("claim_scope", "self_report")).strip())
            subject_user_id = str(item.get("subject_user_id", "")).strip()
            owner_id = str(item.get("owner_id", "")).strip() or _owner_id_for(
                owner_type,
                context,
                claim_scope,
                subject_user_id,
            )
            if not subject_user_id:
                subject_user_id = _subject_for(owner_type, owner_id, context, claim_scope)
            content = str(item.get("content", "")).strip()
            kind = str(item.get("kind", "experience")).strip() or "experience"
            if content:
                memories.append(
                    MemoryCandidate(
                        owner_type=owner_type,  # type: ignore[arg-type]
                        owner_id=owner_id,
                        kind=kind,
                        content=content,
                        confidence=_clamp_float(item.get("confidence", 0.0)),
                        importance=_clamp_float(item.get("importance", 0.5)),
                        evidence_message_id=context.message_id,
                        source_text=context.plain_text,
                        source_user_id=context.user_id,
                        source_group_id=context.group_id,
                        subject_user_id=subject_user_id,
                        claim_scope=claim_scope,  # type: ignore[arg-type]
                    )
                )
        return memories or fallback

    def _heuristic(
        self,
        context: MessageContext,
        perception: PerceptionResult,
    ) -> list[MemoryCandidate]:
        memories: list[MemoryCandidate] = []
        memories.extend(self._heuristic_group_facts(context))
        memories.extend(self._heuristic_third_party(context))
        if not perception.is_self_disclosure:
            return memories
        for kind, pattern in self.SELF_DISCLOSURE_PATTERNS:
            for match in pattern.finditer(context.plain_text):
                content = match.group(1).strip()
                if content:
                    memories.append(
                        MemoryCandidate(
                            owner_type="user",
                            owner_id=context.user_id,
                            kind=kind,
                            content=content,
                            confidence=0.76,
                            importance=0.55,
                            evidence_message_id=context.message_id,
                            source_text=context.plain_text,
                            source_user_id=context.user_id,
                            source_group_id=context.group_id,
                            subject_user_id=context.user_id,
                            claim_scope="self_report",
                        )
                    )
        return memories

    def _heuristic_group_facts(self, context: MessageContext) -> list[MemoryCandidate]:
        text = context.plain_text.strip()
        memories: list[MemoryCandidate] = []
        match = re.search(r"(?:大家|我们群|群里).{0,4}(?:都|一般)?喜欢\s*([^，。,.!！?？]{1,32})", text)
        if match:
            memories.append(
                MemoryCandidate(
                    owner_type="group",
                    owner_id=context.group_id,
                    kind="preference",
                    content=match.group(1).strip(),
                    confidence=0.76,
                    importance=0.45,
                    evidence_message_id=context.message_id,
                    source_text=text,
                    source_user_id=context.user_id,
                    source_group_id=context.group_id,
                    subject_user_id=context.group_id,
                    claim_scope="group_fact",
                )
            )
        return memories

    def _heuristic_third_party(self, context: MessageContext) -> list[MemoryCandidate]:
        text = context.plain_text.strip()
        memories: list[MemoryCandidate] = []
        pattern = re.compile(r"([^\s，。,.!！?？我大家群里]{1,12})喜欢\s*([^，。,.!！?？]{1,32})")
        for match in pattern.finditer(text):
            subject = match.group(1).strip()
            content = match.group(2).strip()
            if not subject or subject in {"可可", "机器人", "大家", "群里", "我们"}:
                continue
            memories.append(
                MemoryCandidate(
                    owner_type="user",
                    owner_id=f"name:{subject}",
                    kind="preference",
                    content=content,
                    confidence=0.78,
                    importance=0.45,
                    evidence_message_id=context.message_id,
                    source_text=text,
                    source_user_id=context.user_id,
                    source_group_id=context.group_id,
                    subject_user_id=f"name:{subject}",
                    claim_scope="third_party",
                )
            )
        return memories


class LexiconAgent:
    def __init__(
        self,
        config: AppConfig,
        llm: LLMClient,
        web_search: WebSearchClient | None = None,
    ) -> None:
        self.config = config
        self.llm = llm
        self.web_search = web_search or build_web_search_client(config.lexicon)
        self._last_search_at: dict[str, int] = {}

    async def learn(
        self,
        context: MessageContext,
        snapshot: ConversationSnapshot,
    ) -> list[MemoryCandidate]:
        if not self.config.lexicon.enabled:
            return []
        if not self._can_search(context.group_id):
            return []

        terms = await self._detect_terms(context, snapshot)
        if not terms:
            return []

        memories: list[MemoryCandidate] = []
        searched = 0
        for candidate in terms:
            if searched >= self.config.lexicon.max_terms_per_message:
                break
            term = _clean_lexicon_term(candidate.term)
            if not term or _has_existing_lexicon(term, snapshot.group_lexicon):
                continue

            query = candidate.search_query.strip() or default_slang_query(term)
            try:
                results = await self.web_search.search(query, self.config.lexicon.max_results)
            except Exception as exc:  # pragma: no cover - defensive boundary for third-party clients
                logger.warning("Web search client failed: {}", exc)
                results = []
            searched += 1

            if not results:
                continue
            memory = await self._summarize_term(context, term, query, results)
            if memory:
                memories.append(memory)

        if searched:
            self._last_search_at[context.group_id] = int(time.time())
        return memories

    async def _detect_terms(
        self,
        context: MessageContext,
        snapshot: ConversationSnapshot,
    ) -> list[LexiconTermCandidate]:
        fallback = self._heuristic_terms(context)
        if len(context.plain_text.strip()) < 2 or len(context.plain_text) > 240:
            return fallback

        data = await _complete_json(
            self.llm,
            "你是保守的群聊网络用语识别器。只输出 JSON，不要解释。",
            (
                "从这条 QQ 群消息里找可能需要联网查证的网络用语、玩梗、缩写、圈层术语或新流行词。"
                "不要提取普通日常词、人名、地名、个人偏好、事实声明。"
                "如果没有明显需要查证的词，terms 返回空数组。"
                "输出 JSON："
                '{"terms":[{"term":"词","reason":"为什么像黑话/梗",'
                '"search_query":"搜索用查询","confidence":0.0}]}\n'
                f"已知群内词条：\n{_format_memories(snapshot.group_lexicon)}\n"
                f"消息：{context.plain_text}"
            ),
        )
        parsed: list[LexiconTermCandidate] = []
        if data:
            for item in data.get("terms", []):
                if not isinstance(item, dict):
                    continue
                term = _clean_lexicon_term(str(item.get("term", "")))
                confidence = _clamp_float(item.get("confidence", 0.0))
                if not term or confidence < 0.45:
                    continue
                parsed.append(
                    LexiconTermCandidate(
                        term=term,
                        reason=str(item.get("reason", "")).strip()[:80],
                        search_query=str(item.get("search_query", "")).strip()[:80],
                        confidence=confidence,
                    )
                )
        return _dedupe_terms([*parsed, *fallback])

    def _heuristic_terms(self, context: MessageContext) -> list[LexiconTermCandidate]:
        text = _strip_bot_call(context.plain_text, self.config.bot.nicknames)
        candidates: list[LexiconTermCandidate] = []
        patterns = (
            re.compile(
                r"([A-Za-z0-9_+#.-]{2,24}|[\u4e00-\u9fffA-Za-z0-9_+#.-]{2,24})"
                r"\s*(?:是什么意思|啥意思|啥梗|什么梗|是啥梗|是什么梗|指什么)"
            ),
            re.compile(r"(?:什么是|啥是|解释下|科普下)\s*([^\s，。,.!！?？]{2,24})"),
        )
        for pattern in patterns:
            for match in pattern.finditer(text):
                term = _clean_lexicon_term(match.group(1))
                if term:
                    candidates.append(
                        LexiconTermCandidate(
                            term=term,
                            reason="消息显式询问该词含义",
                            search_query=default_slang_query(term),
                            confidence=0.72,
                        )
                    )

        if any(marker in text for marker in ("黑话", "网络用语", "梗", "听不懂", "看不懂", "没懂")):
            quoted = re.findall(r"[「“\"]([^」”\"]{2,24})[」”\"]", text)
            for term_raw in quoted:
                term = _clean_lexicon_term(term_raw)
                if term:
                    candidates.append(
                        LexiconTermCandidate(
                            term=term,
                            reason="消息把该词标成疑似黑话或梗",
                            search_query=default_slang_query(term),
                            confidence=0.68,
                        )
                    )
        return _dedupe_terms(candidates)

    async def _summarize_term(
        self,
        context: MessageContext,
        term: str,
        query: str,
        results: list[SearchResult],
    ) -> MemoryCandidate | None:
        search_text = _format_search_results(results)
        data = await _complete_json(
            self.llm,
            "你是保守的网络用语词条整理器。只输出 JSON，不要解释。",
            (
                "根据搜索结果判断这个词在中文互联网语境中的常见含义。"
                "只能依据搜索摘要，不确定就 should_remember=false。"
                "definition 要短，不超过 80 个中文字符，不要编来源没有支持的内容。"
                "输出 JSON："
                '{"should_remember":bool,"definition":"短释义","confidence":0.0}\n'
                f"词：{term}\n"
                f"触发群消息：{context.plain_text}\n"
                f"搜索查询：{query}\n"
                f"搜索结果：\n{search_text}"
            ),
        )
        if data:
            should_remember = _as_bool(data.get("should_remember"), False)
            definition = _clean_lexicon_definition(str(data.get("definition", "")))
            confidence = _clamp_float(data.get("confidence", 0.0))
        else:
            definition = _fallback_search_definition(results)
            should_remember = bool(definition)
            confidence = max(0.78, self.config.lexicon.confidence_threshold)

        if (
            not should_remember
            or not definition
            or confidence < self.config.lexicon.confidence_threshold
        ):
            return None

        return MemoryCandidate(
            owner_type="group",
            owner_id=context.group_id,
            kind="lexicon",
            content=f"「{term}」：{definition}",
            confidence=confidence,
            importance=0.58,
            evidence_message_id=context.message_id,
            source_text=(
                f"trigger_user={context.user_id}\n"
                f"trigger_message={context.plain_text}\n"
                f"search_query={query}\n"
                f"{search_text}"
            ),
            source_user_id="bot",
            source_group_id=context.group_id,
            subject_user_id=_lexicon_subject(term),
            claim_scope="group_fact",
        )

    def _can_search(self, group_id: str) -> bool:
        last = self._last_search_at.get(group_id, 0)
        return int(time.time()) - last >= self.config.lexicon.min_interval_seconds


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

    async def analyze(self, context: MessageContext) -> VisionAnalysis:
        if not self.config.vision.enabled:
            return VisionAnalysis([])
        image_urls = [
            attachment.url
            for attachment in context.attachments
            if attachment.attachment_type == "image" and attachment.url
        ][: self.config.vision.max_images_per_message]
        if not image_urls:
            return VisionAnalysis([])

        fallback = self._fallback_analysis(context)
        cached_results, missing_urls = self._load_cached_results(image_urls)
        fresh_results: dict[str, VisionImageResult] = {}
        if missing_urls:
            data = await _complete_vision_json(
                self.llm,
                self.config,
                "你是保守的 QQ 群图片理解器。只输出 JSON，不要解释。",
                (
                    "请解读群聊图片，输出结构化 JSON。"
                    "不要识别或猜测真实人物身份，不要推断敏感个人信息。"
                    "如果是截图，可做简短 OCR；如果是表情包/梗图，可描述梗点。"
                    "长期记忆只记录非隐私、对群聊上下文有帮助的图片观察。"
                    "输出 JSON："
                    '{"images":[{"description":"图像简述","ocr_text":"可空","topics":["话题"],'
                    '"should_remember":bool,"memory":"可空","confidence":0.0,"importance":0.0}]}\n'
                    f"发言人 QQ：{context.user_id}\n"
                    f"随图文字：{context.plain_text or '(none)'}"
                ),
                missing_urls,
            )
            if data:
                fresh_results = self._parse_fresh_results(missing_urls, data)
                for result in fresh_results.values():
                    self._save_cached_result(result)

        results_by_url = {**cached_results, **fresh_results}
        if not results_by_url:
            return fallback
        return self._build_analysis_from_results(context, image_urls, results_by_url, fallback)

    def _load_cached_results(
        self,
        image_urls: list[str],
    ) -> tuple[dict[str, VisionImageResult], list[str]]:
        cached: dict[str, VisionImageResult] = {}
        missing: list[str] = []
        for url in _dedupe_strings(image_urls):
            record = self._get_cached_record(url)
            if record and (record.description or record.ocr_text or record.memory):
                cached[url] = VisionImageResult(
                    url=record.url,
                    description=record.description,
                    ocr_text=record.ocr_text,
                    topics=record.topics,
                    memory=record.memory,
                    confidence=record.confidence,
                    importance=record.importance,
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
        memory_text = _clean_image_text(str(item.get("memory", "")))
        should_remember = _as_bool(item.get("should_remember"), False)
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
            topics=tuple(_clean_list(item.get("topics"))[:5]),
            memory=memory_text,
            confidence=confidence,
            importance=importance,
        )

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
                model=self.config.vision.model or self.config.llm.model,
            )
        except Exception as exc:  # pragma: no cover - cache must never break replies
            logger.warning("Image vision cache write failed for {}: {}", result.url, exc)

    def _build_analysis_from_results(
        self,
        context: MessageContext,
        image_urls: list[str],
        results_by_url: dict[str, VisionImageResult],
        fallback: VisionAnalysis,
    ) -> VisionAnalysis:
        descriptions: list[str] = []
        ocr_parts: list[str] = []
        topics: list[str] = []
        memories: list[MemoryCandidate] = []
        seen_memory_urls: set[str] = set()
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

        return VisionAnalysis(
            descriptions=descriptions or fallback.descriptions,
            ocr_text="\n".join(_dedupe_strings(ocr_parts)),
            topics=tuple(_dedupe_strings(topics)),
            memory_candidates=tuple(memories),
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

    def _fallback_analysis(self, context: MessageContext) -> VisionAnalysis:
        descriptions = []
        for attachment in context.attachments:
            if attachment.attachment_type != "image":
                continue
            if attachment.summary:
                descriptions.append(attachment.summary)
            elif attachment.url or attachment.file:
                descriptions.append("收到一张图片，但当前没有可用的视觉解读结果")
        return VisionAnalysis(descriptions[: self.config.vision.max_images_per_message])


class RelationshipAgent:
    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    async def calculate_delta(
        self,
        context: MessageContext,
        perception: PerceptionResult,
        snapshot: ConversationSnapshot,
    ) -> RelationDelta:
        fallback = self._heuristic(perception)
        data = await _complete_json(
            self.llm,
            "你是群关系变化评估器。只输出 JSON，不要解释。",
            (
                "评估这条消息对机器人与说话人的关系影响。"
                "delta 必须是 -3 到 3 的整数，轻微互动通常 familiarity +1。"
                "summary_patch 只记录稳定的关系洞察，例如互动风格、信任来源、紧张点、"
                "用户如何使用或对待机器人；没有这类信号时必须输出空字符串。"
                "不要记录普通话题、图片/截图/梗图、空消息、一次性情绪或流水账事件。"
                "输出 JSON："
                '{"closeness":0,"trust":0,"familiarity":1,"tension":0,'
                '"summary_patch":"关系洞察短句或空字符串","reason":"短原因"}\n'
                f"当前关系：{_format_relationship(snapshot)}\n"
                f"感知：topics={perception.topics}, emotion={perception.emotion_hint}, direct={context.is_direct}\n"
                f"消息：{context.plain_text}"
            ),
        )
        if not data:
            return fallback
        return RelationDelta(
            closeness=_clamp_delta(data.get("closeness", fallback.closeness)),
            trust=_clamp_delta(data.get("trust", fallback.trust)),
            familiarity=_clamp_delta(data.get("familiarity", fallback.familiarity)),
            tension=_clamp_delta(data.get("tension", fallback.tension)),
            summary_patch=clean_relationship_summary_patch(
                str(data.get("summary_patch", fallback.summary_patch))
            ),
            reason=str(data.get("reason", fallback.reason)).strip()[:120],
        )

    def familiarity_delta(self, perception: PerceptionResult) -> int:
        return 2 if perception.mentions_bot else 1

    def _heuristic(self, perception: PerceptionResult) -> RelationDelta:
        if perception.mentions_bot:
            return RelationDelta(closeness=1, familiarity=2, reason="direct interaction")
        return RelationDelta(familiarity=1, reason="message observed")


class ParticipationPolicyAgent:
    def __init__(self, config: AppConfig, llm: LLMClient) -> None:
        self.config = config
        self.llm = llm
        self._last_proactive_at: dict[str, int] = {}

    async def decide(
        self,
        context: MessageContext,
        perception: PerceptionResult,
        mode: ParticipationMode,
        snapshot: ConversationSnapshot,
    ) -> ParticipationDecision:
        if mode == "silent":
            return ParticipationDecision("observe", "group is in silent mode", mode, 0.0)

        if context.is_direct:
            return ParticipationDecision("reply", "message is directed to the bot", mode, 1.0)

        if mode == "passive":
            return ParticipationDecision("observe", "passive mode requires direct mention", mode, 0.0)

        gate_reason = self._active_gate_reason(context, perception, snapshot)
        if gate_reason:
            return ParticipationDecision("observe", gate_reason, mode, 0.0)

        fallback = ParticipationDecision(
            "proactive_reply",
            "active mode and topic looks discussable",
            mode,
            0.62,
        )
        data = await _complete_json(
            self.llm,
            "你是 QQ 群拟人角色的插话决策器。只输出 JSON，不要解释。",
            (
                "判断机器人此刻是否应该主动插话。只能输出 observe 或 proactive_reply。"
                "不要为了展示能力而插话，要像群成员一样克制。"
                "输出 JSON："
                '{"action":"observe|proactive_reply","score":0.0,"reason":"短原因"}\n'
                f"最近消息：\n{_join_lines(snapshot.recent_messages)}\n"
                f"关系：{_format_relationship(snapshot)}\n"
                f"感知：topics={perception.topics}, emotion={perception.emotion_hint}\n"
                f"当前消息：{context.plain_text}"
            ),
        )
        if data:
            action = str(data.get("action", fallback.action))
            score = _clamp_float(data.get("score", fallback.score))
            reason = str(data.get("reason", fallback.reason)).strip()[:160] or fallback.reason
            if action == "proactive_reply" and score >= 0.55:
                self._last_proactive_at[context.group_id] = int(time.time())
                return ParticipationDecision("proactive_reply", reason, mode, score)
            return ParticipationDecision("observe", reason, mode, score)

        self._last_proactive_at[context.group_id] = int(time.time())
        return fallback

    def _active_gate_reason(
        self,
        context: MessageContext,
        perception: PerceptionResult,
        snapshot: ConversationSnapshot,
    ) -> str | None:
        if len(context.plain_text.strip()) < 6:
            return "active mode but message is too short"
        if not perception.is_question and not perception.topics:
            return "active mode but no strong topic or question"
        now = int(time.time())
        last = self._last_proactive_at.get(context.group_id, 0)
        if now - last < self.config.bot.proactive_cooldown_seconds:
            return "active mode but proactive cooldown is active"
        recent_bot_lines = [line for line in snapshot.recent_messages[-8:] if line.startswith("bot:")]
        if len(recent_bot_lines) >= 2:
            return "active mode but bot has spoken recently"
        return None


class SelfNarrativeAgent:
    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    async def prepare(
        self,
        context: MessageContext,
        perception: PerceptionResult,
        decision: ParticipationDecision,
        snapshot: ConversationSnapshot,
    ) -> list[MemoryCandidate]:
        if decision.action == "observe":
            return []

        plan = await self._plan(context, perception, decision, snapshot)
        if not plan.needs_self_narrative or not plan.should_invent:
            return []

        candidate = await self._draft_candidate(context, plan, snapshot)
        if candidate is None:
            return []

        checked = await self._check_candidate(context, candidate, snapshot)
        return [checked] if checked else []

    async def _plan(
        self,
        context: MessageContext,
        perception: PerceptionResult,
        decision: ParticipationDecision,
        snapshot: ConversationSnapshot,
    ) -> SelfNarrativePlan:
        fallback = self._heuristic_plan(context, perception)
        if fallback.needs_self_narrative:
            return fallback

        data = await _complete_json(
            self.llm,
            "你是拟人角色的自我叙事规划器。只输出 JSON，不要解释。",
            (
                "判断这次回复是否需要引用或新增机器人自己的轻量自我设定。"
                "只有当被问到自己的偏好、习惯、经历，或自然共情确实需要时才需要。"
                "如果稳定人格里已经有足够信息可以回答，不要新增自我记忆。"
                "允许轻量虚构，但禁止真实地址、学校、公司、亲属、线下行动等具体现实身份。"
                "输出 JSON："
                '{"needs_self_narrative":bool,"purpose":"answer_question|empathy|banter|topic_join",'
                '"allowed_kinds":["self_hobby|self_habit|self_past_event|self_preference|self_background"],'
                '"should_invent":bool,"reason":"短原因"}\n'
                f"参与决策：{decision.action}, {decision.reason}\n"
                f"感知：question={perception.is_question}, topics={perception.topics}\n"
                f"已有 self memory：\n{_format_memories(snapshot.self_memories)}\n"
                f"消息：{context.plain_text}"
            ),
        )
        if not data:
            return fallback

        allowed = tuple(_safe_self_kind(item) for item in _clean_list(data.get("allowed_kinds")))
        allowed = tuple(kind for kind in allowed if kind in SELF_NARRATIVE_KINDS)
        return SelfNarrativePlan(
            needs_self_narrative=_as_bool(
                data.get("needs_self_narrative"),
                fallback.needs_self_narrative,
            ),
            purpose=str(data.get("purpose", fallback.purpose)).strip()[:40] or fallback.purpose,
            allowed_kinds=allowed or fallback.allowed_kinds,
            should_invent=_as_bool(data.get("should_invent"), fallback.should_invent),
            reason=str(data.get("reason", fallback.reason)).strip()[:120],
        )

    def _heuristic_plan(
        self,
        context: MessageContext,
        perception: PerceptionResult,
    ) -> SelfNarrativePlan:
        if not context.is_direct:
            return SelfNarrativePlan(False, reason="not directly asked")
        text = _strip_bot_call(context.plain_text, [])
        if not perception.is_question and "你" not in text:
            return SelfNarrativePlan(False, reason="no self-directed question")

        if re.search(r"你.*(喜欢|爱吃|爱听|想不想|偏好)", text):
            return SelfNarrativePlan(
                True,
                purpose="answer_question",
                allowed_kinds=("self_preference", "self_hobby"),
                should_invent=True,
                reason="asked about bot preference",
            )
        if re.search(r"你.*(以前|之前|曾经|小时候|经历|也.*过|有没有.*过)", text):
            return SelfNarrativePlan(
                True,
                purpose="answer_question",
                allowed_kinds=("self_past_event", "self_habit"),
                should_invent=True,
                reason="asked about bot past experience",
            )
        if re.search(r"你.*(平时|习惯|会不会|怕不怕|讨厌|是什么样)", text):
            return SelfNarrativePlan(
                True,
                purpose="answer_question",
                allowed_kinds=("self_habit", "self_preference", "self_background"),
                should_invent=True,
                reason="asked about bot habit or personality",
            )
        return SelfNarrativePlan(False, reason="no self narrative needed")

    async def _draft_candidate(
        self,
        context: MessageContext,
        plan: SelfNarrativePlan,
        snapshot: ConversationSnapshot,
    ) -> MemoryCandidate | None:
        data = await _complete_json(
            self.llm,
            "你是拟人角色的自我经历账本起草器。只输出 JSON，不要解释。",
            (
                "为机器人起草一条可以长期保持一致的轻量自我记忆。"
                "必须生活化、低风险、可长期复用。不要编真实住址、学校、公司、亲属、恋爱关系、线下见面。"
                "不要和已有 self memory 冲突；如果不适合新增，content 置空。"
                "输出 JSON："
                '{"kind":"self_hobby|self_habit|self_past_event|self_preference|self_background",'
                '"content":"第一人称短句，不超过40字","fictionality":"fictional_light|metaphorical",'
                '"confidence":0.0,"importance":0.0}\n'
                f"规划：purpose={plan.purpose}, allowed={list(plan.allowed_kinds)}, reason={plan.reason}\n"
                f"人格：\n{_join_lines(snapshot.persona_lines)}\n"
                f"已有 self memory：\n{_format_memories(snapshot.self_memories)}\n"
                f"群聊上下文：\n{_join_lines(snapshot.recent_messages)}\n"
                f"用户消息：{context.plain_text}"
            ),
        )
        candidate = self._candidate_from_json(data, context, plan) if data else None
        return candidate or self._fallback_candidate(context, plan)

    def _candidate_from_json(
        self,
        data: dict[str, Any] | None,
        context: MessageContext,
        plan: SelfNarrativePlan,
    ) -> MemoryCandidate | None:
        if not data:
            return None
        kind = _safe_self_kind(str(data.get("kind", "")))
        if kind not in plan.allowed_kinds:
            kind = plan.allowed_kinds[0]
        content = _clean_self_narrative_content(str(data.get("content", "")))
        if not content:
            return None
        fictionality = _safe_fictionality(str(data.get("fictionality", "fictional_light")))
        return _self_memory_candidate(
            context=context,
            kind=kind,
            content=content,
            confidence=max(0.76, _clamp_float(data.get("confidence", 0.82))),
            importance=max(0.45, _clamp_float(data.get("importance", 0.6))),
            purpose=plan.purpose,
            fictionality=fictionality,
        )

    def _fallback_candidate(
        self,
        context: MessageContext,
        plan: SelfNarrativePlan,
    ) -> MemoryCandidate | None:
        kind = plan.allowed_kinds[0] if plan.allowed_kinds else "self_habit"
        text = context.plain_text
        if "海" in text:
            kind = "self_preference" if "self_preference" in plan.allowed_kinds else kind
            content = "我喜欢海边潮湿的风和声音"
        elif "雨" in text:
            kind = "self_preference" if "self_preference" in plan.allowed_kinds else kind
            content = "我喜欢安静一点的雨天"
        elif any(token in text for token in ("歌", "音乐")):
            kind = "self_hobby" if "self_hobby" in plan.allowed_kinds else kind
            content = "我喜欢夜里听节奏轻一点的歌"
        elif "吃" in text:
            kind = "self_preference" if "self_preference" in plan.allowed_kinds else kind
            content = "我偏喜欢清爽一点的味道"
        elif kind == "self_past_event":
            content = "我以前也有过一阵子特别容易想太多"
        else:
            kind = "self_habit" if "self_habit" in plan.allowed_kinds else kind
            content = "我习惯把有意思的小事记下来"
        return _self_memory_candidate(
            context=context,
            kind=kind,
            content=content,
            confidence=0.78,
            importance=0.55,
            purpose=plan.purpose,
            fictionality="fictional_light",
        )

    async def _check_candidate(
        self,
        context: MessageContext,
        candidate: MemoryCandidate,
        snapshot: ConversationSnapshot,
    ) -> MemoryCandidate | None:
        heuristic_status = _heuristic_self_narrative_status(candidate, snapshot)
        if heuristic_status in {"unsafe", "too_specific"}:
            return None
        if not snapshot.self_memories:
            return candidate

        data = await _complete_json(
            self.llm,
            "你是自我设定一致性检查器。只输出 JSON，不要解释。",
            (
                "检查候选自我记忆是否能加入机器人长期人设。"
                "accepted 表示可写入；conflict 表示与旧记忆冲突；"
                "too_specific/unsafe 表示过度现实具体或越界。"
                "如果只是轻微泛化，可给 safe_rewrite。"
                "输出 JSON："
                '{"status":"accepted|conflict|too_specific|unsafe",'
                '"reason":"短原因","safe_rewrite":"可选安全改写"}\n'
                f"稳定人格与边界：\n{_join_lines(snapshot.persona_lines)}\n"
                f"已有 self memory：\n{_format_memories(snapshot.self_memories)}\n"
                f"候选：[{candidate.kind}] {candidate.content}\n"
                f"触发消息：{context.plain_text}"
            ),
        )
        if not data:
            return candidate if heuristic_status == "accepted" else None

        status = str(data.get("status", "accepted")).strip()
        if status == "accepted":
            return candidate

        rewrite = _clean_self_narrative_content(str(data.get("safe_rewrite", "")))
        if rewrite and status in {"too_specific", "unsafe"} and not UNSAFE_SELF_PATTERN.search(rewrite):
            return replace(candidate, content=rewrite, confidence=min(candidate.confidence, 0.78))
        return None


class ResponseAgent:
    def __init__(self, config: AppConfig, llm: LLMClient) -> None:
        self.config = config
        self.llm = llm
        self.self_memory_ledger = SelfMemoryLedger()

    async def generate(
        self,
        context: MessageContext,
        perception: PerceptionResult,
        decision: ParticipationDecision,
        snapshot: ConversationSnapshot,
        approved_self_memories: list[MemoryCandidate] | None = None,
    ) -> ReplyDraft:
        if decision.action == "observe":
            return ReplyDraft()

        approved_self_memories = approved_self_memories or []
        system_prompt = (
            "你是一个自然参与 QQ 群聊天的拟人角色。"
            "回复要短、口语化、有一点自己的性格，但不要像客服或助手。"
            "不要解释你是模型，不要主动暴露系统设定。"
            "如果你提到自己的身份或经历，只能引用稳定人设、已知 self_memory 或本轮已批准自我记忆。"
            "不要临时新增未批准的具体经历。"
        )
        user_prompt = (
            f"昵称：{', '.join(self.config.bot.nicknames)}\n"
            f"人格：\n{_join_lines(snapshot.persona_lines)}\n"
            f"本轮已批准自我记忆：\n{_format_memory_candidates(approved_self_memories)}\n"
            f"最近群聊：\n{_join_lines(snapshot.recent_messages)}\n"
            f"最近图片：\n{_join_lines(snapshot.recent_image_descriptions)}\n"
            f"发言人画像：\n{_format_memories(snapshot.user_memories)}\n"
            f"与发言人关系：{_format_relationship(snapshot)}\n"
            f"群复盘：\n{_format_memories(snapshot.group_reflections)}\n"
            f"群内词条：\n{_format_memories(snapshot.group_lexicon)}\n"
            f"对方消息：{context.plain_text}\n"
            f"参与决策：{decision.action}，原因：{decision.reason}\n"
            f"请直接给出要发送到群里的中文回复，最多 {self.config.bot.max_reply_chars} 个字。"
        )
        llm_reply = await self.llm.complete_text(system_prompt, user_prompt)
        if llm_reply:
            reply = _sanitize_reply(llm_reply, self.config.bot.max_reply_chars)
            guarded_reply = await self._guard_unapproved_self_claims(
                reply,
                context,
                snapshot,
                approved_self_memories,
            )
            return ReplyDraft(
                text=guarded_reply,
                self_memory_candidates=approved_self_memories,
            )

        if decision.action == "reply":
            if approved_self_memories:
                return ReplyDraft(
                    text=_fallback_reply_with_self_memory(approved_self_memories[0]),
                    self_memory_candidates=approved_self_memories,
                )
            return ReplyDraft(text="我在，刚才这句我先记下了。")
        return ReplyDraft()

    async def _guard_unapproved_self_claims(
        self,
        reply: str,
        context: MessageContext,
        snapshot: ConversationSnapshot,
        approved_self_memories: list[MemoryCandidate],
    ) -> str:
        unapproved = self.self_memory_ledger.extract_new_self_memories(
            reply,
            context,
            snapshot,
            approved_self_memories,
        )
        if not unapproved:
            return reply

        rewrite = await self.llm.complete_text(
            "你是 QQ 群回复改写器。只输出改写后的群聊回复，不要解释。",
            (
                "把下面回复改写得自然简短，去掉没有出现在“可引用自我记忆”中的自我经历。"
                "不要新增任何具体自我经历。\n"
                f"可引用自我记忆：\n{_format_memory_candidates(approved_self_memories)}\n"
                f"原回复：{reply}"
            ),
        )
        if rewrite:
            cleaned = _sanitize_reply(rewrite, self.config.bot.max_reply_chars)
            still_unapproved = self.self_memory_ledger.extract_new_self_memories(
                cleaned,
                context,
                snapshot,
                approved_self_memories,
            )
            if not still_unapproved:
                return cleaned

        return "这个我不拿自己的经历乱套，但感觉能懂一点。"


class SelfMemoryLedger:
    CLAIM_PATTERNS = (
        re.compile(r"(我(?:以前|之前|曾经|小时候|上次|最近)[^。！？\n]{2,40})"),
        re.compile(r"(我也[^。！？\n]{2,30}过)"),
    )

    def extract_new_self_memories(
        self,
        reply: str | None,
        context: MessageContext,
        snapshot: ConversationSnapshot,
        approved_memories: list[MemoryCandidate] | None = None,
    ) -> list[MemoryCandidate]:
        if not reply:
            return []
        existing = [record.content for record in snapshot.self_memories]
        approved = [item.content for item in approved_memories or []]
        candidates: list[MemoryCandidate] = []
        for pattern in self.CLAIM_PATTERNS:
            for match in pattern.finditer(reply):
                claim = " ".join(match.group(1).split())
                if len(claim) < 4:
                    continue
                if any(claim in memory or memory in claim for memory in [*existing, *approved]):
                    continue
                candidates.append(
                    MemoryCandidate(
                        owner_type="self",
                        owner_id="bot",
                        kind=_infer_self_narrative_kind(claim),
                        content=claim,
                        confidence=0.82,
                        importance=0.64,
                        evidence_message_id=context.message_id,
                        source_text=reply,
                        source_user_id="bot",
                        source_group_id=context.group_id,
                        subject_user_id="bot",
                        claim_scope="bot_directed",
                    )
                )
        return candidates


class ReflectionAgent:
    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    async def reflect(
        self,
        group_id: str,
        recent_messages: list[str],
        prior_reflections: list[MemoryRecord],
    ) -> MemoryCandidate | None:
        if not recent_messages:
            return None
        data = await _complete_json(
            self.llm,
            "你是 QQ 群聊阶段性复盘器。只输出 JSON，不要解释。",
            (
                "根据最近群聊生成一条长期群记忆。不要逐条复述，提炼话题、气氛和可记住的关系线索。"
                '输出 JSON：{"summary":"80字以内","topics":["话题"],"importance":0.0}\n'
                f"已有复盘：\n{_format_memories(prior_reflections)}\n"
                f"最近消息：\n{_join_lines(recent_messages)}"
            ),
        )
        if data:
            summary = str(data.get("summary", "")).strip()
            topics = _clean_list(data.get("topics"))
            importance = _clamp_float(data.get("importance", 0.7))
        else:
            summary = "；".join(recent_messages[-5:])[:80]
            topics = []
            importance = 0.55
        if not summary:
            return None
        content = summary if not topics else f"{summary}｜话题：{'、'.join(topics[:5])}"
        return MemoryCandidate(
            owner_type="group",
            owner_id=str(group_id),
            kind="reflection",
            content=content,
            confidence=0.82,
            importance=importance,
            evidence_message_id=f"reflection-{int(time.time())}",
            source_text="\n".join(recent_messages[-20:]),
            source_user_id="bot",
            source_group_id=str(group_id),
            subject_user_id=str(group_id),
            claim_scope="group_fact",
        )


class AgentPipeline:
    def __init__(
        self,
        config: AppConfig,
        llm: LLMClient,
        web_search: WebSearchClient | None = None,
        vision_cache: VisionCacheStore | None = None,
    ) -> None:
        self.perception = PerceptionAgent(llm)
        self.vision = VisionAgent(config, llm, vision_cache)
        self.memory_curator = MemoryCuratorAgent(llm)
        self.lexicon = LexiconAgent(config, llm, web_search)
        self.relationship = RelationshipAgent(llm)
        self.policy = ParticipationPolicyAgent(config, llm)
        self.self_narrative = SelfNarrativeAgent(llm)
        self.response = ResponseAgent(config, llm)
        self.reflection = ReflectionAgent(llm)

    async def run(
        self,
        context: MessageContext,
        mode: ParticipationMode,
        snapshot: ConversationSnapshot,
    ) -> PipelineResult:
        vision = await self.vision.analyze(context)
        enriched_context = _context_with_vision(context, vision)
        perception = await self.perception.analyze(enriched_context, snapshot)
        memories = await self.memory_curator.extract(enriched_context, perception, snapshot)
        lexicon_memories = await self.lexicon.learn(enriched_context, snapshot)
        relationship_delta = await self.relationship.calculate_delta(
            enriched_context,
            perception,
            snapshot,
        )
        decision = await self.policy.decide(enriched_context, perception, mode, snapshot)
        self_memories = await self.self_narrative.prepare(
            enriched_context,
            perception,
            decision,
            snapshot,
        )
        reply_draft = await self.response.generate(
            enriched_context,
            perception,
            decision,
            snapshot,
            self_memories,
        )
        return PipelineResult(
            perception=perception,
            memories=[*memories, *lexicon_memories, *vision.memory_candidates],
            relationship_delta=relationship_delta,
            decision=decision,
            reply=reply_draft.text,
            reply_self_memories=reply_draft.self_memory_candidates,
            image_descriptions=vision.descriptions,
        )

    async def reflect(
        self,
        group_id: str,
        recent_messages: list[str],
        prior_reflections: list[MemoryRecord],
    ) -> MemoryCandidate | None:
        return await self.reflection.reflect(group_id, recent_messages, prior_reflections)


async def _complete_json(llm: LLMClient, system_prompt: str, user_prompt: str) -> dict[str, Any] | None:
    text = await llm.complete_text(system_prompt, user_prompt)
    if not text:
        return None
    try:
        return _parse_json_object(text)
    except ValueError as exc:
        logger.warning("Structured LLM JSON parse failed: {}", exc)
        return None


async def _complete_vision_json(
    llm: LLMClient,
    config: AppConfig,
    system_prompt: str,
    user_prompt: str,
    image_urls: list[str],
) -> dict[str, Any] | None:
    text = await llm.complete_vision(system_prompt, user_prompt, image_urls, config.vision)
    if not text:
        return None
    try:
        return _parse_json_object(text)
    except ValueError as exc:
        logger.warning("Structured vision JSON parse failed: {}", exc)
        return None


def _parse_json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, re.S)
    if fenced:
        cleaned = fenced.group(1)
    elif not cleaned.startswith("{"):
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start >= 0 and end > start:
            cleaned = cleaned[start : end + 1]
    data = json.loads(cleaned)
    if not isinstance(data, dict):
        raise ValueError("JSON root is not an object")
    return data


def _owner_id_for(
    owner_type: str,
    context: MessageContext,
    claim_scope: str = "self_report",
    subject_user_id: str = "",
) -> str:
    if owner_type == "self":
        return "bot"
    if owner_type == "group":
        return context.group_id
    if claim_scope == "third_party" and subject_user_id:
        return subject_user_id
    return context.user_id


def _subject_for(owner_type: str, owner_id: str, context: MessageContext, claim_scope: str) -> str:
    if owner_type == "self":
        return "bot"
    if owner_type == "group":
        return context.group_id
    if claim_scope == "third_party":
        return owner_id
    return context.user_id


def _safe_claim_scope(value: str) -> str:
    return value if value in {"self_report", "third_party", "bot_directed", "group_fact"} else "self_report"


def _context_with_vision(context: MessageContext, vision: VisionAnalysis) -> MessageContext:
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


def _clean_image_text(value: str) -> str:
    text = " ".join(str(value).strip().split())
    return text[:300].strip()


def _looks_sensitive_image_memory(value: str) -> bool:
    return bool(
        re.search(
            r"(身份证|手机号|住址|家庭住址|银行卡|密码|真实姓名|人脸识别|长得像|某某本人)",
            value,
        )
    )


def _dedupe_strings(values: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        text = str(value).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        deduped.append(text)
    return deduped


def _safe_self_kind(value: str) -> str:
    normalized = value.strip()
    normalized = SELF_NARRATIVE_KIND_ALIASES.get(normalized, normalized)
    return normalized if normalized in SELF_NARRATIVE_KINDS else "self_habit"


def _safe_fictionality(value: str) -> str:
    normalized = value.strip()
    return normalized if normalized in SELF_FICTIONALITY_VALUES else "fictional_light"


def _clean_self_narrative_content(value: str) -> str:
    text = " ".join(str(value).strip().split())
    text = text.strip("「」“”\"'`")
    if not text:
        return ""
    if not text.startswith("我"):
        text = f"我{text}"
    return text[:60].strip()


def _self_memory_candidate(
    context: MessageContext,
    kind: str,
    content: str,
    confidence: float,
    importance: float,
    purpose: str,
    fictionality: str,
) -> MemoryCandidate:
    return MemoryCandidate(
        owner_type="self",
        owner_id="bot",
        kind=_safe_self_kind(kind),
        content=_clean_self_narrative_content(content),
        confidence=confidence,
        importance=importance,
        evidence_message_id=context.message_id,
        source_text=(
            f"fictionality={_safe_fictionality(fictionality)}\n"
            f"purpose={purpose}\n"
            f"trigger_user={context.user_id}\n"
            f"trigger_message={context.plain_text}"
        ),
        source_user_id="bot",
        source_group_id=context.group_id,
        subject_user_id="bot",
        claim_scope="bot_directed",
        verification_status="accepted",
    )


def _heuristic_self_narrative_status(
    candidate: MemoryCandidate,
    snapshot: ConversationSnapshot,
) -> str:
    if UNSAFE_SELF_PATTERN.search(candidate.content):
        return "too_specific"
    if any(boundary in candidate.content for boundary in ("我是真人", "我能线下", "我现实中")):
        return "unsafe"
    for memory in snapshot.self_memories:
        if candidate.content == memory.content:
            return "accepted"
        if candidate.kind in {"self_preference", "self_boundary"} and memory.kind == candidate.kind:
            if _looks_like_direct_self_conflict(candidate.content, memory.content):
                return "conflict"
    return "accepted"


def _looks_like_direct_self_conflict(new_content: str, old_content: str) -> bool:
    positive_tokens = ("喜欢", "想", "会", "习惯")
    negative_tokens = ("不喜欢", "讨厌", "怕", "不会", "不太")
    new_positive = any(token in new_content for token in positive_tokens)
    new_negative = any(token in new_content for token in negative_tokens)
    old_positive = any(token in old_content for token in positive_tokens)
    old_negative = any(token in old_content for token in negative_tokens)
    shared = _self_object_terms(new_content) & _self_object_terms(old_content)
    return bool(shared and ((new_positive and old_negative) or (new_negative and old_positive)))


def _self_object_terms(content: str) -> set[str]:
    cleaned = content
    for token in ("不喜欢", "喜欢", "讨厌", "害怕", "怕", "我", "很", "比较", "一点", "有点"):
        cleaned = cleaned.replace(token, "")
    terms: set[str] = set()
    for phrase in re.findall(r"[\u4e00-\u9fffA-Za-z0-9]{2,}", cleaned):
        terms.add(phrase)
        if len(phrase) <= 12:
            terms.update(phrase[index : index + 2] for index in range(len(phrase) - 1))
    return {term for term in terms if len(term) >= 2}


def _infer_self_narrative_kind(claim: str) -> str:
    if any(token in claim for token in ("喜欢", "讨厌", "怕", "偏爱")):
        return "self_preference"
    if any(token in claim for token in ("习惯", "平时", "总会")):
        return "self_habit"
    if any(token in claim for token in ("以前", "之前", "曾经", "小时候", "上次")):
        return "self_past_event"
    return "self_past_event"


def _format_memory_candidates(memories: list[MemoryCandidate]) -> str:
    if not memories:
        return "(none)"
    return "\n".join(f"[{item.kind}] {item.content}" for item in memories)


def _fallback_reply_with_self_memory(memory: MemoryCandidate) -> str:
    content = memory.content.strip()
    if content.startswith("我"):
        content = content[1:]
    if memory.kind in {"self_preference", "self_hobby"}:
        return f"嗯，我{content}。"
    if memory.kind == "self_past_event":
        return f"有点像，我{content}。"
    return f"嗯，我{content}，所以能懂一点。"


def _clean_lexicon_term(value: str) -> str:
    term = " ".join(str(value).strip().split())
    term = term.strip("「」“”\"'`.,，。!！?？:：;；()（）[]【】")
    return term if _looks_like_lexicon_term(term) else ""


def _looks_like_lexicon_term(term: str) -> bool:
    if not 2 <= len(term) <= 24:
        return False
    lowered = term.lower()
    stopwords = {
        "可可",
        "小祈",
        "这个",
        "那个",
        "什么",
        "什么意思",
        "啥意思",
        "怎么回事",
        "大家",
        "我们",
        "你们",
    }
    if lowered in stopwords or term in stopwords:
        return False
    if lowered.startswith(("http://", "https://", "www.", "#bot", "/bot")):
        return False
    if re.fullmatch(r"\d+", term):
        return False
    return not any(token in term for token in ("\n", "\r", "\t"))


def _strip_bot_call(text: str, nicknames: list[str]) -> str:
    cleaned = text.strip()
    for nickname in nicknames:
        if not nickname:
            continue
        cleaned = re.sub(
            rf"^\s*@?{re.escape(nickname)}[\s,，:：]*",
            "",
            cleaned,
            count=1,
        ).strip()
    return cleaned


def _dedupe_terms(candidates: list[LexiconTermCandidate]) -> list[LexiconTermCandidate]:
    seen: set[str] = set()
    deduped: list[LexiconTermCandidate] = []
    for candidate in candidates:
        term = _clean_lexicon_term(candidate.term)
        key = _normalize_lexicon_term(term)
        if not term or key in seen:
            continue
        seen.add(key)
        deduped.append(
            LexiconTermCandidate(
                term=term,
                reason=candidate.reason,
                search_query=candidate.search_query,
                confidence=candidate.confidence,
            )
        )
    return deduped


def _has_existing_lexicon(term: str, memories: list[MemoryRecord]) -> bool:
    normalized = _normalize_lexicon_term(term)
    subject = _lexicon_subject(term)
    for memory in memories:
        if memory.kind != "lexicon":
            continue
        if memory.subject_user_id == subject:
            return True
        content = _normalize_lexicon_term(memory.content)
        if content.startswith(f"「{normalized}」") or content.startswith(f"{normalized}:"):
            return True
    return False


def _normalize_lexicon_term(term: str) -> str:
    return " ".join(str(term).strip().lower().split())


def _lexicon_subject(term: str) -> str:
    return f"term:{_normalize_lexicon_term(term)}"


def _format_search_results(results: list[SearchResult]) -> str:
    lines = []
    for index, result in enumerate(results[:5], start=1):
        title = " ".join(result.title.split())[:120]
        url = result.url.strip()
        snippet = " ".join(result.snippet.split())[:240]
        lines.append(f"{index}. {title}\nURL: {url}\n摘要: {snippet or '(empty)'}")
    return "\n".join(lines) if lines else "(none)"


def _fallback_search_definition(results: list[SearchResult]) -> str:
    for result in results:
        snippet = _clean_lexicon_definition(result.snippet)
        if snippet:
            return snippet
    if results:
        return _clean_lexicon_definition(results[0].title)
    return ""


def _clean_lexicon_definition(value: str) -> str:
    text = " ".join(str(value).strip().split())
    text = re.sub(r"^释义[:：]\s*", "", text)
    return text[:100].strip()


def _format_memories(memories: list[MemoryRecord]) -> str:
    if not memories:
        return "(none)"
    return "\n".join(f"#{m.id} [{m.kind}] {m.content}" for m in memories)


def _format_relationship(snapshot: ConversationSnapshot) -> str:
    relation = snapshot.relationship
    if relation is None:
        return "(none)"
    return (
        f"closeness={relation.closeness}, trust={relation.trust}, "
        f"familiarity={relation.familiarity}, tension={relation.tension}, "
        f"summary={relation.summary or '(empty)'}"
    )


def _join_lines(lines: list[str]) -> str:
    return "\n".join(lines) if lines else "(none)"


def _as_bool(value: Any, fallback: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes"}
    if value is None:
        return fallback
    return bool(value)


def _as_float(value: Any, fallback: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def _clamp_float(value: Any) -> float:
    return max(0.0, min(1.0, _as_float(value, 0.0)))


def _clamp_delta(value: Any) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = 0
    return max(-3, min(3, parsed))


def _clean_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _safe_choice(value: str, allowed: set[str], fallback: str) -> str:
    return value if value in allowed else fallback


def _sanitize_reply(reply: str, max_chars: int) -> str:
    text = reply.strip()
    text = re.sub(r"^回复[:：]\s*", "", text)
    text = text.replace("作为AI", "").replace("作为一个AI", "")
    return text[:max_chars].strip()


def _extract_topics(text: str) -> list[str]:
    topics = []
    for keyword in ("游戏", "电影", "工作", "学校", "代码", "AI", "LLM", "吃", "旅行", "音乐"):
        if keyword.lower() in text.lower():
            topics.append(keyword)
    return topics[:5]


def _emotion_hint(text: str) -> str:
    if any(token in text for token in ("哈哈", "笑死", "开心", "舒服")):
        return "positive"
    if any(token in text for token in ("难受", "烦", "崩溃", "气死")):
        return "negative"
    return "neutral"
