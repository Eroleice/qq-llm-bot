from __future__ import annotations

import re
import time

from loguru import logger

from qq_llm_bot.agent_common import as_bool as _as_bool, clamp_float as _clamp_float
from qq_llm_bot.agent_formatters import (
    clean_lexicon_definition as _clean_lexicon_definition,
    fallback_search_definition as _fallback_search_definition,
    format_memories as _format_memories,
    format_search_results as _format_search_results,
)
from qq_llm_bot.agent_models import LexiconTermCandidate
from qq_llm_bot.config import AppConfig
from qq_llm_bot.llm import LLMClient
from qq_llm_bot.llm_json_helpers import complete_json as _complete_json
from qq_llm_bot.models import ConversationSnapshot, MemoryCandidate, MemoryRecord, MessageContext
from qq_llm_bot.text_utils import (
    lexicon_subject as _lexicon_subject,
    normalize_lexicon_term as _normalize_lexicon_term,
)
from qq_llm_bot.web_search import SearchResult, WebSearchClient, build_web_search_client, default_slang_query


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
            purpose="lexicon_detect",
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
            purpose="lexicon_summarize",
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
