from __future__ import annotations

import re

from qq_llm_bot.agent_common import (
    as_bool as _as_bool,
    clamp_float as _clamp_float,
    clean_list as _clean_list,
)
from qq_llm_bot.agent_extractor_utils import (
    _emotion_hint,
    _extract_topics,
)
from qq_llm_bot.agent_fact_helpers import context_mentions_bot as _context_mentions_bot
from qq_llm_bot.agent_formatters import format_recent_context as _format_recent_context
from qq_llm_bot.llm import LLMClient
from qq_llm_bot.llm_json_helpers import complete_json as _complete_json
from qq_llm_bot.models import ConversationSnapshot, MessageContext, PerceptionResult
from qq_llm_bot.text_utils import safe_choice as _safe_choice


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
                f"最近上下文：\n{_format_recent_context(snapshot)}\n"
                f"消息：{context.plain_text}"
            ),
            purpose="perception",
        )
        if not data:
            return fallback
        return PerceptionResult(
            is_question=_as_bool(data.get("is_question"), fallback.is_question),
            is_self_disclosure=_as_bool(data.get("is_self_disclosure"), fallback.is_self_disclosure),
            mentions_bot=_context_mentions_bot(context),
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
            mentions_bot=_context_mentions_bot(context),
            topics=_extract_topics(text),
            emotion_hint=_emotion_hint(text),
            confidence=0.55,
        )
