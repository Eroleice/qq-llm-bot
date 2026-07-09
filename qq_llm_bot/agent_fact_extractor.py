from __future__ import annotations

import re
from typing import Any

from qq_llm_bot.agent_common import clamp_float as _clamp_float
from qq_llm_bot.agent_extractor_utils import _strip_bot_call
from qq_llm_bot.agent_fact_helpers import (
    clean_fact_text as _clean_fact_text,
    dedupe_fact_candidates as _dedupe_fact_candidates,
    extract_mention_claim as _extract_mention_claim,
    fact_candidate as _fact_candidate,
    heuristic_fact_topic as _heuristic_fact_topic,
    heuristic_stance as _heuristic_stance,
    looks_low_value_fact_text as _looks_low_value_fact_text,
    member_mentions as _member_mentions,
    mention_claim_text as _mention_claim_text,
    safe_claim_scope as _safe_claim_scope,
    safe_fact_type as _safe_fact_type,
    safe_stance as _safe_stance,
)
from qq_llm_bot.agent_formatters import (
    format_mentions as _format_mentions,
    format_recent_context as _format_recent_context,
)
from qq_llm_bot.llm import LLMClient
from qq_llm_bot.llm_json_helpers import complete_json as _complete_json
from qq_llm_bot.models import (
    ConversationSnapshot,
    FactCandidate,
    MessageContext,
    PerceptionResult,
)
from qq_llm_bot.onebot_messages import strip_forwarded_records, strip_quoted_messages


class FactExtractorAgent:
    SELF_PATTERNS = (
        ("preference", re.compile(r"我(?:很|比较|超|挺)?喜欢\s*([^，。,.!！?？]{1,50})")),
        ("dislike", re.compile(r"我(?:不喜欢|讨厌)\s*([^，。,.!！?？]{1,50})")),
        ("identity", re.compile(r"我是\s*([^，。,.!！?？]{1,50})")),
        ("opinion", re.compile(r"我(?:觉得|认为|感觉)\s*([^，。,.!！?？]{2,80})")),
    )

    THIRD_PARTY_PATTERN = re.compile(
        r"([^\s，。,.!！?？我大家群里]{1,16})(喜欢|不喜欢|讨厌|认为|觉得)\s*([^，。,.!！?？]{1,80})"
    )

    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    async def extract(
        self,
        context: MessageContext,
        perception: PerceptionResult,
        snapshot: ConversationSnapshot,
    ) -> list[FactCandidate]:
        fallback = self._heuristic(context)
        data = await _complete_json(
            self.llm,
            "你是保守的群聊 FACT 抽取器。只输出 JSON，不要解释。",
            (
                "从这条 QQ 群消息和必要上下文中抽取成员认知 FACT。"
                "FACT 必须是结论性、要素完整、证据文本明确的原子断言。"
                "只记录成员的观点、偏好、身份、稳定倾向或对事件/对象的评价。"
                "不要记录聊天动作、继续聊、分享、发送图片、空消息、一次性情绪或流水账。"
                "如果主语、对象/话题、结论、证据任一不明确，facts 返回空数组。"
                "最近上下文只用于理解当前消息里的指代和话题延续，不能从最近上下文本身抽取新 FACT。"
                "每条 FACT 的 evidence_text 必须是当前消息里的原文片段；如果证据只出现在最近上下文，返回空数组。"
                "本人发言里的自我观点用 self_report；别人转述某成员用 third_party。"
                "输出 JSON："
                '{"facts":[{"subject_user_id":"QQ或name:称呼","fact_type":"preference|dislike|'
                'opinion|identity|habit|skill|boundary|event_stance|other",'
                '"claim_text":"完整结论句","topic":"对象或事件","stance":"positive|negative|neutral|mixed|unknown",'
                '"confidence":0.0,"importance":0.0,"claim_scope":"self_report|third_party",'
                '"evidence_text":"原消息中的证据片段"}]}\n'
                "好例子：我觉得刮刮乐像负期望彩票 -> 用户认为刮刮乐活动像负期望彩票。"
                "坏例子：继续聊比赛感受、分享截图、哈哈、空消息 -> facts=[]。\n"
                f"说话人 QQ：{context.user_id}\n"
                "Mention rule: when this message says this person/he/she/ta/@someone and a QQ mention "
                "clearly identifies that member, use the mentioned QQ as subject_user_id.\n"
                "Forwarded record rule: text between [合并转发聊天记录开始] and "
                "[合并转发聊天记录结束] is quoted chat history. First-person words inside it "
                "refer to the sender label on that forwarded line, not the member who forwarded it.\n"
                "Quoted message rule: text between [被引用消息开始] and [被引用消息结束] "
                "is quoted context. First-person words inside it refer to the quoted sender label, "
                "not the current speaker.\n"
                f"Current mentions:\n{_format_mentions(context)}\n"
                f"最近上下文：\n{_format_recent_context(snapshot)}\n"
                f"当前消息：{context.plain_text}"
            ),
            purpose="fact_extract",
        )
        facts: list[FactCandidate] = []
        if data:
            for item in data.get("facts", []):
                if not isinstance(item, dict):
                    continue
                fact = self._parse_llm_fact(context, item)
                if fact:
                    facts.append(fact)
        return _dedupe_fact_candidates(facts) or fallback

    def _parse_llm_fact(
        self,
        context: MessageContext,
        item: dict[str, Any],
    ) -> FactCandidate | None:
        claim_scope = _safe_claim_scope(str(item.get("claim_scope", "self_report")).strip())
        subject_user_id = str(item.get("subject_user_id", "")).strip()
        if not subject_user_id and claim_scope == "self_report":
            subject_user_id = context.user_id
        claim_text = _clean_fact_text(str(item.get("claim_text", "")), 300)
        topic = _clean_fact_text(str(item.get("topic", "")), 120)
        evidence_text = _clean_fact_text(str(item.get("evidence_text", "")), 1000)
        if not subject_user_id or not claim_text or not topic or not evidence_text:
            return None
        if not _evidence_is_from_current_message(context, evidence_text):
            return None
        if _looks_low_value_fact_text(claim_text, topic, evidence_text):
            return None
        return FactCandidate(
            subject_user_id=subject_user_id,
            fact_type=_safe_fact_type(str(item.get("fact_type", "other")).strip()),
            claim_text=claim_text,
            topic=topic,
            stance=_safe_stance(str(item.get("stance", "unknown")).strip()),
            confidence=_clamp_float(item.get("confidence", 0.0)),
            evidence_message_id=context.message_id,
            evidence_text=evidence_text,
            source_user_id=context.user_id,
            source_group_id=context.group_id,
            claim_scope=claim_scope,  # type: ignore[arg-type]
            importance=_clamp_float(item.get("importance", 0.5)),
        )

    def _heuristic(self, context: MessageContext) -> list[FactCandidate]:
        text = _strip_bot_call(
            strip_quoted_messages(strip_forwarded_records(context.plain_text)),
            [],
        )
        if not text or _looks_low_value_fact_text(text, text, text):
            return []
        facts: list[FactCandidate] = []
        for mention in _member_mentions(context):
            extracted = _extract_mention_claim(text, mention)
            if extracted is None:
                continue
            kind, content = extracted
            fact_type = "identity" if kind == "alias" else kind
            if fact_type not in {"identity", "preference", "dislike", "opinion"}:
                fact_type = "other"
            facts.append(
                _fact_candidate(
                    context=context,
                    subject_user_id=mention.user_id,
                    fact_type=fact_type,
                    claim_text=_mention_claim_text(mention, kind, content),
                    topic=_heuristic_fact_topic(content),
                    stance=_heuristic_stance(content, fact_type),
                    confidence=0.78,
                    claim_scope="third_party",
                    evidence_text=text,
                )
            )

        for fact_type, pattern in self.SELF_PATTERNS:
            for match in pattern.finditer(text):
                value = match.group(1).strip()
                if not value:
                    continue
                topic = _heuristic_fact_topic(value)
                stance = _heuristic_stance(value, fact_type)
                if fact_type == "preference":
                    claim = f"用户{context.user_id}喜欢{value}"
                elif fact_type == "dislike":
                    claim = f"用户{context.user_id}不喜欢{value}"
                elif fact_type == "identity":
                    claim = f"用户{context.user_id}表示自己是{value}"
                else:
                    claim = f"用户{context.user_id}认为{value}"
                facts.append(
                    _fact_candidate(
                        context=context,
                        subject_user_id=context.user_id,
                        fact_type=fact_type,
                        claim_text=claim,
                        topic=topic,
                        stance=stance,
                        confidence=0.8,
                        claim_scope="self_report",
                        evidence_text=match.group(0),
                    )
                )

        for match in self.THIRD_PARTY_PATTERN.finditer(text):
            subject = match.group(1).strip()
            verb = match.group(2).strip()
            value = match.group(3).strip()
            if not subject or subject in {"可可", "机器人", "大家", "群里", "我们", "你"}:
                continue
            fact_type = "preference" if verb == "喜欢" else "dislike" if verb in {"不喜欢", "讨厌"} else "opinion"
            facts.append(
                _fact_candidate(
                    context=context,
                    subject_user_id=f"name:{subject}",
                    fact_type=fact_type,
                    claim_text=f"{subject}{verb}{value}",
                    topic=_heuristic_fact_topic(value),
                    stance=_heuristic_stance(value, fact_type),
                    confidence=0.78,
                    claim_scope="third_party",
                    evidence_text=match.group(0),
                )
            )
        return _dedupe_fact_candidates(facts)


def _evidence_is_from_current_message(context: MessageContext, evidence_text: str) -> bool:
    evidence = _canonical_evidence_text(evidence_text)
    source = _canonical_evidence_text(
        "\n".join(part for part in (context.plain_text, context.raw_message) if part)
    )
    if not evidence or not source:
        return False
    if evidence in source:
        return True
    return len(source) >= 8 and source in evidence


def _canonical_evidence_text(value: str) -> str:
    return re.sub(
        r"[\s，。,.!！?？：:；;\"'“”‘’（）()\[\]【】<>《》]+",
        "",
        str(value or "").strip().lower(),
    )
