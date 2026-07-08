from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

from qq_llm_bot.directness import text_mentions_bot_name
from qq_llm_bot.models import FactCandidate, MessageContext, MessageMention
from qq_llm_bot.onebot_messages import format_mention_label


def context_mentions_bot(context: MessageContext, nicknames: Iterable[str] = ()) -> bool:
    return bool(
        context.is_direct
        or getattr(context, "bot_mentioned", False)
        or (nicknames and text_mentions_bot_name(context.plain_text, nicknames))
    )


def member_mentions(context: MessageContext) -> list[MessageMention]:
    return [
        mention
        for mention in context.mentions
        if mention.user_id.isdecimal() and not mention.is_bot
    ]


def extract_mention_claim(text: str, mention: MessageMention) -> tuple[str, str] | None:
    for label in _mention_text_variants(mention):
        index = text.find(label)
        if index < 0:
            continue
        tail = text[index + len(label) :].strip()
        parsed = _parse_mention_tail(tail)
        if parsed is not None:
            return parsed
    return None


def mention_claim_text(mention: MessageMention, kind: str, content: str) -> str:
    if kind == "alias":
        return f"用户{mention.user_id}叫{content}"
    if kind == "identity":
        return f"用户{mention.user_id}是{content}"
    if kind == "preference":
        return f"用户{mention.user_id}喜欢{content}"
    if kind == "dislike":
        return f"用户{mention.user_id}不喜欢{content}"
    if kind == "opinion":
        return f"用户{mention.user_id}认为{content}"
    return f"用户{mention.user_id}：{content}"


def owner_id_for(
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


def subject_for(owner_type: str, owner_id: str, context: MessageContext, claim_scope: str) -> str:
    if owner_type == "self":
        return "bot"
    if owner_type == "group":
        return context.group_id
    if claim_scope == "third_party":
        return owner_id
    return context.user_id


def safe_claim_scope(value: str) -> str:
    return value if value in {"self_report", "third_party", "bot_directed", "group_fact"} else "self_report"


def safe_fact_type(value: str) -> str:
    allowed = {
        "preference",
        "dislike",
        "opinion",
        "identity",
        "habit",
        "skill",
        "boundary",
        "event_stance",
        "other",
    }
    return value if value in allowed else "other"


def safe_stance(value: str) -> str:
    return value if value in {"positive", "negative", "neutral", "mixed", "unknown"} else "unknown"


def clean_fact_text(value: str, limit: int) -> str:
    text = " ".join(str(value or "").strip().split())
    return text[:limit].strip()


def looks_low_value_fact_text(claim_text: str, topic: str, evidence_text: str) -> bool:
    combined = f"{claim_text} {topic} {evidence_text}".strip()
    if not combined or combined.startswith("[图片解读]") or combined.startswith("[图片文字]"):
        return True
    if len(clean_fact_text(claim_text, 300)) < 4:
        return True
    low_value = (
        "继续聊",
        "随口",
        "发了",
        "发送",
        "分享图片",
        "分享截图",
        "空消息",
        "表情包",
        "接梗",
        "聊天",
        "参与讨论",
        "表达情绪",
    )
    signals = ("认为", "觉得", "喜欢", "不喜欢", "讨厌", "支持", "反对", "评价", "表示自己")
    return any(token in combined for token in low_value) and not any(token in combined for token in signals)


def heuristic_fact_topic(value: str) -> str:
    text = clean_fact_text(value, 120)
    if not text:
        return ""
    for marker in ("像", "是", "不", "很", "太", "应该", "可以", "不能", "好", "差", "离谱"):
        index = text.find(marker)
        if index > 1:
            return text[:index].strip()
    return text[:30].strip()


def heuristic_stance(value: str, fact_type: str) -> str:
    if fact_type == "preference":
        return "positive"
    if fact_type == "dislike":
        return "negative"
    if any(token in value for token in ("不", "差", "烂", "离谱", "讨厌", "恶心", "亏", "负面")):
        return "negative"
    if any(token in value for token in ("好", "喜欢", "支持", "可以", "舒服", "赞")):
        return "positive"
    return "neutral"


def fact_candidate(
    *,
    context: MessageContext,
    subject_user_id: str,
    fact_type: str,
    claim_text: str,
    topic: str,
    stance: str,
    confidence: float,
    claim_scope: str,
    evidence_text: str,
    importance: float = 0.5,
) -> FactCandidate:
    return FactCandidate(
        subject_user_id=subject_user_id,
        fact_type=safe_fact_type(fact_type),
        claim_text=clean_fact_text(claim_text, 300),
        topic=clean_fact_text(topic, 120),
        stance=safe_stance(stance),
        confidence=_clamp_float(confidence),
        evidence_message_id=context.message_id,
        evidence_text=clean_fact_text(evidence_text, 1000),
        source_user_id=context.user_id,
        source_group_id=context.group_id,
        claim_scope=safe_claim_scope(claim_scope),  # type: ignore[arg-type]
        importance=_clamp_float(importance),
    )


def dedupe_fact_candidates(facts: list[FactCandidate]) -> list[FactCandidate]:
    seen: set[tuple[str, str, str]] = set()
    deduped: list[FactCandidate] = []
    for fact in facts:
        if looks_low_value_fact_text(fact.claim_text, fact.topic, fact.evidence_text):
            continue
        key = (fact.subject_user_id, fact.fact_type, fact.claim_text)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(fact)
    return deduped


def _mention_text_variants(mention: MessageMention) -> list[str]:
    variants = [format_mention_label(mention)]
    display_name = " ".join(mention.display_name.split())
    if display_name:
        variants.append(f"@{display_name}")
    if mention.user_id:
        variants.append(f"@QQ:{mention.user_id}")
    deduped: list[str] = []
    for value in variants:
        if value and value not in deduped:
            deduped.append(value)
    return deduped


def _parse_mention_tail(tail: str) -> tuple[str, str] | None:
    tail = re.sub(r"^[\s，。,.!！?？：:]+", "", tail)
    patterns = (
        ("alias", re.compile(r"^(?:名字)?叫\s*([^，。,.!！?？\n]{1,50})")),
        ("identity", re.compile(r"^(?:是|就是)\s*([^，。,.!！?？\n]{1,50})")),
        ("dislike", re.compile(r"^(?:不喜欢|讨厌)\s*([^，。,.!！?？\n]{1,50})")),
        ("preference", re.compile(r"^(?:喜欢|爱|偏好)\s*([^，。,.!！?？\n]{1,50})")),
        ("opinion", re.compile(r"^(?:觉得|认为|感觉)\s*([^，。,.!！?？\n]{2,80})")),
    )
    for kind, pattern in patterns:
        match = pattern.search(tail)
        if not match:
            continue
        content = _clean_mention_claim_content(match.group(1))
        if content:
            return kind, content
    return None


def _clean_mention_claim_content(value: str) -> str:
    text = " ".join(str(value or "").strip().split())
    return re.sub(r"[啊呀哦呢吧啦喔]+$", "", text).strip()


def _clamp_float(value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = 0.0
    return max(0.0, min(1.0, parsed))
