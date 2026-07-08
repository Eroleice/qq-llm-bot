from __future__ import annotations

import re
from typing import Any

from qq_llm_bot.agent_fact_helpers import clean_fact_text as _clean_fact_text
from qq_llm_bot.agent_formatters import compact_target_context as _compact_target_context
from qq_llm_bot.final_qa import sanitize_reply as _sanitize_reply
from qq_llm_bot.models import (
    ConversationSnapshot,
    FactRecord,
    MessageContext,
    ParticipationDecision,
    ParticipationValueType,
    PerceptionResult,
    SemanticContext,
)


FOLLOWUP_CUE_PATTERN = re.compile(
    r"^(那|所以|然后|还有|这个|这呢|那我|那你|为啥|为什么|咋|怎么|能不能|"
    r"可以吗|要不要|是不是|所以呢|细说|展开|继续|刚才|前面|上面|你说)"
)


def _safe_participation_value_type(value: str) -> ParticipationValueType:
    allowed = {
        "none",
        "direct_reply",
        "answer",
        "synthesis",
        "missing_angle",
        "useful_context",
        "clarifying_question",
        "humor",
        "agreement",
        "empathy",
        "rephrase",
    }
    return value if value in allowed else "none"  # type: ignore[return-value]


def _proactive_value_type_allowed(value_type: str, traffic_level: str) -> bool:
    high_value = {"answer", "synthesis", "missing_angle", "useful_context", "clarifying_question"}
    if value_type in high_value:
        return True
    return traffic_level != "busy" and value_type == "humor"


def _needs_context_understanding(snapshot: ConversationSnapshot) -> bool:
    return bool(
        snapshot.recent_messages
        or snapshot.speaker_recent_messages
        or snapshot.other_recent_messages
        or snapshot.target_users
        or snapshot.unknown_name_refs
        or snapshot.ambiguous_name_refs
        or snapshot.user_facts
        or snapshot.user_profile
        or snapshot.group_reflections
        or snapshot.group_lexicon
        or snapshot.recent_bot_reply_to_user
    )


def _fallback_semantic_context(
    context: MessageContext,
    snapshot: ConversationSnapshot,
) -> SemanticContext:
    relevant = snapshot.speaker_recent_messages[-6:] or snapshot.recent_messages[-6:]
    member_context = [_compact_target_context(target) for target in snapshot.target_users[:4]]
    uncertain = list(snapshot.unknown_name_refs)
    uncertain.extend(
        f"{name}: {', '.join(user_ids)}"
        for name, user_ids in snapshot.ambiguous_name_refs.items()
    )
    references = [f"当前消息中的“我”默认指 QQ:{context.user_id}"]
    if context.is_direct or context.bot_mentioned:
        references.append("当前消息中直接称呼机器人时，“你”默认指机器人")
    elif re.search(r"[你他她它]们?|ta|TA", context.plain_text):
        references.append("“你/他/她/ta”等指代需要结合最近消息，未显式确认时不要当事实")
    return SemanticContext(
        current_intent=_clean_fact_text(context.plain_text, 120),
        relevant_messages=relevant,
        resolved_references=references,
        member_context=[item for item in member_context if item],
        uncertain_references=_clean_string_items(uncertain, limit=8, item_limit=120),
    )


def _semantic_context_from_json(data: dict[str, Any] | None) -> SemanticContext:
    if not data:
        return SemanticContext()
    return SemanticContext(
        current_intent=_clean_fact_text(str(data.get("current_intent", "")), 160),
        relevant_messages=_clean_string_items(data.get("relevant_messages"), limit=8, item_limit=180),
        resolved_references=_clean_string_items(data.get("resolved_references"), limit=10, item_limit=180),
        member_context=_clean_string_items(data.get("member_context"), limit=8, item_limit=200),
        uncertain_references=_clean_string_items(data.get("uncertain_references"), limit=8, item_limit=180),
        ignored_noise=_clean_string_items(data.get("ignored_noise"), limit=8, item_limit=120),
    )


def _clean_string_items(value: Any, *, limit: int, item_limit: int) -> list[str]:
    if isinstance(value, tuple):
        value = list(value)
    if not isinstance(value, list):
        return []
    items: list[str] = []
    for item in value[:limit]:
        text = _clean_fact_text(str(item), item_limit)
        if text:
            items.append(text)
    return items


def _has_unresolved_identity_target(snapshot: ConversationSnapshot) -> bool:
    return bool(snapshot.unknown_name_refs or snapshot.ambiguous_name_refs)


def _target_confirmation_reply(snapshot: ConversationSnapshot) -> str:
    if snapshot.ambiguous_name_refs:
        name, user_ids = next(iter(snapshot.ambiguous_name_refs.items()))
        choices = "、".join(f"QQ:{user_id}" for user_id in user_ids[:5])
        return f"我不太确定“{name}”指哪位，是 {choices} 里的谁？"
    if snapshot.unknown_name_refs:
        name = snapshot.unknown_name_refs[0]
        return f"我还不确定“{name}”对应哪位，可以告诉我 QQ 号吗？"
    return ""


def _target_fact_fallback_reply(context: MessageContext, snapshot: ConversationSnapshot) -> str:
    if not snapshot.target_users or not _looks_like_identity_query(context.plain_text):
        return ""
    target = snapshot.target_users[0]
    alias = _best_target_alias(context.plain_text, target.aliases, target.facts)
    if re.search(r"(怎么|如何|要怎么|该怎么).{0,6}称呼|叫什么|叫啥|名字|昵称|外号|称呼", context.plain_text):
        if alias:
            return f"我记得 QQ:{target.user_id} 可以叫“{alias}”。"
        return f"我只确认到是 QQ:{target.user_id}，还没记到稳定称呼。"
    if alias:
        return f"{alias}是 QQ:{target.user_id}。"
    identity_fact = _first_identity_fact_text(target.facts)
    if identity_fact:
        return identity_fact
    return f"我能确认指向 QQ:{target.user_id}。"


def _best_target_alias(text: str, aliases: list[str], facts: list[FactRecord]) -> str:
    for alias in aliases:
        if alias and alias in text:
            return alias
    if aliases:
        return aliases[0]
    for fact in facts:
        alias = _alias_from_fact_text(fact.claim_text, fact.evidence_text)
        if alias:
            return alias
    return ""


def _alias_from_fact_text(*texts: str) -> str:
    combined = "\n".join(str(text or "") for text in texts)
    patterns = (
        r"(?:称呼|昵称|外号|名字)\s*(?:是|叫|为|：|:)?\s*[“\"']?([^，,。；;、\s”\"']{1,30})",
        r"(?:叫做|称作|称为|叫)\s*[“\"']?([^，,。；;、\s”\"']{1,30})",
    )
    for pattern in patterns:
        match = re.search(pattern, combined)
        if match:
            return match.group(1).strip("“”\"'`")
    return ""


def _first_identity_fact_text(facts: list[FactRecord]) -> str:
    for fact in facts:
        if fact.fact_type in {"identity", "alias"}:
            return fact.claim_text
    return ""


def _looks_like_identity_query(text: str) -> bool:
    return bool(
        re.search(
            r"(谁是|是谁|叫啥|叫什么|叫什麼|怎么称呼|如何称呼|要怎么称呼|该怎么称呼|名字|昵称|外号|称呼)",
            text,
        )
    )


def _looks_like_uncertain_reply(reply: str) -> bool:
    compact = re.sub(r"\s+", "", reply)
    return bool(re.search(r"(不知道|不清楚|没印象|没有记到|没查到|不太确定|不确定|无法确认)", compact))


def _looks_like_recent_interaction_followup(
    text: str,
    perception: PerceptionResult,
) -> bool:
    compact = re.sub(r"\s+", "", text.strip())
    if len(compact) < 2 or len(compact) > 80:
        return False
    if FOLLOWUP_CUE_PATTERN.search(compact):
        return True
    if perception.is_question and re.search(r"(那|这个|这样|所以|然后|还|再|刚才|前面|上面|你说)", compact):
        return True
    return False


def _clamp_delta(value: Any) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = 0
    return max(-3, min(3, parsed))


def _reply_has_incremental_value(reply: str | None, decision: ParticipationDecision) -> bool:
    if decision.action != "proactive_reply":
        return True
    text = _sanitize_reply(reply or "", 500)
    if not text:
        return False
    if not _proactive_value_type_allowed(decision.value_type, decision.traffic_level):
        return False
    if _looks_like_low_value_proactive_reply(text):
        return False
    if decision.value_type == "clarifying_question":
        return any(mark in text for mark in ("?", "？", "怎么", "为什么", "要不要", "是不是", "能不能"))
    if decision.value_type == "humor":
        return len(text) >= 6 and decision.traffic_level != "busy"
    return len(text) >= 8


def _looks_like_low_value_proactive_reply(text: str) -> bool:
    compact = re.sub(r"[\s，。,.!！?？~～…、]+", "", text)
    if not compact:
        return True
    exact_low_value = {
        "确实",
        "是的",
        "对",
        "对啊",
        "有道理",
        "我也觉得",
        "我同意",
        "同意",
        "赞同",
        "挺有意思",
        "好像是这样",
        "哈哈",
        "哈哈哈",
        "笑死",
        "你们聊得好热闹",
    }
    if compact in exact_low_value:
        return True
    low_value_prefixes = ("确实", "我也觉得", "有道理", "挺有意思", "同意", "赞同", "哈哈")
    if any(compact.startswith(prefix) for prefix in low_value_prefixes) and len(compact) < 18:
        return True
    if "你们聊得" in compact and len(compact) < 22:
        return True
    return False
