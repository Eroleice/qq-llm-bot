from __future__ import annotations

import time
from typing import Any

from qq_llm_bot.agent_fact_helpers import (
    clean_fact_text as _clean_fact_text,
    dedupe_fact_candidates as _dedupe_fact_candidates,
    fact_candidate as _fact_candidate,
    looks_low_value_fact_text as _looks_low_value_fact_text,
    owner_id_for as _owner_id_for,
    safe_claim_scope as _safe_claim_scope,
    subject_for as _subject_for,
)
from qq_llm_bot.agent_formatters import (
    format_fact_records as _format_fact_records,
    format_memories as _format_memories,
    format_user_profile_record as _format_user_profile_record,
    join_lines as _join_lines,
)
from qq_llm_bot.agent_models import BatchObservationResult
from qq_llm_bot.config import AppConfig
from qq_llm_bot.llm import LLMClient
from qq_llm_bot.llm_json_helpers import complete_json as _complete_json
from qq_llm_bot.models import (
    FactCandidate,
    FactRecord,
    MemoryCandidate,
    MemoryRecord,
    MessageContext,
    UserProfileDraft,
    UserProfileRecord,
)


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
            purpose="reflection",
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


class ProfileAggregatorAgent:
    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    async def aggregate(
        self,
        user_id: str,
        facts: list[FactRecord],
        current_profile: UserProfileRecord | None = None,
    ) -> UserProfileDraft | None:
        if not facts:
            return None
        data = await _complete_json(
            self.llm,
            "你是群成员画像分析器。只输出 JSON，不要解释。",
            (
                "根据 accepted FACT 更新该 QQ 用户的全局画像。"
                "画像必须从 FACT 归纳，不要编造 FACT 没有支持的内容。"
                "summary 用 1-3 句中文概括稳定特征、偏好、观点倾向或互动风格。"
                "traits 是对象，可包含 preferences、opinions、communication_style、interests、boundaries 等数组。"
                "supporting_fact_ids 只填实际支撑画像的 FACT id。"
                "输出 JSON："
                '{"summary":"画像摘要","traits":{"preferences":[],"opinions":[]},'
                '"supporting_fact_ids":[1,2]}\n'
                f"QQ：{user_id}\n"
                f"当前画像：\n{_format_user_profile_record(current_profile)}\n"
                f"FACT：\n{_format_fact_records(facts)}"
            ),
            purpose="profile_aggregate",
        )
        if not data:
            return None
        summary = _clean_fact_text(str(data.get("summary", "")), 500)
        traits = _clean_traits(data.get("traits"))
        fallback_ids = tuple(fact.id for fact in facts[:20])
        supporting_ids = _parse_fact_ids(data.get("supporting_fact_ids"), fallback_ids)
        if not summary:
            return None
        return UserProfileDraft(
            summary=summary,
            traits=traits,
            supporting_fact_ids=supporting_ids,
        )


class BatchObservationAgent:
    def __init__(self, config: AppConfig, llm: LLMClient) -> None:
        self.config = config
        self.llm = llm

    async def summarize(
        self,
        group_id: str,
        contexts: list[MessageContext],
        prior_reflections: list[MemoryRecord],
        group_lexicon: list[MemoryRecord],
    ) -> BatchObservationResult:
        clean_contexts = [context for context in contexts if context.plain_text or context.attachments]
        if not clean_contexts:
            return BatchObservationResult()

        by_message_id = {context.message_id: context for context in clean_contexts}
        data = await _complete_json(
            self.llm,
            "你是 QQ 群聊观察批处理器。只输出 JSON，不要解释。",
            (
                "请批量整理这些群消息，只保留稳定、明确、之后有用的信息。"
                "不要逐条复述，不要记录寒暄、表情、哈哈、流水账、一次性情绪或普通聊天动作。"
                "成员 FACT 只记录观点、偏好、身份、习惯、技能、边界或对对象/事件的稳定评价。"
                "记忆只记录群内长期有用的词条、群事实或很明确的成员自述。"
                "如果某条信息主体、话题、结论或证据不明确，就不要抽取。"
                "reflection 用 80 字以内概括这批消息的主线；如果没有值得长期记住的主线，summary 置空。"
                "输出 JSON："
                '{"memories":[{"message_id":"原消息id","owner_type":"user|group|self",'
                '"owner_id":"可空","subject_user_id":"QQ或group或bot",'
                '"claim_scope":"self_report|third_party|bot_directed|group_fact",'
                '"kind":"alias|identity|preference|dislike|location|experience|persona_fact|lexicon|group_fact",'
                '"content":"短句","confidence":0.0,"importance":0.0}],'
                '"facts":[{"message_id":"原消息id","subject_user_id":"QQ或name:称呼",'
                '"fact_type":"preference|dislike|opinion|identity|habit|skill|boundary|event_stance|other",'
                '"claim_text":"完整结论句","topic":"对象或事件",'
                '"stance":"positive|negative|neutral|mixed|unknown","confidence":0.0,'
                '"importance":0.0,"claim_scope":"self_report|third_party",'
                '"evidence_text":"原消息证据片段"}],'
                '"reflection":{"summary":"可空","topics":["话题"],"importance":0.0}}\n'
                f"群号：{group_id}\n"
                f"已有群复盘：\n{_format_memories(prior_reflections)}\n"
                f"已有群内词条：\n{_format_memories(group_lexicon)}\n"
                f"本批消息：\n{self._format_messages(clean_contexts)}"
            ),
            purpose="batch_observation",
        )
        if not data:
            return BatchObservationResult()

        memories = self._parse_memories(data.get("memories"), by_message_id)
        facts = self._parse_facts(data.get("facts"), by_message_id)
        reflection = self._parse_reflection(group_id, clean_contexts, data.get("reflection"))
        return BatchObservationResult(
            memories=memories,
            facts=_dedupe_fact_candidates(facts),
            reflection=reflection,
        )

    def _format_messages(self, contexts: list[MessageContext]) -> str:
        lines = []
        max_chars = self.config.observation_batch.max_message_chars
        for context in contexts[: self.config.observation_batch.max_messages_per_batch]:
            name = context.sender_name or context.sender_nickname or "-"
            text = " ".join(context.plain_text.split())
            if len(text) > max_chars:
                text = text[:max_chars].rstrip() + "..."
            if not text and context.attachments:
                text = f"[图片 x{len(context.attachments)}]"
            elif context.attachments:
                text = f"{text} [图片 x{len(context.attachments)}]"
            lines.append(
                f"- message_id={context.message_id} user_id={context.user_id} "
                f"name={name} text={text or '(empty)'}"
            )
        return "\n".join(lines) if lines else "(none)"

    def _parse_memories(
        self,
        value: Any,
        by_message_id: dict[str, MessageContext],
    ) -> list[MemoryCandidate]:
        if not isinstance(value, list):
            return []
        memories: list[MemoryCandidate] = []
        for item in value:
            if not isinstance(item, dict):
                continue
            context = self._context_for_item(item, by_message_id)
            if context is None:
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
            content = _clean_fact_text(str(item.get("content", "")), 300)
            if not content:
                continue
            memories.append(
                MemoryCandidate(
                    owner_type=owner_type,  # type: ignore[arg-type]
                    owner_id=owner_id,
                    kind=str(item.get("kind", "experience")).strip()[:40] or "experience",
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
        return memories

    def _parse_facts(
        self,
        value: Any,
        by_message_id: dict[str, MessageContext],
    ) -> list[FactCandidate]:
        if not isinstance(value, list):
            return []
        facts: list[FactCandidate] = []
        for item in value:
            if not isinstance(item, dict):
                continue
            context = self._context_for_item(item, by_message_id)
            if context is None:
                continue
            claim_scope = _safe_claim_scope(str(item.get("claim_scope", "self_report")).strip())
            subject_user_id = str(item.get("subject_user_id", "")).strip()
            if not subject_user_id and claim_scope == "self_report":
                subject_user_id = context.user_id
            claim_text = _clean_fact_text(str(item.get("claim_text", "")), 300)
            topic = _clean_fact_text(str(item.get("topic", "")), 120)
            evidence_text = _clean_fact_text(
                str(item.get("evidence_text", "") or context.plain_text),
                1000,
            )
            if not subject_user_id or not claim_text or not topic or not evidence_text:
                continue
            if _looks_low_value_fact_text(claim_text, topic, evidence_text):
                continue
            facts.append(
                _fact_candidate(
                    context=context,
                    subject_user_id=subject_user_id,
                    fact_type=str(item.get("fact_type", "other")).strip(),
                    claim_text=claim_text,
                    topic=topic,
                    stance=str(item.get("stance", "unknown")).strip(),
                    confidence=_clamp_float(item.get("confidence", 0.0)),
                    claim_scope=claim_scope,
                    evidence_text=evidence_text,
                    importance=_clamp_float(item.get("importance", 0.5)),
                )
            )
        return facts

    def _parse_reflection(
        self,
        group_id: str,
        contexts: list[MessageContext],
        value: Any,
    ) -> MemoryCandidate | None:
        if not isinstance(value, dict):
            return None
        summary = _clean_fact_text(str(value.get("summary", "")), 120)
        if not summary:
            return None
        topics = _clean_list(value.get("topics"))[:5]
        content = summary if not topics else f"{summary}；话题：{'、'.join(topics)}"
        first = contexts[0]
        last = contexts[-1]
        return MemoryCandidate(
            owner_type="group",
            owner_id=str(group_id),
            kind="reflection",
            content=content,
            confidence=0.82,
            importance=_clamp_float(value.get("importance", 0.62)),
            evidence_message_id=f"batch-{first.message_id}-{last.message_id}",
            source_text="\n".join(context.plain_text for context in contexts[-20:]),
            source_user_id="bot",
            source_group_id=str(group_id),
            subject_user_id=str(group_id),
            claim_scope="group_fact",
        )

    def _context_for_item(
        self,
        item: dict[str, Any],
        by_message_id: dict[str, MessageContext],
    ) -> MessageContext | None:
        message_id = str(item.get("message_id", "")).strip()
        if message_id:
            return by_message_id.get(message_id)
        if len(by_message_id) == 1:
            return next(iter(by_message_id.values()))
        return None



def _parse_fact_ids(value: Any, fallback: tuple[int, ...]) -> tuple[int, ...]:
    if not isinstance(value, list):
        return fallback
    ids: list[int] = []
    seen: set[int] = set()
    for item in value:
        try:
            parsed = int(item)
        except (TypeError, ValueError):
            continue
        if parsed in seen:
            continue
        ids.append(parsed)
        seen.add(parsed)
        if len(ids) >= 20:
            break
    return tuple(ids) or fallback



def _clean_traits(value: Any) -> dict[str, object]:
    if not isinstance(value, dict):
        return {}
    cleaned: dict[str, object] = {}
    for key, raw in value.items():
        name = _clean_fact_text(str(key), 40)
        if not name:
            continue
        if isinstance(raw, list):
            cleaned[name] = [_clean_fact_text(str(item), 80) for item in raw if _clean_fact_text(str(item), 80)][:12]
        elif isinstance(raw, (str, int, float, bool)):
            cleaned[name] = _clean_fact_text(str(raw), 120)
    return cleaned



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
