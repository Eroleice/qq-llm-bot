from __future__ import annotations

import re
from typing import Any

from qq_llm_bot.final_qa import UNSAFE_SELF_PATTERN
from qq_llm_bot.models import (
    ConversationSnapshot,
    MemoryCandidate,
    MessageContext,
    ParticipationDecision,
    PerceptionResult,
)

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
TECHNICAL_BACKGROUND_PATTERN = re.compile(
    r"(UE5|Unreal|虚幻|Unity|Godot|Blender|C\+\+|Python|JavaScript|TypeScript|"
    r"React|Vue|Docker|Kubernetes|Linux|Git|SQL|API|LLM|AI|模型|提示词|"
    r"游戏引擎|蓝图|材质|渲染|Nanite|Lumen|代码|编程|程序|数据库|部署|服务器|"
    r"算法|插件|报错|性能|优化|配置|开发|框架)",
    re.I,
)
BACKGROUND_ADVICE_PATTERN = re.compile(
    r"(怎么|如何|建议|用|使用|学习|教程|入门|做|实现|调|优化|配置|排查|报错|"
    r"选|推荐|方案|经验|踩坑|注意什么|有没有必要)"
)

BACKGROUND_KIND_SET = {
    "self_background",
    "self_hobby",
    "self_habit",
    "self_past_event",
    "self_preference",
}
BACKGROUND_KEY_TERMS = (
    "UE5",
    "Unreal",
    "虚幻",
    "Unity",
    "Godot",
    "Blender",
    "C++",
    "Python",
    "JavaScript",
    "TypeScript",
    "React",
    "Vue",
    "Docker",
    "Kubernetes",
    "Linux",
    "Git",
    "SQL",
    "API",
    "LLM",
    "AI",
    "模型",
    "提示词",
    "游戏引擎",
    "蓝图",
    "材质",
    "渲染",
    "Nanite",
    "Lumen",
    "代码",
    "编程",
    "程序",
    "数据库",
    "部署",
    "服务器",
    "算法",
    "插件",
    "性能",
    "优化",
    "配置",
    "开发",
    "框架",
)


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


def _needs_self_background_for_topic(
    context: MessageContext,
    perception: PerceptionResult,
    decision: ParticipationDecision,
) -> bool:
    if decision.action not in {"reply", "proactive_reply"}:
        return False
    if decision.action == "reply" and not context.is_direct:
        return False

    scene = _background_scene_text(context, perception)
    if not TECHNICAL_BACKGROUND_PATTERN.search(scene):
        return False

    if decision.action == "proactive_reply":
        return decision.value_type in {
            "answer",
            "synthesis",
            "missing_angle",
            "useful_context",
            "clarifying_question",
        }
    return bool(BACKGROUND_ADVICE_PATTERN.search(scene))


def _has_relevant_self_background(
    context: MessageContext,
    perception: PerceptionResult,
    snapshot: ConversationSnapshot,
) -> bool:
    terms = _background_terms(context, perception)
    if not terms:
        return False
    texts = [
        memory.content
        for memory in snapshot.self_memories
        if memory.kind in BACKGROUND_KIND_SET and memory.status == "active"
    ]
    texts.extend(line for line in snapshot.persona_lines if "self_memory" in line or "background" in line)
    return any(_text_mentions_background_term(text, terms) for text in texts)


def _background_scene_text(context: MessageContext, perception: PerceptionResult) -> str:
    return " ".join([context.plain_text, *perception.topics])


def _background_terms(context: MessageContext, perception: PerceptionResult) -> set[str]:
    scene = _background_scene_text(context, perception)
    terms: set[str] = set()
    for term in BACKGROUND_KEY_TERMS:
        if re.search(re.escape(term), scene, re.I):
            terms.add(term)
    for term in re.findall(r"\b[A-Za-z][A-Za-z0-9+#.\-]{1,}\b", scene):
        if len(term) >= 2:
            terms.add(term)
    return terms


def _text_mentions_background_term(text: str, terms: set[str]) -> bool:
    haystack = text.lower()
    for term in terms:
        for alias in _background_term_aliases(term):
            if alias.lower() in haystack:
                return True
    return False


def _background_term_aliases(term: str) -> set[str]:
    lower = term.lower()
    if lower in {"ue5", "unreal"} or term in {"虚幻", "游戏引擎"}:
        return {"UE5", "Unreal", "虚幻", "游戏引擎"}
    if lower in {"ai", "llm"} or term in {"模型", "提示词"}:
        return {"AI", "LLM", "模型", "提示词"}
    return {term}


def _fallback_background_memory_content(text: str) -> str:
    if re.search(r"(UE5|Unreal|虚幻)", text, re.I):
        return "我之前翻过 UE5 蓝图和材质的入门资料"
    if re.search(r"Unity", text, re.I):
        return "我之前翻过一些 Unity 入门资料"
    if re.search(r"Blender", text, re.I):
        return "我之前翻过一些 Blender 入门资料"
    if re.search(r"Python", text, re.I):
        return "我之前翻过一些 Python 入门和调试资料"
    if re.search(r"(JavaScript|TypeScript|React|Vue)", text, re.I):
        return "我之前翻过一些前端工具的入门资料"
    if re.search(r"(Docker|Kubernetes|部署|服务器)", text, re.I):
        return "我之前翻过一些部署工具的入门资料"
    if re.search(r"(AI|LLM|模型|提示词)", text, re.I):
        return "我之前翻过一些 AI 工具的入门资料"
    return "我平时会翻一点技术工具的入门资料"


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



def _fallback_reply_with_self_memory(memory: MemoryCandidate) -> str:
    content = memory.content.strip()
    if content.startswith("我"):
        content = content[1:]
    if memory.kind in {"self_preference", "self_hobby"}:
        return f"嗯，我{content}。"
    if memory.kind == "self_past_event":
        return f"有点像，我{content}。"
    return f"嗯，我{content}，所以能懂一点。"



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


def _clean_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]
