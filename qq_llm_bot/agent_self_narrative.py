from __future__ import annotations

import re
from dataclasses import replace
from typing import Any

from qq_llm_bot.agent_formatters import (
    format_memories as _format_memories,
    format_recent_context as _format_recent_context,
    join_lines as _join_lines,
)
from qq_llm_bot.agent_models import SelfNarrativePlan, SelfNarrativePreparation
from qq_llm_bot.agent_self_memory import (
    SELF_NARRATIVE_KINDS,
    SelfMemoryLedger as SelfMemoryLedger,
    _as_bool,
    _clamp_float,
    _clean_list,
    _clean_self_narrative_content,
    _fallback_background_memory_content,
    _fallback_reply_with_self_memory as _fallback_reply_with_self_memory,
    _has_relevant_self_background,
    _heuristic_self_narrative_status,
    _needs_self_background_for_topic,
    _safe_fictionality,
    _safe_self_kind,
    _self_memory_candidate,
    _strip_bot_call,
)
from qq_llm_bot.final_qa import UNSAFE_SELF_PATTERN
from qq_llm_bot.llm import LLMClient
from qq_llm_bot.llm_json_helpers import complete_json as _complete_json
from qq_llm_bot.models import (
    ConversationSnapshot,
    MemoryCandidate,
    MessageContext,
    ParticipationDecision,
    PerceptionResult,
)


class SelfNarrativeAgent:
    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    async def prepare(
        self,
        context: MessageContext,
        perception: PerceptionResult,
        decision: ParticipationDecision,
        snapshot: ConversationSnapshot,
    ) -> SelfNarrativePreparation:
        if decision.action == "observe":
            return SelfNarrativePreparation()

        plan = await self._plan(context, perception, decision, snapshot)
        if plan.requires_background and not plan.should_invent:
            return SelfNarrativePreparation(
                requires_background=True,
                background_available=True,
                fallback_caution=plan.fallback_caution,
            )
        if not plan.needs_self_narrative or not plan.should_invent:
            return SelfNarrativePreparation()

        candidate = await self._draft_candidate(context, plan, snapshot)
        if candidate is None:
            return self._background_gate_result(
                decision,
                plan,
                "no suitable self background candidate",
            )

        checked = await self._check_candidate(context, candidate, snapshot)
        if checked:
            return SelfNarrativePreparation(
                memories=[checked],
                requires_background=plan.requires_background,
                background_available=True,
                fallback_caution=plan.fallback_caution,
            )
        return self._background_gate_result(
            decision,
            plan,
            "candidate rejected by self consistency check",
        )

    def _background_gate_result(
        self,
        decision: ParticipationDecision,
        plan: SelfNarrativePlan,
        reason: str,
    ) -> SelfNarrativePreparation:
        if not plan.requires_background:
            return SelfNarrativePreparation()
        caution = plan.fallback_caution or (
            "缺少可引用的个人学习或使用背景；只能按一般理解谨慎回答，"
            "不要声称亲身经历。"
        )
        return SelfNarrativePreparation(
            requires_background=True,
            background_available=False,
            blocked=decision.action == "proactive_reply",
            block_reason=reason,
            fallback_caution=caution,
        )

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

        background_plan = self._background_plan(context, perception, decision, snapshot)
        if background_plan is not None:
            return background_plan

        if self._should_skip_llm_plan(context, perception, decision):
            return fallback

        data = await _complete_json(
            self.llm,
            "你是拟人角色的自我叙事规划器。只输出 JSON，不要解释。",
            (
                "判断这次回复是否需要引用或新增机器人自己的轻量自我设定。"
                "只有当被问到自己的偏好、习惯、经历，或自然共情确实需要时才需要。"
                "如果要参与技术、工具、创作流程、游戏引擎等经验型话题，并准备给出建议，"
                "需要检查是否有个人学习/使用背景支撑。"
                "缺少背景但又需要背景时，只允许新增轻量背景，例如“我之前翻过 UE5 蓝图和材质入门资料”。"
                "如果稳定人格里已经有足够信息可以回答，不要新增自我记忆。"
                "允许轻量虚构，但禁止真实地址、学校、公司、亲属、线下行动等具体现实身份。"
                "输出 JSON："
                '{"needs_self_narrative":bool,"purpose":"answer_question|empathy|banter|topic_join",'
                '"allowed_kinds":["self_hobby|self_habit|self_past_event|self_preference|self_background"],'
                '"should_invent":bool,"requires_background":bool,'
                '"fallback_caution":"缺少背景时给回复器的谨慎提示","reason":"短原因"}\n'
                f"参与决策：{decision.action}, {decision.reason}\n"
                f"感知：question={perception.is_question}, topics={perception.topics}\n"
                f"已有 self memory：\n{_format_memories(snapshot.self_memories)}\n"
                f"消息：{context.plain_text}"
            ),
            purpose="self_narrative_plan",
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
            requires_background=_as_bool(data.get("requires_background"), fallback.requires_background),
            fallback_caution=str(data.get("fallback_caution", "")).strip()[:160],
        )

    def _background_plan(
        self,
        context: MessageContext,
        perception: PerceptionResult,
        decision: ParticipationDecision,
        snapshot: ConversationSnapshot,
    ) -> SelfNarrativePlan | None:
        if not _needs_self_background_for_topic(context, perception, decision):
            return None
        caution = (
            "这个话题需要个人学习或使用背景；如果没有可引用背景，"
            "只能按一般理解谨慎回答，不要说自己用过或做过项目。"
        )
        if _has_relevant_self_background(context, perception, snapshot):
            return SelfNarrativePlan(
                False,
                purpose="topic_join",
                allowed_kinds=("self_background", "self_past_event", "self_habit"),
                should_invent=False,
                reason="existing self background supports topic",
                requires_background=True,
                fallback_caution=caution,
            )
        return SelfNarrativePlan(
            True,
            purpose="topic_join",
            allowed_kinds=("self_background", "self_past_event", "self_habit"),
            should_invent=True,
            reason="topic advice needs lightweight self background",
            requires_background=True,
            fallback_caution=caution,
        )

    def _should_skip_llm_plan(
        self,
        context: MessageContext,
        perception: PerceptionResult,
        decision: ParticipationDecision,
    ) -> bool:
        if decision.action == "proactive_reply":
            return True
        if not context.is_direct:
            return True
        text = _strip_bot_call(context.plain_text, [])
        return "你" not in text and not perception.is_self_disclosure

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
                "如果规划要求补话题背景，只写轻量学习/接触背景，例如“我之前翻过某工具的入门资料”，"
                "不要写成做过真实项目、在公司使用、上班经历或专家履历。"
                "不要和已有 self memory 冲突；如果不适合新增，content 置空。"
                "输出 JSON："
                '{"kind":"self_hobby|self_habit|self_past_event|self_preference|self_background",'
                '"content":"第一人称短句，不超过40字","fictionality":"fictional_light|metaphorical",'
                '"confidence":0.0,"importance":0.0}\n'
                f"规划：purpose={plan.purpose}, allowed={list(plan.allowed_kinds)}, reason={plan.reason}\n"
                f"人格：\n{_join_lines(snapshot.persona_lines)}\n"
                f"已有 self memory：\n{_format_memories(snapshot.self_memories)}\n"
                f"群聊上下文：\n{_format_recent_context(snapshot)}\n"
                f"用户消息：{context.plain_text}"
            ),
            purpose="self_narrative_draft",
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
        if plan.requires_background:
            kind = "self_background" if "self_background" in plan.allowed_kinds else kind
            content = _fallback_background_memory_content(text)
        elif "海" in text:
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
            purpose="self_narrative_check",
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



