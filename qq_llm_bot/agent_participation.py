from __future__ import annotations

import time

from qq_llm_bot.agent_common import clamp_float as _clamp_float
from qq_llm_bot.agent_fact_helpers import context_mentions_bot as _context_mentions_bot
from qq_llm_bot.agent_formatters import (
    format_recent_context as _format_recent_context,
    format_relationship as _format_relationship,
)
from qq_llm_bot.agent_policy_helpers import (
    _has_unresolved_identity_target,
    _looks_like_recent_interaction_followup,
    _proactive_value_type_allowed,
    _safe_participation_value_type,
)
from qq_llm_bot.config import AppConfig, ParticipationMode
from qq_llm_bot.final_qa import looks_like_live_event_context as _looks_like_live_event_context
from qq_llm_bot.llm import LLMClient
from qq_llm_bot.llm_json_helpers import complete_json as _complete_json
from qq_llm_bot.models import (
    ConversationSnapshot,
    MessageContext,
    ParticipationDecision,
    ParticipationValueType,
    PerceptionResult,
)


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
            value_type: ParticipationValueType = "answer" if perception.is_question else "direct_reply"
            return ParticipationDecision(
                "reply",
                "message is directed to the bot",
                mode,
                1.0,
                value_type,
                1.0,
                self._traffic_level(snapshot),
            )

        addressing_decision = await self._bot_name_addressing_decision(
            context,
            perception,
            mode,
            snapshot,
        )
        if addressing_decision is not None:
            return addressing_decision

        followup_decision = await self._recent_interaction_followup_decision(
            context,
            perception,
            mode,
            snapshot,
        )
        if followup_decision is not None:
            return followup_decision

        if mode == "passive":
            return ParticipationDecision("observe", "passive mode requires direct mention", mode, 0.0)

        gate_reason = self._active_gate_reason(context, perception, snapshot)
        if gate_reason:
            return ParticipationDecision("observe", gate_reason, mode, 0.0)

        traffic_level = self._traffic_level(snapshot)
        data = await _complete_json(
            self.llm,
            "你是 QQ 群拟人角色的插话决策器。只输出 JSON，不要解释。",
            (
                "判断机器人此刻是否应该主动插话。只能输出 observe 或 proactive_reply。"
                "不要为了展示能力而插话，要像群成员一样克制。"
                "主动插话必须提供增量价值，不能只是附和、共情、改写或重复别人观点。"
                "value_type 可选：answer、synthesis、missing_angle、useful_context、"
                "clarifying_question、humor、agreement、empathy、rephrase、none。"
                "只有 answer/synthesis/missing_angle/useful_context/clarifying_question 能稳定主动发言；"
                "humor 只能在聊天不密集且确实很贴切时使用；agreement/empathy/rephrase/none 必须 observe。"
                "如果最近 60 秒人类消息很多，只有能总结分歧、提出遗漏角度、补充有用上下文或问推进问题才插话。"
                "输出 JSON："
                '{"action":"observe|proactive_reply","score":0.0,'
                '"value_type":"answer|synthesis|missing_angle|useful_context|clarifying_question|humor|agreement|empathy|rephrase|none",'
                '"value_score":0.0,"reason":"短原因"}\n'
                f"最近消息：\n{_format_recent_context(snapshot)}\n"
                f"最近60秒人类消息数：{snapshot.recent_human_messages_60s}\n"
                f"最近120秒机器人消息数：{snapshot.recent_bot_messages_120s}\n"
                f"聊天密度：{traffic_level}\n"
                f"关系：{_format_relationship(snapshot)}\n"
                f"感知：topics={perception.topics}, emotion={perception.emotion_hint}\n"
                f"当前消息：{context.plain_text}"
            ),
            purpose="participation_policy",
        )
        if data:
            action = str(data.get("action", "observe"))
            score = _clamp_float(data.get("score", 0.0))
            value_type = _safe_participation_value_type(str(data.get("value_type", "none")).strip())
            value_score = _clamp_float(data.get("value_score", 0.0))
            reason = str(data.get("reason", "")).strip()[:160] or "active mode value decision"
            min_value_score = self._min_value_score(snapshot)
            if (
                action == "proactive_reply"
                and score >= 0.55
                and value_score >= min_value_score
                and _proactive_value_type_allowed(value_type, traffic_level)
            ):
                self._last_proactive_at[context.group_id] = int(time.time())
                return ParticipationDecision(
                    "proactive_reply",
                    reason,
                    mode,
                    score,
                    value_type,
                    value_score,
                    traffic_level,
                )
            if action == "proactive_reply":
                reason = (
                    f"proactive value gate rejected "
                    f"({value_type}:{value_score:.2f}, need {min_value_score:.2f}); {reason}"
                )
            return ParticipationDecision("observe", reason, mode, score, value_type, value_score, traffic_level)

        return ParticipationDecision(
            "observe",
            "active mode but no verified incremental value",
            mode,
            0.0,
            "none",
            0.0,
            traffic_level,
        )

    def _active_gate_reason(
        self,
        context: MessageContext,
        perception: PerceptionResult,
        snapshot: ConversationSnapshot,
    ) -> str | None:
        if _has_unresolved_identity_target(snapshot):
            return "active mode but target identity is unresolved"
        if len(context.plain_text.strip()) < 6:
            return "active mode but message is too short"
        if not perception.is_question and not perception.topics:
            return "active mode but no strong topic or question"
        if _looks_like_live_event_context(context, perception, snapshot):
            return "active mode but live event context is too time-sensitive"
        now = int(time.time())
        last = self._last_proactive_at.get(context.group_id, 0)
        if now - last < self.config.bot.proactive_cooldown_seconds:
            return "active mode but proactive cooldown is active"
        if snapshot.recent_bot_messages_120s >= 1:
            return "active mode but bot joined recently and was not asked"
        recent_bot_lines = [line for line in snapshot.recent_messages[-8:] if line.startswith("bot:")]
        if len(recent_bot_lines) >= 2:
            return "active mode but bot has spoken recently"
        return None

    def _traffic_level(self, snapshot: ConversationSnapshot) -> str:
        if snapshot.recent_human_messages_60s >= self.config.bot.proactive_busy_human_messages:
            return "busy"
        return "normal"

    def _min_value_score(self, snapshot: ConversationSnapshot) -> float:
        if self._traffic_level(snapshot) == "busy":
            return self.config.bot.proactive_busy_value_threshold
        return self.config.bot.proactive_value_threshold

    async def _bot_name_addressing_decision(
        self,
        context: MessageContext,
        perception: PerceptionResult,
        mode: ParticipationMode,
        snapshot: ConversationSnapshot,
    ) -> ParticipationDecision | None:
        if not _context_mentions_bot(context, self.config.bot.nicknames):
            return None

        traffic_level = self._traffic_level(snapshot)
        default_value_type: ParticipationValueType = "answer" if perception.is_question else "direct_reply"
        data = await _complete_json(
            self.llm,
            "你是 QQ 群机器人发言归属判断器。只输出 JSON，不要解释。",
            (
                "当前消息提到了机器人昵称。请判断消息里的昵称实际指代是不是本群机器人。"
                "除非你判断这个名字实际指代的不是本群机器人，否则允许机器人尝试参与。"
                "分类只能是 addressed_to_bot、discussing_bot、ambiguous、other_referent、not_relevant。"
                "addressed_to_bot：发言人在请求、询问、邀请、命令或直接回应机器人。"
                "discussing_bot：机器人是句子的宾语/话题，例如“可可的形象”、“让可可...”、"
                "“给可可...”、“和/跟可可...”、“限制可可...”、“可可的 trust”。"
                "ambiguous：上下文不够明确，但不能排除是在说本群机器人。"
                "other_referent：这个名字明显指向其他群员、别的机器人、角色、作品人物或转发记录中的人。"
                "not_relevant：只是同名词或无关内容。"
                "addressed_to_bot、discussing_bot、ambiguous 都可以回复；"
                "只有 other_referent/not_relevant 才默认观察。"
                "输出 JSON："
                '{"target":"addressed_to_bot|discussing_bot|ambiguous|other_referent|not_relevant",'
                '"confidence":0.0,'
                '"value_type":"answer|direct_reply|clarifying_question|none",'
                '"reason":"短原因"}\n'
                f"机器人昵称：{', '.join(self.config.bot.nicknames)}\n"
                f"最近消息：\n{_format_recent_context(snapshot)}\n"
                f"感知：question={perception.is_question}, topics={perception.topics}, "
                f"emotion={perception.emotion_hint}\n"
                f"当前消息：{context.plain_text}"
            ),
            purpose="addressing_gate",
        )
        if not data:
            return ParticipationDecision(
                "observe",
                "bot name mentioned but addressing is ambiguous",
                mode,
                0.0,
                "none",
                0.0,
                traffic_level,
            )

        target = str(data.get("target", "ambiguous")).strip().lower()
        confidence = _clamp_float(data.get("confidence", 0.0))
        reason = str(data.get("reason", "")).strip()[:140] or target or "bot name addressing gate"
        value_type = _safe_participation_value_type(str(data.get("value_type", default_value_type)))
        if value_type == "none":
            value_type = default_value_type

        if target in {"addressed_to_bot", "discussing_bot", "ambiguous"} and confidence >= 0.62:
            return ParticipationDecision(
                "reply",
                f"bot name addressing gate: {reason}",
                mode,
                confidence,
                value_type,
                confidence,
                traffic_level,
            )

        return ParticipationDecision(
            "observe",
            f"bot name mentioned but not addressed: {reason}",
            mode,
            confidence,
            "none",
            0.0,
            traffic_level,
        )

    async def _recent_interaction_followup_decision(
        self,
        context: MessageContext,
        perception: PerceptionResult,
        mode: ParticipationMode,
        snapshot: ConversationSnapshot,
    ) -> ParticipationDecision | None:
        recent_reply = snapshot.recent_bot_reply_to_user.strip()
        current_text = context.plain_text.strip()
        if not recent_reply or not current_text:
            return None

        traffic_level = self._traffic_level(snapshot)
        data = await _complete_json(
            self.llm,
            "你是 QQ 群机器人续聊门禁。只输出 JSON，不要解释。",
            (
                "判断当前这条没有点名机器人的消息，是否是在延续同一用户刚才和机器人的互动。"
                "只有当用户明显在追问、补充、回应机器人上一句，或省略了机器人名字但仍接着上一轮聊时才 reply。"
                "如果是换新话题、对其他群友说话、自言自语、纯表情/感叹、或上下文不够明确，必须 observe。"
                "输出 JSON："
                '{"action":"reply|observe","confidence":0.0,'
                '"value_type":"answer|direct_reply|clarifying_question|none","reason":"短原因"}\n'
                f"上次机器人回复该用户（约 {snapshot.recent_bot_reply_to_user_seconds} 秒前）："
                f"{recent_reply}\n"
                f"最近消息：\n{_format_recent_context(snapshot)}\n"
                f"感知：question={perception.is_question}, topics={perception.topics}, "
                f"emotion={perception.emotion_hint}\n"
                f"当前消息：{current_text}"
            ),
            purpose="followup_gate",
        )
        if data:
            action = str(data.get("action", "observe")).strip().lower()
            confidence = _clamp_float(data.get("confidence", 0.0))
            value_type = _safe_participation_value_type(
                str(data.get("value_type", "answer" if perception.is_question else "direct_reply"))
            )
            if action == "reply" and confidence >= 0.62 and value_type != "none":
                reason = str(data.get("reason", "")).strip()[:140] or "recent interaction follow-up"
                return ParticipationDecision(
                    "reply",
                    f"recent interaction follow-up: {reason}",
                    mode,
                    confidence,
                    value_type,
                    confidence,
                    traffic_level,
                )
            return None

        if _looks_like_recent_interaction_followup(current_text, perception):
            value_type = "answer" if perception.is_question else "direct_reply"
            return ParticipationDecision(
                "reply",
                "recent interaction follow-up: heuristic continuation cue",
                mode,
                0.66,
                value_type,
                0.66,
                traffic_level,
            )
        return None
