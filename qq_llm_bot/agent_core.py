from __future__ import annotations

from qq_llm_bot.agent_formatters import (
    format_fact_records as _format_fact_records,
    format_memories as _format_memories,
    format_recent_context as _format_recent_context,
    format_relationship as _format_relationship,
    format_target_user_contexts as _format_target_user_contexts,
    format_user_profile_record as _format_user_profile_record,
    semantic_context_has_content as _semantic_context_has_content,
)
from qq_llm_bot.agent_participation import ParticipationPolicyAgent as ParticipationPolicyAgent
from qq_llm_bot.agent_policy_helpers import (
    _clamp_delta as _clamp_delta,
    _fallback_semantic_context as _fallback_semantic_context,
    _needs_context_understanding as _needs_context_understanding,
    _semantic_context_from_json as _semantic_context_from_json,
)
from qq_llm_bot.agent_response import ResponseAgent as ResponseAgent
from qq_llm_bot.config import AppConfig
from qq_llm_bot.llm import LLMClient
from qq_llm_bot.llm_json_helpers import complete_json as _complete_json
from qq_llm_bot.models import (
    ConversationSnapshot,
    MessageContext,
    ParticipationDecision,
    PerceptionResult,
    RelationDelta,
    SemanticContext,
)
from qq_llm_bot.relationship_summary import clean_relationship_summary_patch


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
            purpose="relationship",
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


class ContextUnderstandingAgent:
    def __init__(self, config: AppConfig, llm: LLMClient) -> None:
        self.config = config
        self.llm = llm

    async def analyze(
        self,
        context: MessageContext,
        perception: PerceptionResult,
        decision: ParticipationDecision,
        snapshot: ConversationSnapshot,
    ) -> SemanticContext:
        fallback = _fallback_semantic_context(context, snapshot)
        if (
            not self.config.bot.context_understanding_enabled
            or decision.action == "observe"
            or not _needs_context_understanding(snapshot)
        ):
            return fallback

        data = await _complete_json(
            self.llm,
            (
                "你是 QQ 群聊上下文整理器。不要生成回复，只输出 JSON。"
                "你的任务是降噪、保留相关上下文、解析指代和成员称呼。"
                "成员身份必须优先写成 QQ:<id>；不确定就标注 uncertain，不要强行猜。"
            ),
            (
                "请为下一阶段回复模型整理上下文。最近几条原文可以保留，但要指出哪些真正相关。"
                "解析“我/你/他/她/ta/这个/那个”等指代；群成员称呼使用候选成员资料。"
                "只保留与当前话题相关的成员认知，不要把无关画像塞进去。"
                "输出 JSON："
                '{"current_intent":"当前用户意图",'
                '"relevant_messages":["相关上下文，尽量带 QQ 或显示名"],'
                '"resolved_references":["指代词/称呼 -> QQ:id 或对象，含置信说明"],'
                '"member_context":["与话题相关的成员认知"],'
                '"uncertain_references":["不确定的指代或称呼"],'
                '"ignored_noise":["可忽略噪音类型"]}\n'
                f"机器人昵称：{', '.join(self.config.bot.nicknames)}\n"
                f"当前发言人：QQ:{context.user_id}，昵称：{context.sender_name or context.sender_nickname or '-'}\n"
                f"当前消息：{context.plain_text}\n"
                f"感知：question={perception.is_question}, topics={perception.topics}, "
                f"emotion={perception.emotion_hint}\n"
                f"参与决策：{decision.action}，原因：{decision.reason}\n"
                f"最近群聊：\n{_format_recent_context(snapshot)}\n"
                f"当前发言人资料：\n{_format_user_profile_record(snapshot.user_profile)}\n"
                f"当前发言人 FACT：\n{_format_fact_records(snapshot.user_facts[:8])}\n"
                f"被询问/提及成员资料：\n{_format_target_user_contexts(snapshot)}\n"
                f"与发言人关系：{_format_relationship(snapshot)}\n"
                f"群复盘：\n{_format_memories(snapshot.group_reflections)}\n"
                f"群内词条：\n{_format_memories(snapshot.group_lexicon)}\n"
            ),
            purpose="context_understanding",
        )
        parsed = _semantic_context_from_json(data)
        return parsed if _semantic_context_has_content(parsed) else fallback
