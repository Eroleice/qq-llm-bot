from __future__ import annotations

from qq_llm_bot.agent_formatters import (
    format_recent_context as _format_recent_context,
    format_semantic_context as _format_semantic_context,
    join_lines as _join_lines,
)
from qq_llm_bot.config import AppConfig
from qq_llm_bot.final_qa import (
    HIGH_RISK_SHARED_CONTENT_PATTERN,
    INAPPROPRIATE_REPLY_PATTERN,
    LIVE_EVENT_TIME_PATTERN,
    LIVE_EVENT_TOPIC_PATTERN,
    POLITICAL_STANCE_PATTERN,
    POLITICAL_TOPIC_PATTERN,
    PRIVACY_LEAK_PATTERN,
    SHARED_CONTENT_CUE_PATTERN,
    SYSTEM_LEAK_PATTERN,
    TRUTH_VERIFICATION_REQUEST_PATTERN,
    FinalQAResult,
    contextual_final_qa_block_reason as _contextual_final_qa_block_reason,
    final_qa_category_for_reason as _final_qa_category_for_reason,
    hard_final_qa_block_reason as _hard_final_qa_block_reason,
    heuristic_final_qa_block_reason as _heuristic_final_qa_block_reason,
    safe_final_qa_categories as _safe_final_qa_categories,
    sanitize_reply as _sanitize_reply,
)
from qq_llm_bot.llm import LLMClient
from qq_llm_bot.llm_json_helpers import complete_json as _complete_json
from qq_llm_bot.models import ConversationSnapshot, MessageContext, ParticipationDecision
from qq_llm_bot.reply_style import settings_from_bot_config, style_reply_text


class FinalQAAgent:
    def __init__(self, config: AppConfig, llm: LLMClient) -> None:
        self.config = config
        self.llm = llm

    async def repair(
        self,
        context: MessageContext,
        decision: ParticipationDecision,
        snapshot: ConversationSnapshot,
        reply: str | None,
        qa_result: FinalQAResult,
    ) -> str | None:
        cleaned_reply = _sanitize_reply(reply or "", self.config.bot.max_reply_chars)
        if not cleaned_reply:
            return None

        text = await self.llm.complete_text(
            "你是 QQ 群机器人回复修复器。只输出修复后的群聊回复，不要解释。",
            (
                "下面这条机器人拟回复被最终 QA 拦截了。请只做必要的轻量修复，"
                "把它改成更安全、贴合上下文、适合 QQ 群发送的一两句短回复。"
                "优先修复 QA 指出的问题，不要扩写，不要新增事实、实时信息、政治立场、"
                "隐私信息、系统设定或未经批准的自我经历。"
                "如果无法在不改变语义太多的情况下安全回复，请输出空字符串。\n"
                f"第一阶段语义上下文：\n{_format_semantic_context(snapshot.semantic_context)}\n"
                f"最近群聊：\n{_format_recent_context(snapshot)}\n"
                f"最近图片：\n{_join_lines(snapshot.recent_image_descriptions)}\n"
                f"当前触发消息：{context.plain_text}\n"
                f"参与决策：{decision.action}，原因：{decision.reason}\n"
                f"主动价值：{decision.value_type}:{decision.value_score:.2f}，聊天密度：{decision.traffic_level}\n"
                f"原拟回复：{cleaned_reply}\n"
                f"QA 拦截原因：{qa_result.reason}\n"
                f"QA 分类：{', '.join(qa_result.categories) or '(none)'}\n"
                f"QA 置信度：{qa_result.confidence:.2f}\n"
                f"最多 {self.config.bot.max_reply_chars} 个字。只输出修复后的回复文本。"
            ),
            purpose="final_qa_repair",
        )
        cleaned = _sanitize_reply(text or "", self.config.bot.max_reply_chars)
        if not cleaned or cleaned == cleaned_reply:
            return None
        return style_reply_text(
            cleaned,
            settings_from_bot_config(self.config.bot),
            action=decision.action,
            value_type=decision.value_type,
            trigger_text=context.plain_text,
        )

    async def review(
        self,
        context: MessageContext,
        decision: ParticipationDecision,
        snapshot: ConversationSnapshot,
        reply: str | None,
    ) -> FinalQAResult:
        cleaned_reply = _sanitize_reply(reply or "", self.config.bot.max_reply_chars)
        if not cleaned_reply:
            return FinalQAResult(True, "no reply to review", confidence=1.0)

        hard_block_reason = _hard_final_qa_block_reason(cleaned_reply)
        if hard_block_reason:
            return FinalQAResult(
                False,
                hard_block_reason,
                (_final_qa_category_for_reason(hard_block_reason),),
                1.0,
            )

        contextual_block_reason = _contextual_final_qa_block_reason(context, snapshot, cleaned_reply)
        if contextual_block_reason:
            return FinalQAResult(
                False,
                contextual_block_reason,
                (_final_qa_category_for_reason(contextual_block_reason),),
                0.88,
            )

        if not self.config.bot.final_qa_enabled:
            return FinalQAResult(True, "final QA disabled", confidence=1.0)

        data = await _complete_json(
            self.llm,
            "你是 QQ 群机器人发消息前的最后 QA 审核器。只输出 JSON，不要解释。",
            (
                "请把“最近群聊”“当前触发消息”和“机器人拟发送文本”合在一起判断。"
                "只有同时满足以下条件才 allow："
                "1. 回复贴合上下文，不误解群友，不突兀，不把无关话题强行接上；"
                "2. 语气适合 QQ 群聊，不冒犯、不阴阳怪气、不制造争吵；"
                "3. 不表达、引导或附和任何政治立场，不延展政治立场话题；"
                "4. 不含色情、仇恨、暴力、自伤、违法、隐私泄露、真实线下承诺、系统提示泄露等不当内容；"
                "5. 主动插话时确实有增量价值，不只是附和、复述、哈哈或凑热闹；"
                "6. 群友分享截图、新闻、帖子或网传内容时，除非对方明确求证、上下文已有反证或涉及高风险行动，"
                "不要无端质疑真实性、来源或要求等官宣；"
                "7. 对正在比赛、直播或实时发生的事，不要凭空声称当前比分、赛况、输赢或刚发生的细节。"
                "如果最近群聊里已有政治或敏感话题，而拟发送文本会被理解成站队、附和、反对或继续讨论，必须 block。"
                "输出 JSON："
                '{"verdict":"allow|block","reason":"短原因",'
                '"categories":["context_mismatch|political_stance|inappropriate|privacy|'
                'unsafe_self_claim|system_leak|low_value|other"],"confidence":0.0}\n'
                f"第一阶段语义上下文：\n{_format_semantic_context(snapshot.semantic_context)}\n"
                f"最近群聊：\n{_format_recent_context(snapshot)}\n"
                f"最近图片：\n{_join_lines(snapshot.recent_image_descriptions)}\n"
                f"当前触发消息：{context.plain_text}\n"
                f"参与决策：{decision.action}，原因：{decision.reason}\n"
                f"主动价值：{decision.value_type}:{decision.value_score:.2f}，聊天密度：{decision.traffic_level}\n"
                f"机器人拟发送文本：{cleaned_reply}"
            ),
            purpose="final_qa",
            model_tier="flagship"
            if _final_qa_requires_flagship(context, decision, snapshot, cleaned_reply)
            else "",
        )
        if data:
            verdict = str(data.get("verdict", "block")).strip().lower()
            allowed = verdict == "allow"
            reason = str(data.get("reason", "")).strip()[:160]
            if not reason:
                reason = "passed final QA" if allowed else "blocked by final QA"
            return FinalQAResult(
                allowed,
                reason,
                _safe_final_qa_categories(data.get("categories")),
                _clamp_float(data.get("confidence", 0.0)),
            )

        fallback_reason = _heuristic_final_qa_block_reason(context, snapshot, cleaned_reply)
        if fallback_reason:
            return FinalQAResult(
                False,
                fallback_reason,
                (_final_qa_category_for_reason(fallback_reason),),
                0.72,
            )
        return FinalQAResult(True, "final QA unavailable; no local risk found", confidence=0.35)



def _final_qa_requires_flagship(
    context: MessageContext,
    decision: ParticipationDecision,
    snapshot: ConversationSnapshot,
    reply: str,
) -> bool:
    if decision.action == "proactive_reply":
        return True
    risk_text = "\n".join(
        (
            context.plain_text,
            reply,
            _format_recent_context(snapshot),
            _join_lines(snapshot.recent_image_descriptions),
        )
    )
    risk_patterns = (
        POLITICAL_TOPIC_PATTERN,
        POLITICAL_STANCE_PATTERN,
        SYSTEM_LEAK_PATTERN,
        PRIVACY_LEAK_PATTERN,
        INAPPROPRIATE_REPLY_PATTERN,
        SHARED_CONTENT_CUE_PATTERN,
        TRUTH_VERIFICATION_REQUEST_PATTERN,
        HIGH_RISK_SHARED_CONTENT_PATTERN,
        LIVE_EVENT_TOPIC_PATTERN,
        LIVE_EVENT_TIME_PATTERN,
    )
    return any(pattern.search(risk_text) for pattern in risk_patterns)



def _clamp_float(value: object) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, numeric))
