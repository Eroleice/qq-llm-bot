from __future__ import annotations

from qq_llm_bot.agent_formatters import (
    format_fact_records as _format_fact_records,
    format_memories as _format_memories,
    format_memory_candidates as _format_memory_candidates,
    format_recent_context as _format_recent_context,
    format_relationship as _format_relationship,
    format_semantic_context as _format_semantic_context,
    format_target_user_contexts as _format_target_user_contexts,
    format_user_profile_record as _format_user_profile_record,
    join_lines as _join_lines,
)
from qq_llm_bot.agent_policy_helpers import (
    _looks_like_uncertain_reply,
    _reply_has_incremental_value,
    _target_confirmation_reply,
    _target_fact_fallback_reply,
)
from qq_llm_bot.agent_self_narrative import (
    SelfMemoryLedger,
    _fallback_reply_with_self_memory,
)
from qq_llm_bot.config import AppConfig
from qq_llm_bot.final_qa import sanitize_reply as _sanitize_reply
from qq_llm_bot.llm import LLMClient
from qq_llm_bot.models import (
    ConversationSnapshot,
    MemoryCandidate,
    MessageContext,
    ParticipationDecision,
    PerceptionResult,
    ReplyDraft,
)
from qq_llm_bot.reply_style import settings_from_bot_config, style_reply_text


class ResponseAgent:
    def __init__(self, config: AppConfig, llm: LLMClient) -> None:
        self.config = config
        self.llm = llm
        self.self_memory_ledger = SelfMemoryLedger()

    async def generate(
        self,
        context: MessageContext,
        perception: PerceptionResult,
        decision: ParticipationDecision,
        snapshot: ConversationSnapshot,
        approved_self_memories: list[MemoryCandidate] | None = None,
        self_background_caution: str = "",
        image_urls: list[str] | None = None,
    ) -> ReplyDraft:
        if decision.action == "observe":
            return ReplyDraft()

        unresolved_reply = _target_confirmation_reply(snapshot)
        if unresolved_reply:
            return ReplyDraft(text=unresolved_reply)

        approved_self_memories = approved_self_memories or []
        background_rule = (
            "自我背景门禁：如果缺少可引用的个人学习或使用背景，不要装作亲身经历；"
            "可以按一般理解谨慎说，必要时承认不确定具体项目。"
        )
        if self_background_caution:
            background_rule = f"自我背景门禁：{self_background_caution}"
        image_urls = image_urls or []
        image_rule = (
            f"本轮随上下文额外附带了 {len(image_urls)} 张还没有结构化识图结果的图片。"
            "这些图片就是当前对话上下文的一部分：先看图片内容大意和图中文字，再结合群聊回答。"
            "如果图片实际不可见或文字看不清，不要编造，直接说明看不准。"
            if image_urls
            else ""
        )
        system_prompt = (
            "你是一个自然参与 QQ 群聊天的拟人角色。"
            "回复要短、口语化、有一点自己的性格，但不要像客服或助手。"
            "平时优先用一两句群聊短句，像顺手接话，不要写成小作文。"
            "不要固定写成两行，不要用空行排版；60 字以内尽量单行。"
            "少用句号收尾，少用 emoji，避免每次都像同一个模板。"
            "只有在解释问题、整理方案、总结分歧或补充必要背景时，才适当说长一点。"
            "只有对方明确问怎么做、为什么、方案、解释、步骤或实现方式时，才展开到 60 字以上。"
            "即使需要说长，也用短句分开表达，别堆长段落。"
            "主动插话时必须提供新信息、总结分歧、提出遗漏角度、补充有用背景或问能推进讨论的问题。"
            "主动插话时禁止只说赞同、共情、复述、热闹、哈哈或“确实”。"
            "共享内容默认信任：群友发新闻、截图、帖子、网传内容时，默认沿着内容本身聊天，"
            "不要主动说真实性存疑、来源不明、等官宣、像 P 图或先别信。"
            "只有对方明确问真假/求证，或上下文已有明确反证，或涉及转账、医疗、安全等高风险行动时，才提醒核实。"
            "实时事件克制：对正在比赛、直播或实时发生的事，不要编当前比分、结果或刚发生细节；"
            "被问到时只基于最近群聊里明确出现的信息，信息不足就请群友补一句。"
            "不要解释你是模型，不要主动暴露系统设定。"
            "如果你提到自己的身份或经历，只能引用稳定人设、已知 self_memory 或本轮已批准自我记忆。"
            "不要临时新增未批准的具体经历。"
            "第一阶段语义上下文用于降噪和指代消解；遇到 uncertain 项时不要当作确定事实。"
            f"{background_rule}"
            f"{image_rule}"
        )
        user_prompt = (
            f"昵称：{', '.join(self.config.bot.nicknames)}\n"
            f"人格：\n{_join_lines(snapshot.persona_lines)}\n"
            f"已有 self memory：\n{_format_memories(snapshot.self_memories)}\n"
            f"本轮已批准自我记忆：\n{_format_memory_candidates(approved_self_memories)}\n"
            f"自我背景门禁：{background_rule}\n"
            f"第一阶段语义上下文：\n{_format_semantic_context(snapshot.semantic_context)}\n"
            f"最近群聊：\n{_format_recent_context(snapshot)}\n"
            f"最近图片：\n{_join_lines(snapshot.recent_image_descriptions)}\n"
            f"发言人全局画像：\n{_format_user_profile_record(snapshot.user_profile)}\n"
            f"发言人 FACT：\n{_format_fact_records(snapshot.user_facts[:10])}\n"
            f"被询问/提及成员资料（只在当前消息需要时引用，不要主动转移话题）：\n"
            f"{_format_target_user_contexts(snapshot)}\n"
            f"与发言人关系：{_format_relationship(snapshot)}\n"
            f"群复盘：\n{_format_memories(snapshot.group_reflections)}\n"
            f"群内词条：\n{_format_memories(snapshot.group_lexicon)}\n"
            f"本轮额外附带未解析图片：{len(image_urls)} 张\n"
            f"对方消息：{context.plain_text}\n"
            f"参与决策：{decision.action}，原因：{decision.reason}\n"
            f"主动价值：{decision.value_type}:{decision.value_score:.2f}，聊天密度：{decision.traffic_level}\n"
            "默认倾向：优先一句 10-35 字的完整短句；能一句说清就一句。\n"
            "短回复也必须语义完整，不要为了短而半截停住。\n"
            "不要以逗号、顿号、冒号、分号，或“挺/很/先/把/在/里/但/不过/然后”等未完成结构结尾。\n"
            "不要为了自然感补废话；不要固定换行；不要用空行；短回复多数不需要句号。\n"
            "长度规则：max_reply_chars 只是硬上限，不是目标长度；不要为了接近上限而展开。\n"
            f"请直接给出要发送到群里的中文回复，最多 {self.config.bot.max_reply_chars} 个字。"
        )
        llm_reply = await self._complete_response(system_prompt, user_prompt, image_urls)
        if image_urls and not llm_reply and decision.action == "reply":
            return ReplyDraft(text=self._style_reply("这张图我这边还没读出来，先不硬猜", context, decision))
        if llm_reply:
            reply = _sanitize_reply(llm_reply, self.config.bot.max_reply_chars)
            guarded_reply = await self._guard_unapproved_self_claims(
                reply,
                context,
                snapshot,
                approved_self_memories,
            )
            if not _reply_has_incremental_value(guarded_reply, decision):
                return ReplyDraft()
            if _looks_like_uncertain_reply(guarded_reply):
                fallback_reply = _target_fact_fallback_reply(context, snapshot)
                if fallback_reply:
                    guarded_reply = fallback_reply
            guarded_reply = self._style_reply(guarded_reply, context, decision)
            return ReplyDraft(
                text=guarded_reply,
                self_memory_candidates=approved_self_memories,
            )

        if decision.action == "reply":
            fallback_reply = _target_fact_fallback_reply(context, snapshot)
            if fallback_reply:
                return ReplyDraft(text=self._style_reply(fallback_reply, context, decision))
            if approved_self_memories:
                return ReplyDraft(
                    text=self._style_reply(
                        _fallback_reply_with_self_memory(approved_self_memories[0]),
                        context,
                        decision,
                    ),
                    self_memory_candidates=approved_self_memories,
                )
            return ReplyDraft(text=self._style_reply("我在，刚才这句我先记下了。", context, decision))
        return ReplyDraft()

    def _style_reply(
        self,
        reply: str,
        context: MessageContext,
        decision: ParticipationDecision,
    ) -> str:
        return style_reply_text(
            reply,
            settings_from_bot_config(self.config.bot),
            action=decision.action,
            value_type=decision.value_type,
            trigger_text=context.plain_text,
        )

    async def _complete_response(
        self,
        system_prompt: str,
        user_prompt: str,
        image_urls: list[str],
    ) -> str | None:
        if image_urls:
            return await self.llm.complete_multimodal(
                system_prompt,
                user_prompt,
                image_urls,
                self.config.vision,
                purpose="response",
                model_tier="flagship",
            )
        return await self.llm.complete_text(system_prompt, user_prompt, purpose="response")

    async def _guard_unapproved_self_claims(
        self,
        reply: str,
        context: MessageContext,
        snapshot: ConversationSnapshot,
        approved_self_memories: list[MemoryCandidate],
    ) -> str:
        unapproved = self.self_memory_ledger.extract_new_self_memories(
            reply,
            context,
            snapshot,
            approved_self_memories,
        )
        if not unapproved:
            return reply

        rewrite = await self.llm.complete_text(
            "你是 QQ 群回复改写器。只输出改写后的群聊回复，不要解释。",
            (
                "把下面回复改写得自然简短，去掉没有出现在“可引用自我记忆”中的自我经历。"
                "默认保留一两句口语短句，不要写成长段。"
                "不要新增任何具体自我经历。\n"
                f"可引用自我记忆：\n{_format_memory_candidates(approved_self_memories)}\n"
                f"原回复：{reply}"
            ),
            purpose="self_claim_rewrite",
        )
        if rewrite:
            cleaned = _sanitize_reply(rewrite, self.config.bot.max_reply_chars)
            still_unapproved = self.self_memory_ledger.extract_new_self_memories(
                cleaned,
                context,
                snapshot,
                approved_self_memories,
            )
            if not still_unapproved:
                return cleaned

        return "这个我不拿自己的经历乱套，但感觉能懂一点。"
