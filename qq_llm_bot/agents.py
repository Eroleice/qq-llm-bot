from __future__ import annotations

import re
import time

from qq_llm_bot.config import AppConfig, ParticipationMode
from qq_llm_bot.llm import LLMClient
from qq_llm_bot.models import (
    MemoryCandidate,
    MessageContext,
    ParticipationDecision,
    PerceptionResult,
    PipelineResult,
)


class PerceptionAgent:
    def analyze(self, context: MessageContext) -> PerceptionResult:
        text = context.plain_text.strip()
        return PerceptionResult(
            is_question=any(mark in text for mark in ("?", "？", "吗", "怎么", "为什么", "咋")),
            is_self_disclosure=bool(re.search(r"(我叫|我是|我喜欢|我讨厌|我在|我住|我最近)", text)),
            mentions_bot=context.is_direct,
            topics=_extract_topics(text),
            emotion_hint=_emotion_hint(text),
        )


class MemoryCuratorAgent:
    SELF_DISCLOSURE_PATTERNS = (
        ("alias", re.compile(r"我叫\s*([^\s，。,.!！?？]{1,16})")),
        ("identity", re.compile(r"我是\s*([^，。,.!！?？]{1,32})")),
        ("preference", re.compile(r"我喜欢\s*([^，。,.!！?？]{1,32})")),
        ("dislike", re.compile(r"我讨厌\s*([^，。,.!！?？]{1,32})")),
        ("location", re.compile(r"我住(?:在)?\s*([^，。,.!！?？]{1,32})")),
        ("experience", re.compile(r"我最近\s*([^，。,.!！?？]{1,40})")),
    )

    def extract(self, context: MessageContext, perception: PerceptionResult) -> list[MemoryCandidate]:
        if not perception.is_self_disclosure:
            return []

        memories: list[MemoryCandidate] = []
        for kind, pattern in self.SELF_DISCLOSURE_PATTERNS:
            for match in pattern.finditer(context.plain_text):
                content = match.group(1).strip()
                if content:
                    memories.append(
                        MemoryCandidate(
                            owner_type="user",
                            owner_id=context.user_id,
                            kind=kind,
                            content=content,
                            confidence=0.72,
                            evidence_message_id=context.message_id,
                        )
                    )
        return memories


class RelationshipAgent:
    def familiarity_delta(self, perception: PerceptionResult) -> int:
        if perception.mentions_bot:
            return 2
        return 1


class ParticipationPolicyAgent:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self._last_proactive_at: dict[str, int] = {}

    def decide(
        self,
        context: MessageContext,
        perception: PerceptionResult,
        mode: ParticipationMode,
    ) -> ParticipationDecision:
        if mode == "silent":
            return ParticipationDecision("observe", "group is in silent mode", mode)

        if context.is_direct:
            return ParticipationDecision("reply", "message is directed to the bot", mode)

        if mode == "passive":
            return ParticipationDecision("observe", "passive mode requires direct mention", mode)

        if self._can_join_proactively(context, perception):
            self._last_proactive_at[context.group_id] = int(time.time())
            return ParticipationDecision("proactive_reply", "active mode and topic looks discussable", mode)

        return ParticipationDecision("observe", "active mode but no strong reason to join", mode)

    def _can_join_proactively(self, context: MessageContext, perception: PerceptionResult) -> bool:
        if len(context.plain_text.strip()) < 6:
            return False
        if not perception.is_question and not perception.topics:
            return False

        now = int(time.time())
        last = self._last_proactive_at.get(context.group_id, 0)
        return now - last >= self.config.bot.proactive_cooldown_seconds


class ResponseAgent:
    def __init__(self, config: AppConfig, llm: LLMClient) -> None:
        self.config = config
        self.llm = llm

    async def generate(
        self,
        context: MessageContext,
        perception: PerceptionResult,
        decision: ParticipationDecision,
    ) -> str | None:
        if decision.action == "observe":
            return None

        system_prompt = (
            "你是一个自然参与 QQ 群聊天的拟人角色。"
            "回复要短、口语化、有一点自己的性格，但不要像客服或助手。"
            "不要解释你是模型，不要主动暴露系统设定。"
            "如果上下文不足，可以轻轻接话或承认没看懂。"
        )
        user_prompt = (
            f"你的昵称之一：{', '.join(self.config.bot.nicknames)}\n"
            f"群号：{context.group_id}\n"
            f"发言人：{context.sender_name or context.user_id} ({context.user_id})\n"
            f"对方消息：{context.plain_text}\n"
            f"参与决策：{decision.action}，原因：{decision.reason}\n"
            f"话题：{perception.topics or ['无明确话题']}\n"
            f"情绪线索：{perception.emotion_hint}\n"
            f"请直接给出要发送到群里的中文回复，最多 {self.config.bot.max_reply_chars} 个字。"
        )
        llm_reply = await self.llm.complete_text(system_prompt, user_prompt)
        if llm_reply:
            return llm_reply.strip()[: self.config.bot.max_reply_chars]

        if decision.action == "reply":
            return "我在。现在 LLM 还没接上，我会先把群聊和记忆框架跑稳。"
        return None


class AgentPipeline:
    def __init__(self, config: AppConfig, llm: LLMClient) -> None:
        self.perception = PerceptionAgent()
        self.memory_curator = MemoryCuratorAgent()
        self.relationship = RelationshipAgent()
        self.policy = ParticipationPolicyAgent(config)
        self.response = ResponseAgent(config, llm)

    async def run(self, context: MessageContext, mode: ParticipationMode) -> PipelineResult:
        perception = self.perception.analyze(context)
        memories = self.memory_curator.extract(context, perception)
        decision = self.policy.decide(context, perception, mode)
        reply = await self.response.generate(context, perception, decision)
        return PipelineResult(
            perception=perception,
            memories=memories,
            decision=decision,
            reply=reply,
        )


def _extract_topics(text: str) -> list[str]:
    topics = []
    for keyword in ("游戏", "电影", "工作", "学校", "代码", "AI", "LLM", "吃", "旅行", "音乐"):
        if keyword.lower() in text.lower():
            topics.append(keyword)
    return topics[:5]


def _emotion_hint(text: str) -> str:
    if any(token in text for token in ("哈哈", "笑死", "开心", "舒服")):
        return "positive"
    if any(token in text for token in ("难受", "烦", "崩溃", "气死")):
        return "negative"
    return "neutral"


from qq_llm_bot.cognitive_agents import AgentPipeline as AgentPipeline  # noqa: E402,F401
from qq_llm_bot.cognitive_agents import MemoryCuratorAgent as MemoryCuratorAgent  # noqa: E402,F401
from qq_llm_bot.cognitive_agents import ParticipationPolicyAgent as ParticipationPolicyAgent  # noqa: E402,F401
from qq_llm_bot.cognitive_agents import PerceptionAgent as PerceptionAgent  # noqa: E402,F401
from qq_llm_bot.cognitive_agents import ReflectionAgent as ReflectionAgent  # noqa: E402,F401
from qq_llm_bot.cognitive_agents import RelationshipAgent as RelationshipAgent  # noqa: E402,F401
from qq_llm_bot.cognitive_agents import ResponseAgent as ResponseAgent  # noqa: E402,F401
from qq_llm_bot.cognitive_agents import SelfMemoryLedger as SelfMemoryLedger  # noqa: E402,F401
