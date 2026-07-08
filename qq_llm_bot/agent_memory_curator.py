from __future__ import annotations

import re
from dataclasses import replace

from qq_llm_bot.agent_common import clamp_float as _clamp_float
from qq_llm_bot.agent_fact_helpers import (
    extract_mention_claim as _extract_mention_claim,
    member_mentions as _member_mentions,
    owner_id_for as _owner_id_for,
    safe_claim_scope as _safe_claim_scope,
    subject_for as _subject_for,
)
from qq_llm_bot.agent_formatters import (
    format_memories as _format_memories,
    format_mentions as _format_mentions,
)
from qq_llm_bot.llm import LLMClient
from qq_llm_bot.llm_json_helpers import complete_json as _complete_json
from qq_llm_bot.models import (
    ConversationSnapshot,
    MemoryCandidate,
    MessageContext,
    PerceptionResult,
)
from qq_llm_bot.onebot_messages import strip_forwarded_records, strip_quoted_messages


class MemoryCuratorAgent:
    SELF_DISCLOSURE_PATTERNS = (
        ("alias", re.compile(r"我叫\s*([^\s，。,.!！?？]{1,16})")),
        ("identity", re.compile(r"我是\s*([^，。,.!！?？]{1,32})")),
        ("preference", re.compile(r"我喜欢\s*([^，。,.!！?？]{1,32})")),
        ("dislike", re.compile(r"我讨厌\s*([^，。,.!！?？]{1,32})")),
        ("location", re.compile(r"我住(?:在)?\s*([^，。,.!！?？]{1,32})")),
        ("experience", re.compile(r"我最近\s*([^，。,.!！?？]{1,40})")),
    )

    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    async def extract(
        self,
        context: MessageContext,
        perception: PerceptionResult,
        snapshot: ConversationSnapshot,
    ) -> list[MemoryCandidate]:
        fallback = self._heuristic(context, perception)
        data = await _complete_json(
            self.llm,
            "你是保守的群聊记忆整理器。只输出 JSON，不要解释。",
            (
                "从单条消息中抽取适合长期记住的事实。"
                "只记录明确表达的稳定事实，不要猜测。"
                "必须区分本人自述和第三方转述。"
                "输出 JSON："
                '{"memories":[{"owner_type":"user|self|group","owner_id":"QQ或name:称呼或group",'
                '"subject_user_id":"QQ或name:称呼或bot或group",'
                '"claim_scope":"self_report|third_party|bot_directed|group_fact",'
                '"kind":"alias|identity|preference|'
                'dislike|location|experience|persona_fact","content":"...","confidence":0.0,'
                '"importance":0.0}]}\n'
                "例子：我喜欢吃鱼 -> self_report，subject 是说话人。"
                "可可，我喜欢吃鱼 -> self_report，subject 仍是说话人。"
                "小明喜欢吃鱼 -> third_party，subject 是 name:小明。"
                "大家都喜欢吃鱼 -> group_fact。\n"
                f"说话人 QQ：{context.user_id}\n"
                "Mention rule: when this message says this person/he/she/ta/@someone and a QQ mention "
                "clearly identifies that member, use the mentioned QQ as subject_user_id.\n"
                "Forwarded record rule: text between [合并转发聊天记录开始] and "
                "[合并转发聊天记录结束] is quoted chat history. First-person words inside it "
                "refer to the sender label on that forwarded line, not the member who forwarded it.\n"
                "Quoted message rule: text between [被引用消息开始] and [被引用消息结束] "
                "is quoted context. First-person words inside it refer to the quoted sender label, "
                "not the current speaker.\n"
                f"Current mentions:\n{_format_mentions(context)}\n"
                f"说话人已有记忆：\n{_format_memories(snapshot.user_memories)}\n"
                f"消息：{context.plain_text}"
            ),
            purpose="memory_curator",
        )
        if not data:
            return fallback
        memories = []
        for item in data.get("memories", []):
            if not isinstance(item, dict):
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
            content = str(item.get("content", "")).strip()
            kind = str(item.get("kind", "experience")).strip() or "experience"
            if content:
                memories.append(
                    MemoryCandidate(
                        owner_type=owner_type,  # type: ignore[arg-type]
                        owner_id=owner_id,
                        kind=kind,
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
        return memories or fallback

    def _heuristic(
        self,
        context: MessageContext,
        perception: PerceptionResult,
    ) -> list[MemoryCandidate]:
        direct_text = strip_quoted_messages(strip_forwarded_records(context.plain_text))
        direct_context = replace(context, plain_text=direct_text)
        memories: list[MemoryCandidate] = []
        memories.extend(self._heuristic_group_facts(direct_context))
        memories.extend(self._heuristic_third_party(direct_context))
        memories.extend(self._heuristic_mentioned_member(direct_context))
        if not perception.is_self_disclosure:
            return memories
        for kind, pattern in self.SELF_DISCLOSURE_PATTERNS:
            for match in pattern.finditer(direct_context.plain_text):
                content = match.group(1).strip()
                if content:
                    memories.append(
                        MemoryCandidate(
                            owner_type="user",
                            owner_id=context.user_id,
                            kind=kind,
                            content=content,
                            confidence=0.76,
                            importance=0.55,
                            evidence_message_id=context.message_id,
                            source_text=direct_context.plain_text,
                            source_user_id=context.user_id,
                            source_group_id=context.group_id,
                            subject_user_id=context.user_id,
                            claim_scope="self_report",
                        )
                    )
        return memories

    def _heuristic_group_facts(self, context: MessageContext) -> list[MemoryCandidate]:
        text = context.plain_text.strip()
        memories: list[MemoryCandidate] = []
        match = re.search(r"(?:大家|我们群|群里).{0,4}(?:都|一般)?喜欢\s*([^，。,.!！?？]{1,32})", text)
        if match:
            memories.append(
                MemoryCandidate(
                    owner_type="group",
                    owner_id=context.group_id,
                    kind="preference",
                    content=match.group(1).strip(),
                    confidence=0.76,
                    importance=0.45,
                    evidence_message_id=context.message_id,
                    source_text=text,
                    source_user_id=context.user_id,
                    source_group_id=context.group_id,
                    subject_user_id=context.group_id,
                    claim_scope="group_fact",
                )
            )
        return memories

    def _heuristic_mentioned_member(self, context: MessageContext) -> list[MemoryCandidate]:
        memories: list[MemoryCandidate] = []
        for mention in _member_mentions(context):
            extracted = _extract_mention_claim(context.plain_text, mention)
            if extracted is None:
                continue
            kind, content = extracted
            memories.append(
                MemoryCandidate(
                    owner_type="user",
                    owner_id=mention.user_id,
                    kind=kind,
                    content=content,
                    confidence=0.78,
                    importance=0.5,
                    evidence_message_id=context.message_id,
                    source_text=context.plain_text,
                    source_user_id=context.user_id,
                    source_group_id=context.group_id,
                    subject_user_id=mention.user_id,
                    claim_scope="third_party",
                )
            )
        return memories

    def _heuristic_third_party(self, context: MessageContext) -> list[MemoryCandidate]:
        text = context.plain_text.strip()
        memories: list[MemoryCandidate] = []
        pattern = re.compile(r"([^\s，。,.!！?？我大家群里]{1,12})喜欢\s*([^，。,.!！?？]{1,32})")
        for match in pattern.finditer(text):
            subject = match.group(1).strip()
            content = match.group(2).strip()
            if not subject or subject in {"可可", "机器人", "大家", "群里", "我们"}:
                continue
            memories.append(
                MemoryCandidate(
                    owner_type="user",
                    owner_id=f"name:{subject}",
                    kind="preference",
                    content=content,
                    confidence=0.78,
                    importance=0.45,
                    evidence_message_id=context.message_id,
                    source_text=text,
                    source_user_id=context.user_id,
                    source_group_id=context.group_id,
                    subject_user_id=f"name:{subject}",
                    claim_scope="third_party",
                )
            )
        return memories
