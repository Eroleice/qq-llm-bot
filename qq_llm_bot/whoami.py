from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from qq_llm_bot.knowledge_models import FactRecord, MemoryRecord, UserProfileRecord

if TYPE_CHECKING:
    from qq_llm_bot.llm_models import LLMClient
    from qq_llm_bot.pipeline_models import RelationshipState


WHOAMI_MIN_FACTS = 5
WHOAMI_FACT_LIMIT = 40
WHOAMI_NOT_ENOUGH_FACTS_REPLY = "可可和你还不是很熟悉呢，再多聊聊天吧！"
WHOAMI_LLM_EMPTY_REPLY = "可可现在有点整理不出来，等会儿再问问我吧！"


async def build_whoami_reply(
    storage: Any,
    llm: LLMClient,
    user_id: str,
    *,
    group_id: str = "",
    display_name: str = "",
) -> str:
    facts = _list_user_facts(storage, user_id)
    if len(facts) < WHOAMI_MIN_FACTS:
        return WHOAMI_NOT_ENOUGH_FACTS_REPLY

    profile = _get_user_profile(storage, user_id)
    memories = _list_user_memories(storage, user_id)
    relationship = _get_relationship(storage, group_id, user_id)
    summary = await llm.complete_text(
        _whoami_system_prompt(),
        _whoami_user_prompt(
            user_id=user_id,
            display_name=display_name,
            facts=facts,
            memories=memories,
            profile=profile,
            relationship=relationship,
        ),
        purpose="whoami",
    )
    return str(summary or "").strip() or WHOAMI_LLM_EMPTY_REPLY


def _list_user_facts(storage: Any, user_id: str) -> list[FactRecord]:
    return list(
        storage.list_user_facts(
            user_id,
            limit=WHOAMI_FACT_LIMIT,
            status="accepted",
            include_faded=True,
        )
    )


def _get_user_profile(storage: Any, user_id: str) -> UserProfileRecord | None:
    getter = getattr(storage, "get_user_profile", None)
    if not callable(getter):
        return None
    return getter(user_id)


def _list_user_memories(storage: Any, user_id: str) -> list[MemoryRecord]:
    lister = getattr(storage, "list_memories", None)
    if not callable(lister):
        return []
    return list(lister("user", user_id, limit=12, status="active"))


def _get_relationship(storage: Any, group_id: str, user_id: str) -> RelationshipState | None:
    getter = getattr(storage, "get_relationship", None)
    if not callable(getter):
        return None
    return getter(group_id, user_id)


def _whoami_system_prompt() -> str:
    return (
        "你是 QQ 群机器人“可可”的用户认知总结助手。"
        "只输出一段中文总结，不要解释，不要编号。"
    )


def _whoami_user_prompt(
    *,
    user_id: str,
    display_name: str,
    facts: list[FactRecord],
    memories: list[MemoryRecord],
    profile: UserProfileRecord | None,
    relationship: RelationshipState | None,
) -> str:
    user_label = f"{display_name}(QQ:{user_id})" if display_name else f"QQ:{user_id}"
    return (
        "请根据可可对这位发言用户的记忆认知，整理一份不超过 80 字的总结。\n"
        "要求：\n"
        "- 优先用“你”称呼对方，语气温和自然。\n"
        "- 只能使用下方资料支持的信息，不要编造。\n"
        "- 不要提到 FACT、id、置信度、画像或关系分。\n"
        "- 不要列表，不要加标题，不要写“总结：”。\n"
        "- 字数控制交给你，不要超过 80 字。\n\n"
        f"用户：{user_label}\n\n"
        f"已有画像：\n{_format_profile(profile)}\n\n"
        f"关系认知：\n{_format_relationship(relationship)}\n\n"
        f"用户记忆：\n{_format_memories(memories)}\n\n"
        f"accepted FACT：\n{_format_facts(facts)}"
    )


def _format_profile(profile: UserProfileRecord | None) -> str:
    if profile is None:
        return "(none)"
    traits = json.dumps(profile.traits, ensure_ascii=False) if profile.traits else "{}"
    return f"{profile.summary}\ntraits={traits}"


def _format_relationship(relationship: RelationshipState | None) -> str:
    if relationship is None:
        return "(none)"
    pieces = [
        f"closeness={relationship.closeness}",
        f"trust={relationship.trust}",
        f"familiarity={relationship.familiarity}",
        f"tension={relationship.tension}",
    ]
    if relationship.summary:
        pieces.append(f"summary={relationship.summary}")
    return ", ".join(pieces)


def _format_memories(memories: list[MemoryRecord]) -> str:
    if not memories:
        return "(none)"
    return "\n".join(f"- [{memory.kind}] {memory.content}" for memory in memories)


def _format_facts(facts: list[FactRecord]) -> str:
    if not facts:
        return "(none)"
    return "\n".join(_format_fact(fact) for fact in facts)


def _format_fact(fact: FactRecord) -> str:
    details = []
    if fact.fact_type:
        details.append(f"type={fact.fact_type}")
    if fact.topic:
        details.append(f"topic={fact.topic}")
    if fact.stance:
        details.append(f"stance={fact.stance}")
    suffix = f" ({', '.join(details)})" if details else ""
    return f"- {fact.claim_text}{suffix}"
