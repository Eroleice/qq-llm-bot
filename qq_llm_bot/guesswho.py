from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import Any, Callable

from qq_llm_bot.knowledge_models import FactRecord, MemoryRecord, UserProfileRecord
from qq_llm_bot.llm_json_helpers import complete_json
from qq_llm_bot.pipeline_models import RelationshipState
from qq_llm_bot.storage_record_identity import _dashboard_user_id


GUESSWHO_FAMILIARITY = 100
GUESSWHO_FACT_LIMIT = 30
GUESSWHO_MEMORY_LIMIT = 12
GUESSWHO_ACTIVE_REPLY = "上一轮猜人还没揭晓呢，先用 #bot tellmewho 看答案吧。"
GUESSWHO_NO_GAME_REPLY = "现在没有正在进行的猜人游戏。"
GUESSWHO_NO_CANDIDATE_REPLY = "当前群里还没有熟悉度 100 且可出题的成员。"
GUESSWHO_LLM_EMPTY_REPLY = "可可现在整理不出题目，稍后再试试吧。"

IDENTITY_FACT_TYPES = {"identity", "alias"}
IDENTITY_MEMORY_KINDS = {"identity", "alias"}


@dataclass(frozen=True)
class GuessWhoGame:
    group_id: str
    answer_user_id: str
    clue: str


_active_games: dict[str, GuessWhoGame] = {}


async def start_guesswho_game(
    storage: Any,
    llm: Any,
    group_id: str,
    current_member_user_ids: list[str],
    *,
    chooser: Callable[[list[str]], str] | None = None,
) -> str:
    group_key = str(group_id)
    if group_key in _active_games:
        return GUESSWHO_ACTIVE_REPLY

    candidates = list_guesswho_candidates(storage, current_member_user_ids)
    if not candidates:
        return GUESSWHO_NO_CANDIDATE_REPLY

    chooser = chooser or random.choice
    answer_user_id = chooser(candidates)
    clue = await build_guesswho_clue(storage, llm, answer_user_id)
    if not clue:
        return GUESSWHO_LLM_EMPTY_REPLY

    _active_games[group_key] = GuessWhoGame(
        group_id=group_key,
        answer_user_id=answer_user_id,
        clue=clue,
    )
    return f"猜猜这是谁：\n{clue}\n\n用 #bot tellmewho 揭晓答案。"


def finish_guesswho_game(group_id: str) -> GuessWhoGame | None:
    return _active_games.pop(str(group_id), None)


def active_guesswho_game(group_id: str) -> GuessWhoGame | None:
    return _active_games.get(str(group_id))


def clear_guesswho_games() -> None:
    _active_games.clear()


def list_guesswho_candidates(storage: Any, current_member_user_ids: list[str]) -> list[str]:
    current_members = {
        canonical
        for user_id in current_member_user_ids
        if (canonical := _dashboard_user_id(user_id)).isdigit()
    }
    if not current_members:
        return []
    familiar_user_ids = set(storage.list_familiar_user_ids(min_familiarity=GUESSWHO_FAMILIARITY))
    return sorted(current_members & familiar_user_ids)


async def build_guesswho_clue(storage: Any, llm: Any, user_id: str) -> str:
    facts = _clue_facts(storage, user_id)
    profile = _get_user_profile(storage, user_id)
    memories = _clue_memories(storage, user_id)
    relationship = _get_relationship(storage, user_id)
    data = await complete_json(
        llm,
        _guesswho_system_prompt(),
        _guesswho_user_prompt(
            user_id=user_id,
            facts=facts,
            profile=profile,
            memories=memories,
            relationship=relationship,
        ),
        purpose="guesswho",
    )
    if not data:
        return ""
    return _format_clue(data, facts, user_id)


def _clue_facts(storage: Any, user_id: str) -> list[FactRecord]:
    facts = storage.list_user_facts(
        user_id,
        limit=GUESSWHO_FACT_LIMIT,
        status="accepted",
        include_faded=True,
    )
    return [fact for fact in facts if fact.fact_type not in IDENTITY_FACT_TYPES]


def _get_user_profile(storage: Any, user_id: str) -> UserProfileRecord | None:
    getter = getattr(storage, "get_user_profile", None)
    return getter(user_id) if callable(getter) else None


def _clue_memories(storage: Any, user_id: str) -> list[MemoryRecord]:
    lister = getattr(storage, "list_memories", None)
    if not callable(lister):
        return []
    memories = lister("user", user_id, limit=GUESSWHO_MEMORY_LIMIT, status="active")
    return [memory for memory in memories if memory.kind not in IDENTITY_MEMORY_KINDS]


def _get_relationship(storage: Any, user_id: str) -> RelationshipState | None:
    getter = getattr(storage, "get_relationship", None)
    return getter("", user_id) if callable(getter) else None


def _guesswho_system_prompt() -> str:
    return (
        "你是 QQ 群猜人小游戏的出题助手。只输出 JSON，不要解释。"
        "你必须保护答案身份，不透露任何名字、昵称、群称呼、QQ号、头像或直接身份线索。"
    )


def _guesswho_user_prompt(
    *,
    user_id: str,
    facts: list[FactRecord],
    profile: UserProfileRecord | None,
    memories: list[MemoryRecord],
    relationship: RelationshipState | None,
) -> str:
    return (
        "请根据下方资料生成猜人题。\n"
        "输出 JSON：{\"intro\":\"40-80字介绍\",\"facts\":[\"脱敏FACT1\",\"脱敏FACT2\",\"脱敏FACT3\"]}\n"
        "要求：\n"
        "- intro 控制在 40-80 字。\n"
        "- facts 必须是 3 条，短句即可。\n"
        "- 不要出现任何称呼类提示，包括名字、昵称、别名、QQ号、群头衔、亲属/关系称呼。\n"
        "- 不要出现“用户42”“QQ:42”这类编号提示。\n"
        "- 不要直接说这是男/女、管理员、群主等容易定位的信息。\n"
        "- 只能使用资料支持的信息，不要编造。\n\n"
        f"资料主语编号（仅用于脱敏，禁止输出）：{user_id}\n"
        f"画像：\n{_format_profile(profile, user_id)}\n\n"
        f"关系认知：\n{_format_relationship(relationship, user_id)}\n\n"
        f"记忆：\n{_format_memories(memories, user_id)}\n\n"
        f"FACT：\n{_format_facts(facts, user_id)}"
    )


def _format_profile(profile: UserProfileRecord | None, user_id: str) -> str:
    if profile is None:
        return "(none)"
    return _desensitize(profile.summary, user_id)


def _format_relationship(relationship: RelationshipState | None, user_id: str) -> str:
    if relationship is None:
        return "(none)"
    parts = [
        f"closeness={relationship.closeness}",
        f"trust={relationship.trust}",
        f"familiarity={relationship.familiarity}",
    ]
    if relationship.summary:
        parts.append("summary=" + _desensitize(relationship.summary, user_id))
    return ", ".join(parts)


def _format_memories(memories: list[MemoryRecord], user_id: str) -> str:
    if not memories:
        return "(none)"
    return "\n".join(
        f"- [{memory.kind}] {_desensitize(memory.content, user_id)}"
        for memory in memories
        if _desensitize(memory.content, user_id)
    ) or "(none)"


def _format_facts(facts: list[FactRecord], user_id: str) -> str:
    if not facts:
        return "(none)"
    return "\n".join(_format_fact(fact, user_id) for fact in facts)


def _format_fact(fact: FactRecord, user_id: str) -> str:
    pieces = [_desensitize(fact.claim_text, user_id)]
    if fact.topic:
        pieces.append(f"topic={_desensitize(fact.topic, user_id)}")
    if fact.stance:
        pieces.append(f"stance={fact.stance}")
    return "- " + "；".join(piece for piece in pieces if piece)


def _format_clue(data: dict[str, Any], fallback_facts: list[FactRecord], user_id: str) -> str:
    intro = _clean_clue_text(data.get("intro", ""), user_id)
    facts = _clean_clue_facts(data.get("facts"), user_id)
    for fact in fallback_facts:
        if len(facts) >= 3:
            break
        fallback = _clean_clue_text(fact.claim_text, user_id)
        if fallback and fallback not in facts:
            facts.append(fallback)
    if not intro and not facts:
        return ""
    lines = []
    if intro:
        lines.append(f"介绍：{intro}")
    lines.extend(f"{index}. {fact}" for index, fact in enumerate(facts[:3], start=1))
    return "\n".join(lines)


def _clean_clue_facts(value: Any, user_id: str) -> list[str]:
    if not isinstance(value, list):
        return []
    facts: list[str] = []
    for item in value:
        text = _clean_clue_text(item, user_id)
        if text and text not in facts:
            facts.append(text)
        if len(facts) >= 3:
            break
    return facts


def _clean_clue_text(value: Any, user_id: str) -> str:
    text = _desensitize(str(value or ""), user_id)
    text = re.sub(r"^\s*(?:[-*]|\d+[.、])\s*", "", text).strip()
    return " ".join(text.split())


def _desensitize(value: str, user_id: str) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    escaped = re.escape(str(user_id))
    replacements = [
        rf"(?i)QQ[:：]?\s*{escaped}",
        rf"用户\s*{escaped}",
        rf"群友\s*{escaped}",
        escaped,
    ]
    for pattern in replacements:
        text = re.sub(pattern, "这个人", text)
    return text.strip()
