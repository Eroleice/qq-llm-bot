from __future__ import annotations

import random
import re
import time
from dataclasses import dataclass, replace
from math import ceil
from typing import Any, Callable

from qq_llm_bot.knowledge_models import (
    FactRecord,
    GuessWhoScoreRecord,
    MemoryRecord,
    UserProfileRecord,
)
from qq_llm_bot.llm_json_helpers import complete_json
from qq_llm_bot.pipeline_models import RelationshipState
from qq_llm_bot.storage_record_identity import _dashboard_user_id


GUESSWHO_FAMILIARITY = 100
GUESSWHO_FACT_LIMIT = 30
GUESSWHO_MEMORY_LIMIT = 12
GUESSWHO_HINT_COOLDOWN_SECONDS = 2 * 60
GUESSWHO_ANSWER_COOLDOWN_SECONDS = 5 * 60
GUESSWHO_MAX_HINTS = 2
GUESSWHO_REQUIRED_FACTS = 3 * (1 + GUESSWHO_MAX_HINTS)
GUESSWHO_ACTIVE_REPLY = "上一轮猜人还没结束，答对或用 #guess answer 揭晓后才能开下一局。"
GUESSWHO_NO_GAME_REPLY = "现在没有正在进行的猜人游戏。"
GUESSWHO_NO_CANDIDATE_REPLY = "当前群里还没有熟悉度 100 且可出题的成员。"
GUESSWHO_LLM_EMPTY_REPLY = "可可现在整理不出题目，稍后再试试吧。"
GUESSWHO_USAGE_REPLY = (
    "用法：\n"
    "#guess who 开始游戏\n"
    "#guess hint 获取 3 条额外 FACT（最多两次）\n"
    "#guess @成员 提交答案\n"
    "#guess answer 揭晓答案\n"
    "#guess rank 查看猜对榜\n"
    "#guess rank wrong 查看猜错榜"
)

IDENTITY_FACT_TYPES = {"identity", "alias"}
IDENTITY_MEMORY_KINDS = {"identity", "alias"}


@dataclass(frozen=True)
class GuessWhoGame:
    group_id: str
    answer_user_id: str
    clue: str
    facts: tuple[str, ...] = ()
    started_at: float = 0.0
    hint_count: int = 0
    last_hint_at: float | None = None


@dataclass(frozen=True)
class GuessWhoHintResult:
    status: str
    facts: tuple[str, ...] = ()
    hint_number: int = 0
    remaining_seconds: int = 0


@dataclass(frozen=True)
class GuessWhoAnswerResult:
    status: str
    game: GuessWhoGame | None = None
    remaining_seconds: int = 0


@dataclass(frozen=True)
class GuessWhoGuessResult:
    status: str
    game: GuessWhoGame | None = None


@dataclass(frozen=True)
class _GuessWhoClueMaterial:
    intro: str
    facts: tuple[str, ...]


_active_games: dict[str, GuessWhoGame] = {}
_starting_groups: set[str] = set()


async def start_guesswho_game(
    storage: Any,
    llm: Any,
    group_id: str,
    current_member_user_ids: list[str],
    *,
    chooser: Callable[[list[str]], str] | None = None,
    now: float | None = None,
) -> str:
    group_key = str(group_id)
    if group_key in _active_games or group_key in _starting_groups:
        return GUESSWHO_ACTIVE_REPLY

    _starting_groups.add(group_key)
    try:
        candidates = list_guesswho_candidates(storage, current_member_user_ids)
        if not candidates:
            return GUESSWHO_NO_CANDIDATE_REPLY

        chooser = chooser or random.choice
        answer_user_id = chooser(candidates)
        material = await _build_guesswho_clue_material(storage, llm, answer_user_id)
        if (
            material is None
            or not material.intro
            or len(material.facts) < GUESSWHO_REQUIRED_FACTS
        ):
            return GUESSWHO_LLM_EMPTY_REPLY

        clue = _format_clue(material.intro, material.facts[:3])
        _active_games[group_key] = GuessWhoGame(
            group_id=group_key,
            answer_user_id=answer_user_id,
            clue=clue,
            facts=material.facts[:GUESSWHO_REQUIRED_FACTS],
            started_at=time.monotonic() if now is None else float(now),
        )
        return (
            f"猜猜这是谁：\n{clue}\n\n"
            "后续指令：\n"
            "#guess hint 获取 3 条额外 FACT（最多两次，开局 2 分钟后可用）\n"
            "#guess @成员 提交答案\n"
            "#guess answer 揭晓答案（开局 5 分钟后可用）\n"
            "#guess rank 查看猜对榜；#guess rank wrong 查看猜错榜"
        )
    finally:
        _starting_groups.discard(group_key)


def finish_guesswho_game(group_id: str) -> GuessWhoGame | None:
    return _active_games.pop(str(group_id), None)


def active_guesswho_game(group_id: str) -> GuessWhoGame | None:
    return _active_games.get(str(group_id))


def clear_guesswho_games() -> None:
    _active_games.clear()
    _starting_groups.clear()


def request_guesswho_hint(group_id: str, *, now: float | None = None) -> GuessWhoHintResult:
    group_key = str(group_id)
    game = _active_games.get(group_key)
    if game is None:
        return GuessWhoHintResult(status="no_game")
    if game.hint_count >= GUESSWHO_MAX_HINTS:
        return GuessWhoHintResult(status="exhausted")

    current_time = time.monotonic() if now is None else float(now)
    available_at = (
        game.started_at + GUESSWHO_HINT_COOLDOWN_SECONDS
        if game.last_hint_at is None
        else game.last_hint_at + GUESSWHO_HINT_COOLDOWN_SECONDS
    )
    if current_time < available_at:
        return GuessWhoHintResult(
            status="cooldown",
            remaining_seconds=max(1, ceil(available_at - current_time)),
        )

    hint_number = game.hint_count + 1
    fact_start = 3 * hint_number
    facts = game.facts[fact_start : fact_start + 3]
    if len(facts) < 3:  # pragma: no cover - start_game enforces enough facts
        return GuessWhoHintResult(status="unavailable")
    _active_games[group_key] = replace(
        game,
        hint_count=hint_number,
        last_hint_at=current_time,
    )
    return GuessWhoHintResult(status="ok", facts=facts, hint_number=hint_number)


def reveal_guesswho_answer(group_id: str, *, now: float | None = None) -> GuessWhoAnswerResult:
    group_key = str(group_id)
    game = _active_games.get(group_key)
    if game is None:
        return GuessWhoAnswerResult(status="no_game")

    current_time = time.monotonic() if now is None else float(now)
    available_at = game.started_at + GUESSWHO_ANSWER_COOLDOWN_SECONDS
    if current_time < available_at:
        return GuessWhoAnswerResult(
            status="cooldown",
            remaining_seconds=max(1, ceil(available_at - current_time)),
        )
    return GuessWhoAnswerResult(status="revealed", game=_active_games.pop(group_key))


def submit_guesswho_guess(group_id: str, guessed_user_id: str) -> GuessWhoGuessResult:
    group_key = str(group_id)
    game = _active_games.get(group_key)
    if game is None:
        return GuessWhoGuessResult(status="no_game")
    guessed_id = _dashboard_user_id(guessed_user_id)
    if guessed_id != game.answer_user_id:
        return GuessWhoGuessResult(status="incorrect", game=game)
    return GuessWhoGuessResult(status="correct", game=_active_games.pop(group_key))


def format_guesswho_ranking(
    scores: list[GuessWhoScoreRecord],
    *,
    wrong: bool = False,
    member_names: dict[str, str] | None = None,
) -> str:
    title = "猜错排行榜 TOP 10：" if wrong else "猜对排行榜 TOP 10："
    if not scores:
        return "猜错排行榜还没有记录。" if wrong else "猜对排行榜还没有记录。"
    names = member_names or {}
    lines = [title]
    ranked_scores = sorted(
        scores,
        key=lambda score: (
            -(score.wrong_count if wrong else score.correct_count),
            score.user_id,
        ),
    )
    for index, score in enumerate(ranked_scores[:10], start=1):
        nickname = " ".join(names.get(score.user_id, score.nickname).split()) or "QQ用户"
        count = score.wrong_count if wrong else score.correct_count
        lines.append(f"{index}. {nickname}（QQ:{score.user_id}）— {count} 次")
    return "\n".join(lines)


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
    material = await _build_guesswho_clue_material(storage, llm, user_id)
    if material is None:
        return ""
    return _format_clue(material.intro, material.facts[:3])


async def _build_guesswho_clue_material(
    storage: Any,
    llm: Any,
    user_id: str,
) -> _GuessWhoClueMaterial | None:
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
        return None
    return _parse_clue_material(data, facts, user_id)


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
        "输出 JSON：{\"intro\":\"40-80字介绍\",\"facts\":[\"脱敏FACT1\",...,\"脱敏FACT9\"]}\n"
        "要求：\n"
        "- intro 控制在 40-80 字。\n"
        "- facts 必须正好是 9 条互不重复的短句；前 3 条用于开局，后 6 条分两次作为提示。\n"
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


def _parse_clue_material(
    data: dict[str, Any],
    fallback_facts: list[FactRecord],
    user_id: str,
) -> _GuessWhoClueMaterial | None:
    intro = _clean_clue_text(data.get("intro", ""), user_id)
    facts = _clean_clue_facts(data.get("facts"), user_id, limit=GUESSWHO_REQUIRED_FACTS)
    for fact in fallback_facts:
        if len(facts) >= GUESSWHO_REQUIRED_FACTS:
            break
        fallback = _clean_clue_text(fact.claim_text, user_id)
        if fallback and fallback not in facts:
            facts.append(fallback)
    if not intro and not facts:
        return None
    return _GuessWhoClueMaterial(intro=intro, facts=tuple(facts))


def _format_clue(intro: str, facts: tuple[str, ...] | list[str], *, start_index: int = 1) -> str:
    lines = []
    if intro:
        lines.append(f"介绍：{intro}")
    lines.extend(f"{index}. {fact}" for index, fact in enumerate(facts, start=start_index))
    return "\n".join(lines)


def _clean_clue_facts(value: Any, user_id: str, *, limit: int = 3) -> list[str]:
    if not isinstance(value, list):
        return []
    facts: list[str] = []
    for item in value:
        text = _clean_clue_text(item, user_id)
        if text and text not in facts:
            facts.append(text)
        if len(facts) >= limit:
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
