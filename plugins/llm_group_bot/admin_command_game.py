from __future__ import annotations

from typing import Any

from loguru import logger
from nonebot.adapters.onebot.v11 import Bot, GroupMessageEvent, Message, MessageSegment

from plugins.llm_group_bot import admin_command_state as _state
from qq_llm_bot.guesswho import (
    GUESSWHO_ACTIVE_REPLY,
    GUESSWHO_NO_GAME_REPLY,
    GUESSWHO_USAGE_REPLY,
    active_guesswho_game,
    finish_guesswho_game,
    format_guesswho_ranking,
    request_guesswho_hint,
    reveal_guesswho_answer,
    start_guesswho_game,
    submit_guesswho_guess,
)
from qq_llm_bot.outbound_queue_errors import onebot_group_id


async def handle_guess_command(
    bot: Bot,
    event: GroupMessageEvent,
    args: Message,
) -> None:
    user_id = str(event.user_id)
    group_id = str(event.group_id)
    if _state.storage.is_user_ignored(user_id):
        return

    action = " ".join(args.extract_plain_text().split()).lower()
    if action == "end":
        await _force_end_game(group_id, user_id)
        return
    if action == "who":
        await _start_game(bot, group_id)
        return
    if action == "hint":
        await _send_hint(group_id)
        return
    if action == "answer":
        await _reveal_answer(group_id)
        return
    if action == "rank":
        await _send_rank(bot, group_id, wrong=False)
        return
    if action == "rank wrong":
        await _send_rank(bot, group_id, wrong=True)
        return

    mentioned_user_ids = _mentioned_user_ids(args)
    if not action and len(mentioned_user_ids) == 1:
        await _submit_guess(group_id, user_id, mentioned_user_ids[0])
        return
    await _state.finish_command(_state.guess_cmd, GUESSWHO_USAGE_REPLY)


async def _force_end_game(group_id: str, user_id: str) -> None:
    if not _state.storage.is_admin(user_id):
        await _state.finish_command(_state.guess_cmd, "权限不足。")
        return
    if finish_guesswho_game(group_id) is None:
        await _state.finish_command(_state.guess_cmd, GUESSWHO_NO_GAME_REPLY)
        return
    await _state.finish_command(_state.guess_cmd, "本轮猜人游戏已由管理员强制结束。")


async def _start_game(bot: Bot, group_id: str) -> None:
    if active_guesswho_game(group_id) is not None:
        await _state.finish_command(_state.guess_cmd, GUESSWHO_ACTIVE_REPLY)
        return
    member_user_ids = await _current_group_member_user_ids(bot, group_id)
    if member_user_ids is None:
        await _state.finish_command(_state.guess_cmd, "获取当前群成员列表失败，稍后再试试。")
        return
    reply = await start_guesswho_game(
        _state.storage,
        _state.llm,
        group_id,
        member_user_ids,
    )
    await _state.finish_command(_state.guess_cmd, reply)


async def _send_hint(group_id: str) -> None:
    result = request_guesswho_hint(group_id)
    if result.status == "no_game":
        await _state.finish_command(_state.guess_cmd, GUESSWHO_NO_GAME_REPLY)
        return
    if result.status == "cooldown":
        await _state.finish_command(
            _state.guess_cmd,
            f"还不能获取提示，请再等 {_format_wait(result.remaining_seconds)}。",
        )
        return
    if result.status == "exhausted":
        await _state.finish_command(_state.guess_cmd, "本局两次提示已经用完了。")
        return
    if result.status != "ok":
        await _state.finish_command(_state.guess_cmd, "这局暂时整理不出更多提示。")
        return

    start_index = 4 if result.hint_number == 1 else 7
    lines = [f"更多提示（{result.hint_number}/2）："]
    lines.extend(
        f"{index}. {fact}"
        for index, fact in enumerate(result.facts, start=start_index)
    )
    await _state.finish_command(_state.guess_cmd, "\n".join(lines))


async def _submit_guess(group_id: str, user_id: str, guessed_user_id: str) -> None:
    result = submit_guesswho_guess(group_id, guessed_user_id)
    if result.status == "no_game":
        await _state.finish_command(_state.guess_cmd, GUESSWHO_NO_GAME_REPLY)
        return
    if result.status == "incorrect":
        _state.storage.record_guesswho_result(group_id, user_id, correct=False)
        await _state.finish_command(_state.guess_cmd, "猜错啦，游戏继续。")
        return

    _state.storage.record_guesswho_result(group_id, user_id, correct=True)
    game = result.game
    if game is None:  # pragma: no cover - result invariant
        await _state.finish_command(_state.guess_cmd, GUESSWHO_NO_GAME_REPLY)
        return
    message = Message()
    message += MessageSegment.text("答对了！答案就是：")
    message += MessageSegment.at(game.answer_user_id)
    await _state.finish_command(_state.guess_cmd, message)


async def _send_rank(bot: Bot, group_id: str, *, wrong: bool) -> None:
    scores = _state.storage.list_guesswho_scores(group_id, wrong=wrong, limit=10)
    members = await _current_group_members(bot, group_id)
    member_names = _group_member_names(members or [])
    await _state.finish_command(
        _state.guess_cmd,
        format_guesswho_ranking(scores, wrong=wrong, member_names=member_names),
    )


async def _reveal_answer(group_id: str) -> None:
    result = reveal_guesswho_answer(group_id)
    if result.status == "no_game":
        await _state.finish_command(_state.guess_cmd, GUESSWHO_NO_GAME_REPLY)
        return
    if result.status == "cooldown":
        await _state.finish_command(
            _state.guess_cmd,
            f"开局 5 分钟后才能揭晓答案，请再等 {_format_wait(result.remaining_seconds)}。",
        )
        return

    game = result.game
    if game is None:  # pragma: no cover - result invariant
        await _state.finish_command(_state.guess_cmd, GUESSWHO_NO_GAME_REPLY)
        return
    message = Message()
    message += MessageSegment.text("答案是：")
    message += MessageSegment.at(game.answer_user_id)
    await _state.finish_command(_state.guess_cmd, message)


def _mentioned_user_ids(message: Message) -> list[str]:
    user_ids: list[str] = []
    for segment in message:
        if segment.type != "at":
            continue
        user_id = str(segment.data.get("qq", "") or "").strip()
        if user_id and user_id != "all":
            user_ids.append(user_id)
    return user_ids


def _format_wait(seconds: int) -> str:
    minutes, remaining_seconds = divmod(max(1, seconds), 60)
    if minutes and remaining_seconds:
        return f"{minutes} 分 {remaining_seconds} 秒"
    if minutes:
        return f"{minutes} 分钟"
    return f"{remaining_seconds} 秒"


async def _current_group_member_user_ids(bot: Bot, group_id: str) -> list[str] | None:
    members = await _current_group_members(bot, group_id)
    if members is None:
        return None
    self_id = str(getattr(bot, "self_id", "") or "").strip()
    return [
        user_id
        for member in members
        if (user_id := _member_user_id(member)) and user_id != self_id
    ]


async def _current_group_members(bot: Bot, group_id: str) -> list[Any] | None:
    onebot_id = onebot_group_id(group_id)
    if onebot_id is None:
        return []
    try:
        try:
            members = await bot.get_group_member_list(group_id=onebot_id, no_cache=True)
        except TypeError:
            members = await bot.get_group_member_list(group_id=onebot_id)
    except Exception as exc:
        logger.warning("Guesswho failed to fetch group member list for group {}: {}", group_id, exc)
        return None
    return list(members)


def _group_member_names(members: list[Any]) -> dict[str, str]:
    names: dict[str, str] = {}
    for member in members:
        user_id = _member_user_id(member)
        name = _member_display_name(member)
        if user_id and name:
            names[user_id] = name
    return names


def _member_user_id(member: Any) -> str:
    if isinstance(member, dict):
        value = member.get("user_id", "")
    else:
        value = getattr(member, "user_id", "")
    return str(value or "").strip()


def _member_display_name(member: Any) -> str:
    if isinstance(member, dict):
        value = member.get("card") or member.get("nickname") or ""
    else:
        value = getattr(member, "card", "") or getattr(member, "nickname", "")
    return " ".join(str(value or "").split())
