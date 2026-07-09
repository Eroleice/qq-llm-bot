from __future__ import annotations

from typing import Any

from loguru import logger
from nonebot.adapters.onebot.v11 import Bot, Message, MessageSegment

from plugins.llm_group_bot import admin_command_state as _state
from qq_llm_bot.guesswho import (
    GUESSWHO_ACTIVE_REPLY,
    GUESSWHO_NO_GAME_REPLY,
    active_guesswho_game,
    finish_guesswho_game,
    start_guesswho_game,
)
from qq_llm_bot.outbound_queue_errors import onebot_group_id


async def handle_guesswho(bot: Bot, group_id: str) -> None:
    if active_guesswho_game(group_id) is not None:
        await _state.finish_command(_state.admin_cmd, GUESSWHO_ACTIVE_REPLY)
        return
    member_user_ids = await _current_group_member_user_ids(bot, group_id)
    if member_user_ids is None:
        await _state.finish_command(_state.admin_cmd, "获取当前群成员列表失败，稍后再试试。")
        return
    reply = await start_guesswho_game(
        _state.storage,
        _state.llm,
        group_id,
        member_user_ids,
    )
    await _state.finish_command(_state.admin_cmd, reply)


async def handle_tellmewho(group_id: str) -> None:
    game = finish_guesswho_game(group_id)
    if game is None:
        await _state.finish_command(_state.admin_cmd, GUESSWHO_NO_GAME_REPLY)
        return
    message = Message()
    message += MessageSegment.text("答案是：")
    message += MessageSegment.at(game.answer_user_id)
    await _state.finish_command(_state.admin_cmd, message)


async def _current_group_member_user_ids(bot: Bot, group_id: str) -> list[str] | None:
    onebot_id = onebot_group_id(group_id)
    if onebot_id is None:
        return []
    try:
        members = await bot.get_group_member_list(group_id=onebot_id, no_cache=True)
    except TypeError:
        members = await bot.get_group_member_list(group_id=onebot_id)
    except Exception as exc:
        logger.warning("Guesswho failed to fetch group member list for group {}: {}", group_id, exc)
        return None
    self_id = str(getattr(bot, "self_id", "") or "").strip()
    user_ids = []
    for member in members:
        user_id = _member_user_id(member)
        if user_id and user_id != self_id:
            user_ids.append(user_id)
    return user_ids


def _member_user_id(member: Any) -> str:
    if isinstance(member, dict):
        value = member.get("user_id", "")
    else:
        value = getattr(member, "user_id", "")
    return str(value or "").strip()
