from __future__ import annotations

from plugins.llm_group_bot import admin_command_state as _state
from plugins.llm_group_bot import admin_command_utils as _utils


async def handle_whitelist(rest: list[str]) -> None:
    if not rest or rest[0] == "list":
        groups = _state.storage.list_enabled_groups()
        await _state.finish_command(_state.admin_cmd, "已启用群：" + (", ".join(groups) if groups else "(none)"))

    action = rest[0].lower()
    if len(rest) < 2 or action not in {"add", "remove"}:
        await _state.finish_command(_state.admin_cmd, "用法：#bot whitelist list|add <group_id>|remove <group_id>")

    group_id = rest[1]
    _state.storage.set_group_enabled(group_id, action == "add")
    await _state.finish_command(_state.admin_cmd, f"已{'启用' if action == 'add' else '停用'}群：{group_id}")

async def handle_admin(rest: list[str], current_user_id: str) -> None:
    if not rest or rest[0] == "list":
        await _state.finish_command(_state.admin_cmd, "管理员：" + (", ".join(_state.storage.list_admins()) or "(none)"))

    action = rest[0].lower()
    if len(rest) < 2 or action not in {"add", "remove"}:
        await _state.finish_command(_state.admin_cmd, "用法：#bot admin list|add <qq_id>|remove <qq_id>")

    target = rest[1]
    if action == "add":
        _state.storage.add_admin(target)
        await _state.finish_command(_state.admin_cmd, f"已添加管理员：{target}")

    if target == current_user_id:
        await _state.finish_command(_state.admin_cmd, "不能移除当前正在操作的管理员。")

    _state.storage.remove_admin(target)
    await _state.finish_command(_state.admin_cmd, f"已移除管理员：{target}")

async def handle_ignore(rest: list[str]) -> None:
    if not rest or rest[0] == "list":
        ignored_users = _state.storage.list_ignored_users()
        await _state.finish_command(
            _state.admin_cmd,
            "ignored users: " + (", ".join(ignored_users) if ignored_users else "(none)"),
        )

    action = rest[0].lower()
    if len(rest) < 2 or action not in {"add", "remove"}:
        await _state.finish_command(_state.admin_cmd, "Usage: #bot ignore list|add <qq_id>|remove <qq_id>")

    target = rest[1]
    if action == "add":
        _state.storage.add_ignored_user(target)
        await _state.finish_command(_state.admin_cmd, f"ignored user added: {target}")

    _state.storage.remove_ignored_user(target)
    await _state.finish_command(_state.admin_cmd, f"ignored user removed: {target}")

async def handle_profile(rest: list[str]) -> None:
    if not rest:
        await _state.finish_command(_state.admin_cmd, "用法：#bot profile <qq_id>")
    await _state.finish_command(_state.admin_cmd, _state.storage.format_user_profile(rest[0]))

async def handle_relation(rest: list[str], group_id: str) -> None:
    if not rest:
        await _state.finish_command(_state.admin_cmd, "用法：#bot relation <qq_id>|top [数量]|rank [数量]")
    action = rest[0].lower()
    if action in {"top", "rank", "ranking", "排行", "排行榜"}:
        limit = _utils.parse_memory_id(rest[1]) if len(rest) >= 2 else 5
        if limit is None:
            await _state.finish_command(_state.admin_cmd, "数量必须是数字，例如：#bot relation top 5")
        await _state.finish_command(_state.admin_cmd, _state.storage.format_relationship_ranking(group_id, limit))
    await _state.finish_command(_state.admin_cmd, _state.storage.format_relationship(group_id, rest[0]))

async def handle_forget(rest: list[str]) -> None:
    if not rest:
        await _state.finish_command(_state.admin_cmd, "用法：#bot forget <memory_id>")
    memory_id = _utils.parse_memory_id(rest[0])
    if memory_id is None:
        await _state.finish_command(_state.admin_cmd, "memory_id 必须是数字，例如：#bot forget 12")
    ok = _state.storage.forget_memory(memory_id)
    await _state.finish_command(_state.admin_cmd, "已遗忘。" if ok else "没有找到可遗忘的记忆。")
