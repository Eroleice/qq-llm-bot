from __future__ import annotations

from typing import Any

from nonebot.adapters.onebot.v11 import GroupMessageEvent, Message

from plugins.llm_group_bot import admin_command_state as _state
from plugins.llm_group_bot import admin_command_utils as _utils


async def handle_memory(rest: list[str], group_id: str) -> None:
    if not rest:
        await _state.finish_command(_state.admin_cmd, _utils.memory_help_text())
    if rest[0] == "lexicon":
        term = " ".join(rest[1:]).strip()
        memories = _state.storage.list_group_lexicon(group_id, term=term)
        await _state.finish_command(_state.admin_cmd, "\n".join(memories) if memories else "暂无词条记忆。")
    if rest[0] == "pending":
        memories = _state.storage.list_memories_by_status("pending_confirmation")
        await _state.finish_command(_state.admin_cmd, "\n".join(memories) if memories else "暂无待确认记忆。")
    if rest[0] == "conflicts":
        memories = _state.storage.list_memories_by_status("conflict")
        await _state.finish_command(_state.admin_cmd, "\n".join(memories) if memories else "暂无冲突记忆。")
    if len(rest) >= 2 and rest[0] == "approve":
        memory_id = _utils.parse_memory_id(rest[1])
        if memory_id is None:
            await _state.finish_command(_state.admin_cmd, "memory_id 必须是数字，例如：#bot memory approve 12")
        ok = _state.storage.approve_memory(memory_id)
        await _state.finish_command(_state.admin_cmd, "已批准。" if ok else "没有找到可批准的记忆。")
    if len(rest) >= 2 and rest[0] == "reject":
        memory_id = _utils.parse_memory_id(rest[1])
        if memory_id is None:
            await _state.finish_command(_state.admin_cmd, "memory_id 必须是数字，例如：#bot memory reject 12")
        ok = _state.storage.reject_memory(memory_id)
        await _state.finish_command(_state.admin_cmd, "已拒绝。" if ok else "没有找到可拒绝的记忆。")
    await _state.finish_command(_state.admin_cmd, _utils.memory_help_text())

async def handle_facts(rest: list[str]) -> None:
    if not rest:
        await _state.finish_command(_state.admin_cmd, _utils.facts_help_text())
    action = rest[0].lower()
    if len(rest) >= 2 and action == "user":
        facts = _state.storage.list_user_facts_text(rest[1], limit=20)
        await _state.finish_command(_state.admin_cmd, "\n".join(facts) if facts else "暂无该用户 FACT。")
    if action == "pending":
        facts = _state.storage.list_pending_facts(limit=20)
        lines = [
            f"#{fact.id} [{fact.fact_type}/{fact.claim_scope}] {fact.claim_text} "
            f"(subject={fact.subject_user_id}, src={fact.source_user_id}, conf={fact.confidence:.2f})"
            for fact in facts
        ]
        await _state.finish_command(_state.admin_cmd, "\n".join(lines) if lines else "暂无待确认 FACT。")
    if len(rest) >= 2 and action == "approve":
        fact_id = _utils.parse_memory_id(rest[1])
        if fact_id is None:
            await _state.finish_command(_state.admin_cmd, "fact_id 必须是数字，例如：#bot facts approve 12")
        record = _state.storage.approve_fact(fact_id)
        if record is None:
            await _state.finish_command(_state.admin_cmd, "没有找到可批准的 FACT。")
        await _state.maybe_update_profiles([record.subject_user_id], force=True)
        await _state.finish_command(_state.admin_cmd, "已批准。")
    if len(rest) >= 2 and action == "reject":
        fact_id = _utils.parse_memory_id(rest[1])
        if fact_id is None:
            await _state.finish_command(_state.admin_cmd, "fact_id 必须是数字，例如：#bot facts reject 12")
        ok = _state.storage.reject_fact(fact_id)
        await _state.finish_command(_state.admin_cmd, "已拒绝。" if ok else "没有找到可拒绝的 FACT。")
    if len(rest) >= 2 and action == "forget":
        fact_id = _utils.parse_memory_id(rest[1])
        if fact_id is None:
            await _state.finish_command(_state.admin_cmd, "fact_id 必须是数字，例如：#bot facts forget 12")
        record = _state.storage.forget_fact(fact_id)
        if record is None:
            await _state.finish_command(_state.admin_cmd, "没有找到可遗忘的 FACT。")
        await _state.maybe_update_profiles([record.subject_user_id], force=True)
        await _state.finish_command(_state.admin_cmd, "已遗忘。")
    await _state.finish_command(_state.admin_cmd, _utils.facts_help_text())

async def handle_persona(rest: list[str]) -> None:
    if not rest or rest[0] == "show":
        await _state.finish_command(_state.admin_cmd, _state.storage.format_persona())
    if rest[0] == "self":
        await handle_persona_self(rest[1:])
    await _state.finish_command(_state.admin_cmd, _utils.persona_help_text())

async def handle_persona_self(rest: list[str]) -> None:
    if not rest or rest[0] == "show":
        memories = _state.storage.list_self_memories("active")
        await _state.finish_command(_state.admin_cmd, "\n".join(memories) if memories else "暂无自我记忆。")
    if rest[0] == "pending":
        memories = _state.storage.list_self_memories("pending_confirmation")
        await _state.finish_command(_state.admin_cmd, "\n".join(memories) if memories else "暂无待确认自我记忆。")
    if rest[0] == "conflicts":
        memories = _state.storage.list_self_memories("conflict")
        await _state.finish_command(_state.admin_cmd, "\n".join(memories) if memories else "暂无冲突自我记忆。")
    if len(rest) >= 2 and rest[0] == "approve":
        memory_id = _utils.parse_memory_id(rest[1])
        if memory_id is None:
            await _state.finish_command(_state.admin_cmd, "memory_id 必须是数字，例如：#bot persona self approve 12")
        ok = _state.storage.approve_memory(memory_id)
        await _state.finish_command(_state.admin_cmd, "已批准。" if ok else "没有找到可批准的自我记忆。")
    if len(rest) >= 2 and rest[0] == "reject":
        memory_id = _utils.parse_memory_id(rest[1])
        if memory_id is None:
            await _state.finish_command(_state.admin_cmd, "memory_id 必须是数字，例如：#bot persona self reject 12")
        ok = _state.storage.reject_memory(memory_id)
        await _state.finish_command(_state.admin_cmd, "已拒绝。" if ok else "没有找到可拒绝的自我记忆。")
    if len(rest) >= 2 and rest[0] == "forget":
        memory_id = _utils.parse_memory_id(rest[1])
        if memory_id is None:
            await _state.finish_command(_state.admin_cmd, "memory_id 必须是数字，例如：#bot persona self forget 12")
        ok = _state.storage.forget_memory(memory_id)
        await _state.finish_command(_state.admin_cmd, "已遗忘。" if ok else "没有找到可遗忘的自我记忆。")
    await _state.finish_command(_state.admin_cmd, _utils.persona_help_text())

async def handle_user_fact_decision(
    matcher: Any,
    event: GroupMessageEvent,
    args: Message,
    *,
    approve: bool,
) -> None:
    user_id = str(event.user_id)
    if _state.storage.is_user_ignored(user_id):
        return

    command = "#approval" if approve else "#reject"
    parts = args.extract_plain_text().strip().split()
    if not parts:
        await _state.finish_command(matcher, f"用法：{command} <fact_id>")

    fact_id = _utils.parse_memory_id(parts[0])
    if fact_id is None:
        await _state.finish_command(matcher, f"fact_id 必须是数字，例如：{command} 12")

    if approve:
        accepted = _state.storage.approve_user_pending_fact(user_id, fact_id)
        if accepted is None:
            await _state.finish_command(matcher, "没有找到属于你的 pending FACT。")
        await _state.maybe_update_profiles([user_id], force=True)
        await _state.finish_command(matcher, f"已批准 FACT #{fact_id}。")

    ok = _state.storage.reject_user_pending_fact(user_id, fact_id)
    await _state.finish_command(matcher, f"已拒绝 FACT #{fact_id}。" if ok else "没有找到属于你的 pending FACT。")
