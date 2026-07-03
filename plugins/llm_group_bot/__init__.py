from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from nonebot import get_driver, on_command, on_message
from nonebot.adapters.onebot.v11 import Bot, GroupMessageEvent, Message, MessageSegment
from nonebot.params import CommandArg
from nonebot.plugin import PluginMetadata

from qq_llm_bot.cognitive_agents import AgentPipeline
from qq_llm_bot.cognitive_storage import BotStorage
from qq_llm_bot.config import ParticipationMode, load_config
from qq_llm_bot.dashboard import register_dashboard_routes
from qq_llm_bot.llm import build_llm_client, is_llm_configured, normalize_chat_completions_url
from qq_llm_bot.models import MessageAttachment, MessageContext, StickerAssetRecord, StickerCandidate
from qq_llm_bot.stickers import StickerLocalStore

__plugin_meta__ = PluginMetadata(
    name="llm-group-bot",
    description="LLM-ready QQ group character bot core.",
    usage="#bot status",
)

config = load_config()
storage = BotStorage.from_config(config)
llm = build_llm_client(config.llm)
pipeline = AgentPipeline(config, llm, vision_cache=storage)
sticker_store = StickerLocalStore(config)
driver = get_driver()

if config.dashboard.enabled:
    register_dashboard_routes(driver, storage, config)


@driver.on_startup
async def _startup() -> None:
    storage.setup()


admin_cmd = on_command("bot", priority=5, block=True)


@admin_cmd.handle()
async def _handle_admin_command(event: GroupMessageEvent, args: Message = CommandArg()) -> None:
    user_id = str(event.user_id)
    group_id = str(event.group_id)
    text = args.extract_plain_text().strip()

    if not storage.is_admin(user_id):
        await admin_cmd.finish("权限不足。")

    parts = text.split()
    if not parts:
        await admin_cmd.finish(_help_text())

    topic = parts[0].lower()
    rest = parts[1:]

    if topic == "status":
        enabled = storage.is_group_enabled(group_id)
        mode = storage.get_group_mode(group_id, config.bot.default_group_mode)
        await admin_cmd.finish(
            f"group={group_id}\nenabled={enabled}\nmode={mode}\n"
            f"admins={', '.join(storage.list_admins()) or '(none)'}"
        )

    if topic == "mode":
        if not rest:
            await admin_cmd.finish("用法：#bot mode silent|passive|active")
        mode = _normalize_mode(rest[0])
        if mode is None:
            await admin_cmd.finish("模式只能是 silent/passive/active，或 静默/被动/主动。")
        storage.set_group_mode(group_id, mode)
        await admin_cmd.finish(f"已切换本群模式：{mode}")

    if topic == "whitelist":
        await _handle_whitelist(rest)

    if topic == "admin":
        await _handle_admin(rest, user_id)

    if topic == "memory":
        await _handle_memory(rest, group_id)

    if topic in {"stickers", "sticker", "表情", "表情包"}:
        await _handle_stickers(rest, group_id)

    if topic == "persona":
        await _handle_persona(rest)

    if topic == "llm":
        await _handle_llm(rest)

    if topic == "why":
        await admin_cmd.finish(storage.get_last_decision(group_id))

    if topic == "relation":
        await _handle_relation(rest, group_id)

    if topic == "forget":
        await _handle_forget(rest)

    await admin_cmd.finish(_help_text())


group_message = on_message(priority=50, block=False)


@group_message.handle()
async def _handle_group_message(bot: Bot, event: GroupMessageEvent) -> None:
    group_id = str(event.group_id)
    if not storage.is_group_enabled(group_id):
        return

    context = _build_context(bot, event)
    storage.record_message(context)

    mode = storage.get_group_mode(group_id, config.bot.default_group_mode)
    snapshot = storage.build_snapshot(context)
    result = await pipeline.run(context, mode, snapshot)

    memory_write = storage.record_memory_candidates(result.memories)
    storage.record_memory_candidates(result.reply_self_memories)
    storage.update_image_descriptions(context.group_id, context.message_id, result.image_descriptions)
    await _record_sticker_candidates(context, result.sticker_candidates)
    storage.apply_relationship_delta(context.group_id, context.user_id, result.relationship_delta)
    await _maybe_reflect_group(context.group_id)
    conflict_reply = storage.build_conflict_confirmation(memory_write.conflicts, context, mode)
    final_reply = conflict_reply or result.reply
    selected_sticker = None if conflict_reply else result.selected_sticker
    decision_reply = _reply_record_text(final_reply, selected_sticker)
    storage.record_decision(context, result.decision, decision_reply)

    if final_reply:
        storage.record_bot_reply(context.group_id, str(bot.self_id), decision_reply)
        if selected_sticker:
            storage.record_sticker_sent(selected_sticker.id)
        await group_message.finish(_reply_message(final_reply, selected_sticker))


async def _handle_whitelist(rest: list[str]) -> None:
    if not rest or rest[0] == "list":
        groups = storage.list_enabled_groups()
        await admin_cmd.finish("已启用群：" + (", ".join(groups) if groups else "(none)"))

    action = rest[0].lower()
    if len(rest) < 2 or action not in {"add", "remove"}:
        await admin_cmd.finish("用法：#bot whitelist list|add <group_id>|remove <group_id>")

    group_id = rest[1]
    storage.set_group_enabled(group_id, action == "add")
    await admin_cmd.finish(f"已{'启用' if action == 'add' else '停用'}群：{group_id}")


async def _handle_admin(rest: list[str], current_user_id: str) -> None:
    if not rest or rest[0] == "list":
        await admin_cmd.finish("管理员：" + (", ".join(storage.list_admins()) or "(none)"))

    action = rest[0].lower()
    if len(rest) < 2 or action not in {"add", "remove"}:
        await admin_cmd.finish("用法：#bot admin list|add <qq_id>|remove <qq_id>")

    target = rest[1]
    if action == "add":
        storage.add_admin(target)
        await admin_cmd.finish(f"已添加管理员：{target}")

    if target == current_user_id:
        await admin_cmd.finish("不能移除当前正在操作的管理员。")

    storage.remove_admin(target)
    await admin_cmd.finish(f"已移除管理员：{target}")


async def _handle_memory(rest: list[str], group_id: str) -> None:
    if not rest:
        await admin_cmd.finish(_memory_help_text())
    if len(rest) >= 2 and rest[0] == "user":
        memories = storage.list_user_memories(rest[1])
        await admin_cmd.finish("\n".join(memories) if memories else "暂无该用户记忆。")
    if rest[0] == "lexicon":
        term = " ".join(rest[1:]).strip()
        memories = storage.list_group_lexicon(group_id, term=term)
        await admin_cmd.finish("\n".join(memories) if memories else "暂无词条记忆。")
    if rest[0] == "pending":
        memories = storage.list_memories_by_status("pending_confirmation")
        await admin_cmd.finish("\n".join(memories) if memories else "暂无待确认记忆。")
    if rest[0] == "conflicts":
        memories = storage.list_memories_by_status("conflict")
        await admin_cmd.finish("\n".join(memories) if memories else "暂无冲突记忆。")
    if len(rest) >= 2 and rest[0] == "approve":
        memory_id = _parse_memory_id(rest[1])
        if memory_id is None:
            await admin_cmd.finish("memory_id 必须是数字，例如：#bot memory approve 12")
        ok = storage.approve_memory(memory_id)
        await admin_cmd.finish("已批准。" if ok else "没有找到可批准的记忆。")
    if len(rest) >= 2 and rest[0] == "reject":
        memory_id = _parse_memory_id(rest[1])
        if memory_id is None:
            await admin_cmd.finish("memory_id 必须是数字，例如：#bot memory reject 12")
        ok = storage.reject_memory(memory_id)
        await admin_cmd.finish("已拒绝。" if ok else "没有找到可拒绝的记忆。")
    await admin_cmd.finish(_memory_help_text())


async def _handle_stickers(rest: list[str], group_id: str) -> None:
    action = rest[0].lower() if rest else "list"
    if action == "list":
        limit = _parse_memory_id(rest[1]) if len(rest) >= 2 else 20
        stickers = storage.list_stickers(group_id, limit=limit or 20)
        await admin_cmd.finish("\n\n".join(stickers) if stickers else "本群暂无已保存表情包。")
    if len(rest) >= 2 and action in {"enable", "disable"}:
        sticker_id = _parse_memory_id(rest[1])
        if sticker_id is None:
            await admin_cmd.finish("sticker_id 必须是数字，例如：#bot stickers disable 12")
        ok = storage.set_sticker_enabled(sticker_id, action == "enable")
        await admin_cmd.finish("已更新表情状态。" if ok else "没有找到该表情。")
    await admin_cmd.finish("用法：#bot stickers list [数量]|enable <id>|disable <id>")


async def _handle_persona(rest: list[str]) -> None:
    if not rest or rest[0] == "show":
        await admin_cmd.finish(storage.format_persona())
    if rest[0] == "self":
        await _handle_persona_self(rest[1:])
    await admin_cmd.finish(_persona_help_text())


async def _handle_persona_self(rest: list[str]) -> None:
    if not rest or rest[0] == "show":
        memories = storage.list_self_memories("active")
        await admin_cmd.finish("\n".join(memories) if memories else "暂无自我记忆。")
    if rest[0] == "pending":
        memories = storage.list_self_memories("pending_confirmation")
        await admin_cmd.finish("\n".join(memories) if memories else "暂无待确认自我记忆。")
    if rest[0] == "conflicts":
        memories = storage.list_self_memories("conflict")
        await admin_cmd.finish("\n".join(memories) if memories else "暂无冲突自我记忆。")
    if len(rest) >= 2 and rest[0] == "approve":
        memory_id = _parse_memory_id(rest[1])
        if memory_id is None:
            await admin_cmd.finish("memory_id 必须是数字，例如：#bot persona self approve 12")
        ok = storage.approve_memory(memory_id)
        await admin_cmd.finish("已批准。" if ok else "没有找到可批准的自我记忆。")
    if len(rest) >= 2 and rest[0] == "reject":
        memory_id = _parse_memory_id(rest[1])
        if memory_id is None:
            await admin_cmd.finish("memory_id 必须是数字，例如：#bot persona self reject 12")
        ok = storage.reject_memory(memory_id)
        await admin_cmd.finish("已拒绝。" if ok else "没有找到可拒绝的自我记忆。")
    if len(rest) >= 2 and rest[0] == "forget":
        memory_id = _parse_memory_id(rest[1])
        if memory_id is None:
            await admin_cmd.finish("memory_id 必须是数字，例如：#bot persona self forget 12")
        ok = storage.forget_memory(memory_id)
        await admin_cmd.finish("已遗忘。" if ok else "没有找到可遗忘的自我记忆。")
    await admin_cmd.finish(_persona_help_text())


async def _handle_llm(rest: list[str]) -> None:
    action = rest[0].lower() if rest else "status"
    if action == "status":
        provider = config.llm.provider
        configured = is_llm_configured(config.llm)
        url = normalize_chat_completions_url(config.llm.base_url) if config.llm.base_url else "(empty)"
        await admin_cmd.finish(
            "LLM 状态：\n"
            f"provider={provider}\n"
            f"configured={configured}\n"
            f"model={config.llm.model or '(empty)'}\n"
            f"url={url}\n"
            f"api_key_env={config.llm.api_key_env}"
        )

    if action == "test":
        prompt = " ".join(rest[1:]).strip() or "用一句话自然地打个招呼。"
        reply = await llm.complete_text(
            "你是 QQ 群里的拟人角色，说话自然、简短。",
            prompt,
        )
        await admin_cmd.finish(reply or "LLM 没有返回内容，请检查 provider/base_url/model/key。")

    await admin_cmd.finish("用法：#bot llm status|test [prompt]")


async def _handle_relation(rest: list[str], group_id: str) -> None:
    if not rest:
        await admin_cmd.finish("用法：#bot relation <qq_id>")
    await admin_cmd.finish(storage.format_relationship(group_id, rest[0]))


async def _handle_forget(rest: list[str]) -> None:
    if not rest:
        await admin_cmd.finish("用法：#bot forget <memory_id>")
    memory_id = _parse_memory_id(rest[0])
    if memory_id is None:
        await admin_cmd.finish("memory_id 必须是数字，例如：#bot forget 12")
    ok = storage.forget_memory(memory_id)
    await admin_cmd.finish("已遗忘。" if ok else "没有找到可遗忘的记忆。")


async def _maybe_reflect_group(group_id: str) -> None:
    if not config.reflection.enabled:
        return
    if not storage.should_reflect(
        group_id,
        config.reflection.message_threshold,
        config.reflection.min_interval_seconds,
    ):
        return
    recent_messages = storage.get_recent_messages(group_id, config.reflection.recent_limit)
    prior_reflections = storage.list_memories("group", group_id, limit=3)
    reflection = await pipeline.reflect(group_id, recent_messages, prior_reflections)
    if reflection:
        storage.record_memory_candidates([reflection])


async def _record_sticker_candidates(
    context: MessageContext,
    candidates: list[StickerCandidate],
) -> None:
    if not config.stickers.enabled or not candidates:
        return
    for candidate in candidates:
        saved = await sticker_store.save_candidate(context, candidate)
        if saved is None:
            continue
        storage.upsert_sticker_asset(
            context,
            candidate,
            local_path=saved.local_path,
            sha256=saved.sha256,
        )


def _reply_message(reply: str, sticker: StickerAssetRecord | None) -> Message | str:
    if sticker is None:
        return reply
    file_ref = _sticker_file_ref(sticker)
    if not file_ref:
        return reply
    message = Message()
    if reply:
        message += MessageSegment.text(reply)
        message += MessageSegment.text("\n")
    message += MessageSegment.image(file=file_ref)
    return message


def _sticker_file_ref(sticker: StickerAssetRecord) -> str:
    local_path = sticker.local_path.strip()
    if local_path:
        path = Path(local_path)
        if path.exists():
            return path.resolve().as_uri()
    return sticker.url.strip()


def _reply_record_text(reply: str | None, sticker: StickerAssetRecord | None) -> str:
    text = reply or ""
    if sticker is None:
        return text
    label = sticker.usage or sticker.description or sticker.local_path or sticker.url
    marker = f"[表情 #{sticker.id}: {label}]"
    return f"{text}\n{marker}".strip()


def _build_context(bot: Bot, event: GroupMessageEvent) -> MessageContext:
    plain_text = event.get_plaintext().strip()
    sender = getattr(event, "sender", None)
    sender_nickname = _sender_field(sender, "nickname")
    sender_name = _sender_field(sender, "card") or sender_nickname
    sender_role = _sender_field(sender, "role")

    return MessageContext(
        group_id=str(event.group_id),
        user_id=str(event.user_id),
        message_id=str(event.message_id),
        plain_text=plain_text,
        raw_message=str(event.message),
        sender_name=sender_name,
        sender_nickname=sender_nickname,
        sender_role=sender_role,
        is_direct=_is_direct_message(bot, event, plain_text),
        timestamp=int(getattr(event, "time", 0) or time.time()),
        attachments=_extract_attachments(event),
    )


def _is_direct_message(bot: Bot, event: GroupMessageEvent, plain_text: str) -> bool:
    if getattr(event, "to_me", False):
        return True
    for segment in event.message:
        if segment.type == "at" and str(segment.data.get("qq")) == str(bot.self_id):
            return True
    return any(name and name in plain_text for name in config.bot.nicknames)


def _sender_field(sender: Any, field: str) -> str:
    if sender is None:
        return ""
    if isinstance(sender, dict):
        return str(sender.get(field, "") or "")
    return str(getattr(sender, field, "") or "")


def _extract_attachments(event: GroupMessageEvent) -> list[MessageAttachment]:
    attachments: list[MessageAttachment] = []
    for segment in event.message:
        if segment.type != "image":
            continue
        data = dict(segment.data)
        attachments.append(
            MessageAttachment(
                attachment_type="image",
                file=str(data.get("file", "") or ""),
                url=str(data.get("url", "") or data.get("file_url", "") or ""),
                summary=str(data.get("summary", "") or ""),
                raw_data=json.dumps(data, ensure_ascii=False),
            )
        )
    return attachments


def _parse_memory_id(value: str) -> int | None:
    try:
        return int(value.lstrip("#"))
    except ValueError:
        return None


def _normalize_mode(value: str) -> ParticipationMode | None:
    mapping = {
        "silent": "silent",
        "静默": "silent",
        "passive": "passive",
        "被动": "passive",
        "被动回复": "passive",
        "active": "active",
        "主动": "active",
        "主动参与": "active",
    }
    mode = mapping.get(value.lower(), mapping.get(value))
    return mode if mode in {"silent", "passive", "active"} else None  # type: ignore[return-value]


def _help_text() -> str:
    return (
        "可用指令：\n"
        "#bot status\n"
        "#bot mode silent|passive|active\n"
        "#bot whitelist list|add <group_id>|remove <group_id>\n"
        "#bot admin list|add <qq_id>|remove <qq_id>\n"
        "#bot memory user <qq_id>|lexicon [term]|pending|conflicts|approve <id>|reject <id>\n"
        "#bot stickers list [数量]|enable <id>|disable <id>\n"
        "#bot persona show|self [pending|conflicts|approve <id>|reject <id>|forget <id>]\n"
        "#bot llm status|test [prompt]\n"
        "#bot why\n"
        "#bot relation <qq_id>\n"
        "#bot forget <memory_id>"
    )


def _memory_help_text() -> str:
    return (
        "用法：\n"
        "#bot memory user <qq_id>\n"
        "#bot memory lexicon [term]\n"
        "#bot memory pending\n"
        "#bot memory conflicts\n"
        "#bot memory approve <memory_id>\n"
        "#bot memory reject <memory_id>"
    )


def _persona_help_text() -> str:
    return (
        "用法：\n"
        "#bot persona show\n"
        "#bot persona self\n"
        "#bot persona self pending\n"
        "#bot persona self conflicts\n"
        "#bot persona self approve <memory_id>\n"
        "#bot persona self reject <memory_id>\n"
        "#bot persona self forget <memory_id>"
    )
