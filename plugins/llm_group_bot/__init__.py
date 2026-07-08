from __future__ import annotations

import asyncio
from contextvars import ContextVar
from dataclasses import replace
from typing import Any, Iterable

from loguru import logger
from nonebot import get_driver, on_command, on_message
from nonebot.adapters.onebot.v11 import Bot, GroupMessageEvent, Message, MessageSegment
from nonebot.params import CommandArg
from nonebot.plugin import PluginMetadata

from plugins.llm_group_bot import admin_commands as _admin_commands
from plugins.llm_group_bot import deferred_vision as _deferred_vision
from plugins.llm_group_bot import draw_command as _draw_command
from plugins.llm_group_bot import realtime_reply as _realtime_reply
from plugins.llm_group_bot.maintenance import PluginMaintenance
from plugins.llm_group_bot.observation_batch import ObservationBatchCoordinator
from qq_llm_bot.cognitive_agents import AgentPipeline
from qq_llm_bot.cognitive_storage import BotStorage
from qq_llm_bot.config import ParticipationMode, load_config
from qq_llm_bot.dashboard import register_dashboard_routes
from qq_llm_bot.llm import build_llm_client
from qq_llm_bot.models import MessageContext
from qq_llm_bot.onebot_context import build_message_context
from qq_llm_bot.outbound_queue import (
    OutboundGroupSendQueue,
)
from plugins.llm_group_bot.reply_sending import (
    ReplySender,
    message_has_reply as _message_has_reply,
    reply_record_text as _reply_record_text,
)
from qq_llm_bot.realtime_merge import split_image_descriptions_by_context
from qq_llm_bot.stickers import StickerLocalStore

__plugin_meta__ = PluginMetadata(
    name="llm-group-bot",
    description="LLM-ready QQ group character bot core.",
    usage="#bot status",
)

config = load_config()
storage = BotStorage.from_config(config)
llm = build_llm_client(config.llm, usage_recorder=storage.record_llm_usage)
pipeline = AgentPipeline(config, llm, vision_cache=storage)
sticker_store = StickerLocalStore(config)
_maintenance = PluginMaintenance(
    config=config,
    storage=storage,
    pipeline=pipeline,
    sticker_store=sticker_store,
)
observation_batch = ObservationBatchCoordinator(
    config=config,
    storage=storage,
    pipeline=pipeline,
    maintenance=_maintenance,
)
driver = get_driver()
outbound_queue = OutboundGroupSendQueue(
    config.bot,
    on_sticker_sent=_maintenance.record_outbound_sticker_sent,
)

if config.dashboard.enabled:
    register_dashboard_routes(
        driver,
        storage,
        config,
        on_fact_changed=lambda user_ids: _maintenance.update_profiles(user_ids, force=True),
    )


@driver.on_startup
async def _startup() -> None:
    storage.setup()
    _maintenance.cleanup_unused_stickers()
    outbound_queue.start_retry_worker(lambda: driver.bots.values())


@driver.on_shutdown
async def _shutdown() -> None:
    await _realtime_reply.stop_realtime_reply_workers()
    await outbound_queue.stop_retry_worker()


@driver.on_bot_connect
async def _on_bot_connect(bot: Bot) -> None:
    await asyncio.sleep(0.2)
    await outbound_queue.flush(bot, "bot connected")


@driver.on_bot_disconnect
async def _on_bot_disconnect(bot: Bot) -> None:
    queued = await outbound_queue.queue_size(bot.self_id)
    if queued:
        logger.warning("Bot {} disconnected with {} outbound messages waiting", bot.self_id, queued)


admin_cmd = on_command("bot", priority=5, block=True)
user_relation_cmd = on_command("relation", priority=5, block=True)
user_ignore_cmd = on_command("ignore", priority=5, block=True)
user_pending_cmd = on_command("pending", priority=5, block=True)
user_approval_cmd = on_command("approval", priority=5, block=True)
user_reject_cmd = on_command("reject", priority=5, block=True)
draw_cmd = on_command("draw", priority=5, block=True)

_command_reply_message_id: ContextVar[str] = ContextVar("command_reply_message_id", default="")


def _remember_command_reply(event: GroupMessageEvent) -> None:
    _command_reply_message_id.set(str(getattr(event, "message_id", "") or ""))


async def _finish_command(matcher: Any, message: Message | str) -> None:
    await matcher.finish(_with_command_reply(message))


async def _send_command(matcher: Any, message: Message | str) -> None:
    await matcher.send(_with_command_reply(message))


def _with_command_reply(message: Message | str) -> Message | str:
    reply_to = _command_reply_message_id.get("")
    if not reply_to:
        return message
    if isinstance(message, Message) and _message_has_reply(message):
        return message
    wrapped = Message()
    wrapped += MessageSegment.reply(reply_to)
    if isinstance(message, Message):
        wrapped += message
    elif message:
        wrapped += MessageSegment.text(message)
    return wrapped


_draw_command.configure(
    config_=config,
    storage_=storage,
    llm_=llm,
    outbound_queue_=outbound_queue,
    maintenance_=_maintenance,
    draw_cmd_=draw_cmd,
    finish_command=_finish_command,
    remember_command_reply=_remember_command_reply,
)


@admin_cmd.handle()
async def _handle_admin_command(
    bot: Bot,
    event: GroupMessageEvent,
    args: Message = CommandArg(),
) -> None:
    _remember_command_reply(event)
    user_id = str(event.user_id)
    group_id = str(event.group_id)
    text = args.extract_plain_text().strip()

    if storage.is_user_ignored(user_id):
        return

    if not storage.is_admin(user_id):
        await _finish_command(admin_cmd, "权限不足。")

    parts = text.split()
    if not parts:
        await _finish_command(admin_cmd, _admin_commands.help_text())

    topic = parts[0].lower()
    rest = parts[1:]

    if topic == "status":
        enabled = storage.is_group_enabled(group_id)
        mode = storage.get_group_mode(group_id, config.bot.default_group_mode)
        await _finish_command(
            admin_cmd,
            f"group={group_id}\nenabled={enabled}\nmode={mode}\n"
            f"admins={', '.join(storage.list_admins()) or '(none)'}"
        )

    if topic == "mode":
        if not rest:
            await _finish_command(admin_cmd, "用法：#bot mode silent|passive|active")
        mode = _admin_commands.normalize_mode(rest[0])
        if mode is None:
            await _finish_command(admin_cmd, "模式只能是 silent/passive/active，或 静默/被动/主动。")
        storage.set_group_mode(group_id, mode)
        await _finish_command(admin_cmd, f"已切换本群模式：{mode}")

    if topic == "whitelist":
        await _admin_commands.handle_whitelist(rest)

    if topic == "admin":
        await _admin_commands.handle_admin(rest, user_id)

    if topic in {"ignore", "ignored"}:
        await _admin_commands.handle_ignore(rest)

    if topic == "memory":
        await _admin_commands.handle_memory(rest, group_id)

    if topic == "facts":
        await _admin_commands.handle_facts(rest)

    if topic == "profile":
        await _admin_commands.handle_profile(rest)

    if topic in {"stickers", "sticker", "表情", "表情包"}:
        await _admin_commands.handle_stickers(rest, group_id)

    if topic == "persona":
        await _admin_commands.handle_persona(rest)

    if topic == "llm":
        await _admin_commands.handle_llm(bot, event, rest)

    if topic in {"token", "tokens"}:
        await _admin_commands.handle_token_usage()

    if topic == "why":
        await _finish_command(admin_cmd, storage.get_last_decision(group_id))

    if topic == "relation":
        await _admin_commands.handle_relation(rest, group_id)

    if topic == "forget":
        await _admin_commands.handle_forget(rest)

    await _finish_command(admin_cmd, _admin_commands.help_text())


@user_relation_cmd.handle()
async def _handle_user_relation_command(event: GroupMessageEvent) -> None:
    _remember_command_reply(event)
    user_id = str(event.user_id)
    if storage.is_user_ignored(user_id):
        return
    await _finish_command(
        user_relation_cmd,
        _admin_commands.format_user_relation(str(event.group_id), user_id),
    )


@user_ignore_cmd.handle()
async def _handle_user_ignore_command(event: GroupMessageEvent) -> None:
    _remember_command_reply(event)
    user_id = str(event.user_id)
    if storage.is_user_ignored(user_id):
        storage.remove_ignored_user(user_id)
        await _finish_command(user_ignore_cmd, "已取消忽略。之后普通消息会重新进入机器人处理。")
    storage.add_ignored_user(user_id)
    await _finish_command(user_ignore_cmd, "已加入忽略名单。之后普通消息不会进入机器人处理。")


@user_pending_cmd.handle()
async def _handle_user_pending_command(event: GroupMessageEvent) -> None:
    _remember_command_reply(event)
    user_id = str(event.user_id)
    if storage.is_user_ignored(user_id):
        return
    facts = storage.list_user_facts(user_id, limit=0, status="pending_confirmation")
    if not facts:
        await _finish_command(user_pending_cmd, "暂无关于你的 pending FACT。")
    await _finish_command(
        user_pending_cmd,
        "\n".join(_admin_commands.format_user_pending_fact(fact) for fact in facts),
    )


@user_approval_cmd.handle()
async def _handle_user_approval_command(
    event: GroupMessageEvent,
    args: Message = CommandArg(),
) -> None:
    _remember_command_reply(event)
    await _admin_commands.handle_user_fact_decision(user_approval_cmd, event, args, approve=True)


@user_reject_cmd.handle()
async def _handle_user_reject_command(
    event: GroupMessageEvent,
    args: Message = CommandArg(),
) -> None:
    _remember_command_reply(event)
    await _admin_commands.handle_user_fact_decision(user_reject_cmd, event, args, approve=False)


@draw_cmd.handle()
async def _handle_draw_command(
    bot: Bot,
    event: GroupMessageEvent,
    args: Message = CommandArg(),
) -> None:
    await _draw_command.handle_draw_command(bot, event, args)


group_message = on_message(priority=50, block=False)
reply_sender = ReplySender(
    bot_config=config.bot,
    outbound_queue=outbound_queue,
    fallback_sender=group_message.send,
    recent_bot_reply_texts=storage.get_recent_bot_reply_texts,
)


@group_message.handle()
async def _handle_group_message(bot: Bot, event: GroupMessageEvent) -> None:
    group_id = str(event.group_id)
    if not storage.is_group_enabled(group_id):
        return

    context = await build_message_context(bot, event, bot_names=config.bot.nicknames)
    storage.record_message(context)
    if storage.is_user_ignored(context.user_id):
        return
    _maintenance.cleanup_unused_stickers()
    observation_batch.register_pending_vision(context)
    observation_batch.ensure_deferred_vision_task(context)

    mode = storage.get_group_mode(group_id, config.bot.default_group_mode)
    if await _realtime_reply.maybe_enqueue_realtime_reply(bot, context, mode):
        return

    if observation_batch.should_defer_realtime_pipeline(context, mode):
        await observation_batch.defer_observation(context, mode)
        return

    sent_reply = await _process_group_context(bot, context, mode)
    if sent_reply:
        await group_message.finish()


async def _process_group_context(
    bot: Bot,
    context: MessageContext,
    mode: ParticipationMode,
    source_contexts: Iterable[MessageContext] | None = None,
    commit_guard: Any | None = None,
) -> bool:
    contexts = tuple(source_contexts or (context,))
    await observation_batch.wait_for_relevant_pending_vision(context)
    context = observation_batch.context_with_relevant_pending_images(context)

    await observation_batch.flush_observation_batch(context.group_id)
    snapshot = storage.build_snapshot(context)
    try:
        result = await pipeline.run(context, mode, snapshot, analyze_images=False)
    except asyncio.CancelledError:
        raise
    except Exception:
        raise

    if commit_guard is not None and not await commit_guard():
        return False

    fact_write = storage.record_fact_candidates(result.facts)
    memory_write = storage.record_memory_candidates(result.memories)
    _record_image_descriptions(contexts, result.image_descriptions)
    await _maintenance.record_sticker_candidates(context, result.sticker_candidates)
    await _maintenance.update_profiles(
        [fact.subject_user_id for fact in fact_write.accepted],
        force=bool(fact_write.accepted),
    )
    storage.apply_relationship_delta(context.group_id, context.user_id, result.relationship_delta)
    await _maintenance.reflect_group(context.group_id)
    conflict_reply = storage.build_conflict_confirmation(memory_write.conflicts, context, mode)
    final_reply = conflict_reply or result.reply
    selected_sticker = None if conflict_reply else result.selected_sticker

    if not final_reply:
        storage.record_decision(context, result.decision, "")
        if result.final_qa_blocked_reply:
            storage.record_final_qa_block(
                context,
                result.decision,
                snapshot,
                candidate_reply=result.final_qa_blocked_reply,
                qa_reason=result.final_qa_reason,
                qa_categories=result.final_qa_categories,
                qa_confidence=result.final_qa_confidence,
            )
        return False

    if conflict_reply:
        qa_result = await pipeline.review_reply(context, result.decision, snapshot, final_reply)
        if not qa_result.allowed:
            blocked_decision = _final_qa_blocked_decision(result.decision, qa_result.reason)
            storage.record_decision(
                context,
                blocked_decision,
                "",
            )
            storage.record_final_qa_block(
                context,
                blocked_decision,
                snapshot,
                candidate_reply=final_reply,
                qa_reason=qa_result.reason,
                qa_categories=qa_result.categories,
                qa_confidence=qa_result.confidence,
            )
            return False
    else:
        storage.record_memory_candidates(result.reply_self_memories)

    send_result = await reply_sender.send_group_reply(
        final_reply,
        selected_sticker,
        context.message_id,
        bot=bot,
        context=context,
        decision=result.decision,
        allow_bubbles=not bool(conflict_reply),
    )
    decision_reply = _reply_record_text(send_result.parts, send_result.sticker)
    storage.record_decision(context, result.decision, decision_reply)
    storage.record_bot_reply_parts(
        context.group_id,
        str(bot.self_id),
        send_result.parts or ((decision_reply,) if decision_reply else ()),
    )
    if send_result.sticker and not send_result.queued:
        storage.record_sticker_sent(send_result.sticker.id, usage_date=_maintenance.usage_date())
        _maintenance.cleanup_unused_stickers()
    return True


def _record_image_descriptions(
    contexts: Iterable[MessageContext],
    descriptions: list[str],
) -> None:
    for context, context_descriptions in split_image_descriptions_by_context(contexts, descriptions):
        storage.update_image_descriptions(
            context.group_id,
            context.message_id,
            context_descriptions,
        )


_realtime_reply.configure(
    config_=config,
    process_group_context=_process_group_context,
    should_defer_realtime_pipeline=observation_batch.should_defer_realtime_pipeline,
)

_deferred_vision.configure(
    config_=config,
    record_deferred_vision=observation_batch.record_deferred_vision,
    refresh_observation_buffer_image_descriptions=(
        observation_batch.refresh_observation_buffer_image_descriptions
    ),
)

_admin_commands.configure(
    storage_=storage,
    config_=config,
    admin_cmd_=admin_cmd,
    sticker_store_=sticker_store,
    llm_=llm,
    finish_command=_finish_command,
    update_profiles=_maintenance.update_profiles,
)

def _final_qa_blocked_decision(decision: Any, reason: str) -> Any:
    return replace(
        decision,
        action="observe",
        reason=f"{decision.reason}; final QA blocked reply: {reason}",
        score=min(decision.score, 0.49),
    )
