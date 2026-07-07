from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import re
import time
from contextvars import ContextVar
from dataclasses import dataclass, replace
from datetime import datetime
from itertools import count
from pathlib import Path
from typing import Any, Iterable

from loguru import logger
from nonebot import get_driver, on_command, on_message
from nonebot.adapters.onebot.v11 import Bot, GroupMessageEvent, Message, MessageSegment
from nonebot.adapters.onebot.v11.exception import ActionFailed, ApiNotAvailable, NetworkError
from nonebot.exception import WebSocketClosed
from nonebot.params import CommandArg
from nonebot.plugin import PluginMetadata
from websockets.exceptions import ConnectionClosed

from qq_llm_bot.cognitive_agents import AgentPipeline
from qq_llm_bot.cognitive_storage import BotStorage
from qq_llm_bot.config import ParticipationMode, load_config
from qq_llm_bot.dashboard import register_dashboard_routes
from qq_llm_bot.directness import looks_like_bot_address, text_mentions_bot_name
from qq_llm_bot.draw_images import prepare_draw_reference_images
from qq_llm_bot.draw_reference import DrawIntentPlanner
from qq_llm_bot.image_generation import GeneratedImageStore
from qq_llm_bot.llm import (
    build_llm_client,
    is_llm_configured,
    normalize_chat_completions_url,
    normalize_responses_url,
)
from qq_llm_bot.llm_usage_report import format_llm_token_report
from qq_llm_bot.models import (
    ConversationSnapshot,
    FactRecord,
    MemoryRecord,
    MessageAttachment,
    MessageContext,
    ParticipationDecision,
    RelationDelta,
    StickerAssetRecord,
    StickerCandidate,
    TargetUserContext,
    UserProfileRecord,
)
from qq_llm_bot.onebot_messages import (
    parse_outgoing_mention_parts,
    render_message_text_and_mentions,
    render_message_text_and_mentions_with_forwards,
)
from qq_llm_bot.realtime_merge import (
    merge_realtime_contexts,
    split_image_descriptions_by_context,
)
from qq_llm_bot.reply_style import settings_from_bot_config, split_reply_bubbles, style_reply_text
from qq_llm_bot.stickers import StickerLocalStore, sticker_file_ref, sticker_file_refs

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
generated_image_store = GeneratedImageStore(config)
draw_intent_planner = DrawIntentPlanner(
    llm,
    bot_names=tuple(config.bot.nicknames),
)
driver = get_driver()
observation_buffers: dict[str, list[MessageContext]] = {}
observation_last_flush_at: dict[str, int] = {}


@dataclass
class _PendingVision:
    future: asyncio.Future[list[str]]
    created_at: float
    context: MessageContext
    worker: asyncio.Task[list[str]] | None = None


@dataclass
class _PendingRealtimeReply:
    group_id: str
    user_id: str
    mode: ParticipationMode
    contexts: list[MessageContext]
    first_message_at: float
    updated_at: float
    generation: int = 1
    committing: bool = False
    task: asyncio.Task[None] | None = None


@dataclass(frozen=True)
class _ReplySendResult:
    parts: tuple[str, ...]
    sticker: StickerAssetRecord | None = None
    queued: bool = False


@dataclass(frozen=True)
class _SingleReplySendResult:
    sticker: StickerAssetRecord | None = None
    queued: bool = False


@dataclass(frozen=True)
class _QueuedSendAttempt:
    message: Message | str
    sticker: StickerAssetRecord | None = None


@dataclass
class _QueuedOutboundMessage:
    id: int
    bot_self_id: str
    group_id: str
    send_attempts: tuple[_QueuedSendAttempt, ...]
    created_at: float
    next_attempt_at: float
    attempts: int = 0
    source: str = ""
    reason: str = ""


pending_vision_tasks: dict[tuple[str, str], _PendingVision] = {}
_PENDING_VISION_MAX_MESSAGES = 3
_PENDING_VISION_MAX_AGE_SECONDS = 120.0
_OUTBOUND_RETRY_WORKER_INTERVAL_SECONDS = 3.0
_outbound_queue_ids = count(1)
_outbound_queue: list[_QueuedOutboundMessage] = []
_outbound_queue_lock = asyncio.Lock()
_outbound_flush_lock = asyncio.Lock()
_outbound_retry_worker: asyncio.Task[None] | None = None
_pending_realtime_replies: dict[tuple[str, str], _PendingRealtimeReply] = {}
_pending_realtime_lock = asyncio.Lock()
_deferred_vision_lock: asyncio.Lock | None = None

if config.dashboard.enabled:
    register_dashboard_routes(
        driver,
        storage,
        config,
        on_fact_changed=lambda user_ids: _maybe_update_profiles(user_ids, force=True),
    )


@driver.on_startup
async def _startup() -> None:
    storage.setup()
    _maybe_cleanup_unused_stickers()
    _start_outbound_retry_worker()


@driver.on_shutdown
async def _shutdown() -> None:
    await _stop_realtime_reply_workers()
    await _stop_outbound_retry_worker()


@driver.on_bot_connect
async def _on_bot_connect(bot: Bot) -> None:
    await asyncio.sleep(0.2)
    await _flush_outbound_queue(bot, "bot connected")


@driver.on_bot_disconnect
async def _on_bot_disconnect(bot: Bot) -> None:
    queued = await _outbound_queue_size(bot.self_id)
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
_PROCESSING_ACK_EMOJI_ID = "124"  # QQ [OK]
_DRAW_FAILURE_REPLY = "哎呀，图片不见了，我的我的~"


def _remember_command_reply(event: GroupMessageEvent) -> None:
    _command_reply_message_id.set(str(getattr(event, "message_id", "") or ""))


async def _acknowledge_processing(bot: Bot, message_id: object, reason: str) -> None:
    target_message_id = _onebot_message_id(message_id)
    if target_message_id is None:
        return
    try:
        await bot.call_api(
            "set_msg_emoji_like",
            message_id=target_message_id,
            emoji_id=_PROCESSING_ACK_EMOJI_ID,
        )
    except Exception as exc:  # pragma: no cover - depends on NapCat extension availability
        logger.warning("Processing acknowledgement failed for {}: {}", reason, exc)


def _onebot_message_id(message_id: object) -> int | str | None:
    text = str(message_id or "").strip()
    if not text:
        return None
    try:
        return int(text)
    except ValueError:
        return text


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
        await _finish_command(admin_cmd, _help_text())

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
        mode = _normalize_mode(rest[0])
        if mode is None:
            await _finish_command(admin_cmd, "模式只能是 silent/passive/active，或 静默/被动/主动。")
        storage.set_group_mode(group_id, mode)
        await _finish_command(admin_cmd, f"已切换本群模式：{mode}")

    if topic == "whitelist":
        await _handle_whitelist(rest)

    if topic == "admin":
        await _handle_admin(rest, user_id)

    if topic in {"ignore", "ignored"}:
        await _handle_ignore(rest)

    if topic == "memory":
        await _handle_memory(rest, group_id)

    if topic == "facts":
        await _handle_facts(rest)

    if topic == "profile":
        await _handle_profile(rest)

    if topic in {"stickers", "sticker", "表情", "表情包"}:
        await _handle_stickers(rest, group_id)

    if topic == "persona":
        await _handle_persona(rest)

    if topic == "llm":
        await _handle_llm(bot, event, rest)

    if topic in {"token", "tokens"}:
        await _handle_token_usage()

    if topic == "why":
        await _finish_command(admin_cmd, storage.get_last_decision(group_id))

    if topic == "relation":
        await _handle_relation(rest, group_id)

    if topic == "forget":
        await _handle_forget(rest)

    await _finish_command(admin_cmd, _help_text())


@user_relation_cmd.handle()
async def _handle_user_relation_command(event: GroupMessageEvent) -> None:
    _remember_command_reply(event)
    user_id = str(event.user_id)
    if storage.is_user_ignored(user_id):
        return
    await _finish_command(user_relation_cmd, _format_user_relation(str(event.group_id), user_id))


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
    await _finish_command(user_pending_cmd, "\n".join(_format_user_pending_fact(fact) for fact in facts))


@user_approval_cmd.handle()
async def _handle_user_approval_command(
    event: GroupMessageEvent,
    args: Message = CommandArg(),
) -> None:
    _remember_command_reply(event)
    await _handle_user_fact_decision(user_approval_cmd, event, args, approve=True)


@user_reject_cmd.handle()
async def _handle_user_reject_command(
    event: GroupMessageEvent,
    args: Message = CommandArg(),
) -> None:
    _remember_command_reply(event)
    await _handle_user_fact_decision(user_reject_cmd, event, args, approve=False)


@draw_cmd.handle()
async def _handle_draw_command(
    bot: Bot,
    event: GroupMessageEvent,
    args: Message = CommandArg(),
) -> None:
    _remember_command_reply(event)
    group_id = str(event.group_id)
    user_id = str(event.user_id)
    is_admin = storage.is_admin(user_id)
    if not storage.is_group_enabled(group_id):
        return
    if storage.is_user_ignored(user_id):
        return
    if not config.image_generation.enabled:
        await _finish_command(draw_cmd, "生图功能未开启。")
    if not is_llm_configured(config.llm):
        await _finish_command(draw_cmd, "生图模型未配置，请检查 provider/base_url/model/key。")

    if not config.image_generation.model.strip():
        await _finish_command(draw_cmd, "image_generation.model is required for #draw.")

    prompt = args.extract_plain_text().strip()
    if not prompt:
        await _finish_command(draw_cmd, "用法：#draw <图片描述>")
    if len(prompt) > config.image_generation.max_prompt_chars:
        await _finish_command(
            draw_cmd,
            f"图片描述太长了，最多 {config.image_generation.max_prompt_chars} 个字符。"
        )

    relation = storage.get_relationship(group_id, user_id)
    if not is_admin and relation.trust < config.image_generation.min_trust:
        await _finish_command(
            draw_cmd,
            f"现在还不能生图：当前 trust={relation.trust}，需要 >= {config.image_generation.min_trust}。"
        )

    usage_date = _draw_usage_date()
    used_count = storage.count_image_generation_usage(user_id, usage_date)
    if not is_admin and used_count >= config.image_generation.daily_limit:
        await _finish_command(
            draw_cmd,
            f"今天的生图次数已经用完了（{used_count}/{config.image_generation.daily_limit}）。",
        )

    await _acknowledge_processing(bot, event.message_id, "draw")

    context = await _build_context(bot, event)
    storage.record_message(context)
    snapshot = storage.build_snapshot(context)
    reference_images = await prepare_draw_reference_images(
        await _draw_reference_image_attachments(bot, event, context),
        max_images=config.image_generation.max_reference_images,
        max_bytes=config.image_generation.reference_image_max_bytes,
        max_dimension=config.image_generation.reference_image_max_dimension,
        quality=config.image_generation.reference_image_quality,
        timeout_seconds=min(config.image_generation.timeout_seconds, 60.0),
    )
    if reference_images.error:
        await _finish_command(draw_cmd, reference_images.error)

    image_prompt = await _compose_draw_prompt(
        context,
        snapshot,
        prompt,
        reference_image_count=len(reference_images.image_urls),
    )
    if image_prompt is None:
        logger.warning(
            "Draw prompt composition failed: group={} user={} message={}",
            group_id,
            user_id,
            context.message_id,
        )
        await _finish_command(draw_cmd, _draw_failure_reply("提示词整理失败", is_admin))

    generated = await llm.generate_image(
        image_prompt,
        config.image_generation,
        image_urls=reference_images.image_urls,
    )
    if generated is None:
        detail = str(getattr(llm, "last_image_generation_error", "") or "")
        logger.warning(
            "Draw image generation returned no image: group={} user={} message={} detail={}",
            group_id,
            user_id,
            context.message_id,
            detail,
        )
        await _finish_command(
            draw_cmd,
            _draw_failure_reply("Responses image_generation 没有返回图片", is_admin, detail),
        )
    saved = generated_image_store.save(context, generated)
    if saved is None:
        logger.warning(
            "Draw generated image save failed: group={} user={} message={}",
            group_id,
            user_id,
            context.message_id,
        )
        await _finish_command(draw_cmd, _draw_failure_reply("图片保存失败", is_admin))

    if not await _send_generated_image(bot, context, saved):
        detail = saved.local_path or saved.url or saved.file_ref
        await _finish_command(
            draw_cmd,
            _draw_failure_reply("图片已生成但 QQ 发送失败", is_admin, detail),
        )

    storage.record_image_generation_usage(
        group_id,
        user_id,
        usage_date,
        image_prompt,
        saved.local_path or saved.url or saved.file_ref,
    )
    await draw_cmd.finish()


async def _send_generated_image(bot: Bot, context: MessageContext, saved: Any) -> bool:
    send_refs = [("file", saved.file_ref)]
    base64_ref = _generated_image_base64_ref(saved.local_path)
    if base64_ref:
        send_refs.append(("base64", base64_ref))

    prepared_attempts = [
        (ref_kind, include_reply, _generated_image_message(context.message_id if include_reply else "", file_ref))
        for ref_kind, file_ref in send_refs
        for include_reply in (True, False)
    ]
    for index, (ref_kind, include_reply, message) in enumerate(prepared_attempts):
        try:
            await draw_cmd.send(message)
            if ref_kind != "file" or not include_reply:
                logger.info(
                    "Generated image send succeeded via {} fallback reply={} for {}",
                    ref_kind,
                    include_reply,
                    saved.local_path or saved.url or saved.file_ref,
                )
            return True
        except ActionFailed as exc:
            logger.warning(
                "Generated image send failed via {} reply={} for {}: {}",
                ref_kind,
                include_reply,
                saved.local_path or saved.url or saved.file_ref,
                exc,
            )
        except Exception as exc:
            if not _should_queue_send_error(exc):
                raise
            queued = await _queue_outbound_group_attempts(
                bot,
                context.group_id,
                tuple(
                    _QueuedSendAttempt(attempt_message)
                    for _, _, attempt_message in prepared_attempts[index:]
                ),
                source="draw image",
                reason=_send_error_detail(exc),
            )
            if queued:
                logger.warning(
                    "Generated image send queued after transient failure for {}: {}",
                    saved.local_path or saved.url or saved.file_ref,
                    exc,
                )
                return True
            return False
    return False


def _generated_image_message(reply_to_message_id: str, file_ref: str) -> Message:
    message = Message()
    if reply_to_message_id:
        message += MessageSegment.reply(reply_to_message_id)
    message += MessageSegment.text("画好了：\n")
    message += MessageSegment.image(file=file_ref)
    return message


def _generated_image_base64_ref(local_path: str) -> str:
    if not local_path:
        return ""
    try:
        data = Path(local_path).read_bytes()
    except OSError as exc:
        logger.warning("Generated image fallback read failed for {}: {}", local_path, exc)
        return ""
    if not data:
        return ""
    return "base64://" + base64.b64encode(data).decode("ascii")


def _draw_failure_reply(stage: str, is_admin: bool, detail: str = "") -> str:
    if not is_admin:
        return _DRAW_FAILURE_REPLY
    suffix = f"\n详情：{detail}" if detail else ""
    return f"{_DRAW_FAILURE_REPLY}\n管理员调试：{stage}{suffix}"


group_message = on_message(priority=50, block=False)


@group_message.handle()
async def _handle_group_message(bot: Bot, event: GroupMessageEvent) -> None:
    group_id = str(event.group_id)
    if not storage.is_group_enabled(group_id):
        return

    context = await _build_context(bot, event)
    storage.record_message(context)
    if storage.is_user_ignored(context.user_id):
        return
    _maybe_cleanup_unused_stickers()
    _register_pending_vision(context)
    _ensure_deferred_vision_task(context)

    mode = storage.get_group_mode(group_id, config.bot.default_group_mode)
    if await _maybe_enqueue_realtime_reply(bot, context, mode):
        return

    if _should_defer_realtime_pipeline(context, mode):
        await _defer_observation(context, mode)
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
    await _wait_for_relevant_pending_vision(context)
    context = _context_with_relevant_pending_images(context)

    await _flush_observation_batch(context.group_id)
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
    await _record_sticker_candidates(context, result.sticker_candidates)
    await _maybe_update_profiles(
        [fact.subject_user_id for fact in fact_write.accepted],
        force=bool(fact_write.accepted),
    )
    storage.apply_relationship_delta(context.group_id, context.user_id, result.relationship_delta)
    await _maybe_reflect_group(context.group_id)
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

    send_result = await _send_group_reply(
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
        storage.record_sticker_sent(send_result.sticker.id, usage_date=_draw_usage_date())
        _maybe_cleanup_unused_stickers()
    return True


async def _maybe_enqueue_realtime_reply(
    bot: Bot,
    context: MessageContext,
    mode: ParticipationMode,
) -> bool:
    if not config.bot.realtime_merge_enabled:
        return False

    key = _realtime_reply_key(context)
    now = time.time()
    async with _pending_realtime_lock:
        pending = _pending_realtime_replies.get(key)
        if pending is not None and _can_append_realtime_context(pending, context, now):
            pending.contexts.append(context)
            pending.mode = mode
            pending.updated_at = now
            pending.generation += 1
            _cancel_realtime_task(pending)
            pending.task = asyncio.create_task(
                _run_pending_realtime_reply(bot, key, pending.generation)
            )
            logger.info(
                "Merged realtime reply context group={} user={} messages={} generation={}",
                context.group_id,
                context.user_id,
                len(pending.contexts),
                pending.generation,
            )
            return True

        if pending is not None:
            return False

        if _should_defer_realtime_pipeline(context, mode):
            return False

        pending = _PendingRealtimeReply(
            group_id=context.group_id,
            user_id=context.user_id,
            mode=mode,
            contexts=[context],
            first_message_at=now,
            updated_at=now,
        )
        pending.task = asyncio.create_task(_run_pending_realtime_reply(bot, key, pending.generation))
        _pending_realtime_replies[key] = pending
        return True


async def _run_pending_realtime_reply(
    bot: Bot,
    key: tuple[str, str],
    generation: int,
) -> None:
    try:
        pending = await _get_pending_realtime_reply(key, generation)
        if pending is None:
            return
        contexts = tuple(pending.contexts)
        context = merge_realtime_contexts(contexts)

        async def _commit_guard() -> bool:
            await _wait_realtime_grace(key, generation)
            return await _claim_pending_realtime_reply(key, generation)

        await _process_group_context(
            bot,
            context,
            pending.mode,
            source_contexts=contexts,
            commit_guard=_commit_guard,
        )
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # pragma: no cover - realtime task must not break the matcher loop
        logger.exception("Realtime reply task failed for {}: {}", key, exc)
    finally:
        await _discard_pending_realtime_reply(key, generation)


async def _get_pending_realtime_reply(
    key: tuple[str, str],
    generation: int,
) -> _PendingRealtimeReply | None:
    async with _pending_realtime_lock:
        pending = _pending_realtime_replies.get(key)
        if pending is None or pending.generation != generation:
            return None
        return pending


async def _wait_realtime_grace(key: tuple[str, str], generation: int) -> None:
    grace_seconds = max(0.0, float(config.bot.realtime_merge_grace_seconds))
    while grace_seconds > 0:
        pending = await _get_pending_realtime_reply(key, generation)
        if pending is None:
            return
        max_window = max(0.0, float(config.bot.realtime_merge_max_window_seconds))
        deadline = min(pending.updated_at + grace_seconds, pending.first_message_at + max_window)
        remaining = deadline - time.time()
        if remaining <= 0:
            return
        await asyncio.sleep(min(remaining, 0.25))


async def _claim_pending_realtime_reply(key: tuple[str, str], generation: int) -> bool:
    async with _pending_realtime_lock:
        pending = _pending_realtime_replies.get(key)
        if pending is None or pending.generation != generation or pending.committing:
            return False
        pending.committing = True
        return True


async def _discard_pending_realtime_reply(key: tuple[str, str], generation: int) -> None:
    async with _pending_realtime_lock:
        pending = _pending_realtime_replies.get(key)
        if pending is not None and pending.generation == generation:
            _pending_realtime_replies.pop(key, None)


async def _stop_realtime_reply_workers() -> None:
    async with _pending_realtime_lock:
        tasks = [
            pending.task
            for pending in _pending_realtime_replies.values()
            if pending.task is not None and not pending.task.done()
        ]
        _pending_realtime_replies.clear()
    for task in tasks:
        task.cancel()
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)


def _can_append_realtime_context(
    pending: _PendingRealtimeReply,
    context: MessageContext,
    now: float,
) -> bool:
    if pending.committing:
        return False
    if len(pending.contexts) >= config.bot.realtime_merge_max_messages:
        return False
    if now - pending.first_message_at > config.bot.realtime_merge_max_window_seconds:
        return False
    return (
        pending.group_id == context.group_id
        and pending.user_id == context.user_id
    )


def _cancel_realtime_task(pending: _PendingRealtimeReply) -> None:
    if pending.task is not None and not pending.task.done():
        pending.task.cancel()


def _realtime_reply_key(context: MessageContext) -> tuple[str, str]:
    return (context.group_id, context.user_id)


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


def _register_pending_vision(context: MessageContext) -> _PendingVision | None:
    if not _has_image_attachments(context):
        return None
    _cleanup_pending_vision_tasks()
    key = _pending_vision_key(context)
    existing = pending_vision_tasks.get(key)
    if existing is not None:
        return existing
    future: asyncio.Future[list[str]] = asyncio.get_running_loop().create_future()
    pending = _PendingVision(future=future, created_at=time.time(), context=context)
    pending_vision_tasks[key] = pending
    return pending


def _ensure_deferred_vision_task(context: MessageContext) -> asyncio.Task[list[str]] | None:
    pending = _register_pending_vision(context)
    if pending is None:
        return None
    if pending.worker is None or pending.worker.done():
        pending.worker = asyncio.create_task(_run_pending_deferred_vision(context))
    return pending.worker


async def _run_pending_deferred_vision(context: MessageContext) -> list[str]:
    descriptions: list[str] = []
    try:
        async with _get_deferred_vision_lock():
            descriptions = await _record_deferred_vision(context)
        _refresh_observation_buffer_image_descriptions(context, descriptions)
    except Exception as exc:  # pragma: no cover - pending vision must never block the group
        logger.warning(
            "Pending deferred image observation failed for group {} message {}: {}",
            context.group_id,
            context.message_id,
            exc,
        )
    finally:
        _finish_pending_vision(context, descriptions)
    return descriptions


def _get_deferred_vision_lock() -> asyncio.Lock:
    global _deferred_vision_lock
    if _deferred_vision_lock is None:
        _deferred_vision_lock = asyncio.Lock()
    return _deferred_vision_lock


def _finish_pending_vision(context: MessageContext, descriptions: list[str]) -> None:
    key = _pending_vision_key(context)
    pending = pending_vision_tasks.pop(key, None)
    if pending is None or pending.future.done():
        return
    pending.future.set_result(list(descriptions))


def _context_with_relevant_pending_images(context: MessageContext) -> MessageContext:
    if _has_image_attachments(context) or not _looks_like_image_context_followup(context):
        return context
    attachments: list[MessageAttachment] = []
    seen_urls = {
        attachment.url
        for attachment in context.attachments
        if attachment.attachment_type == "image" and attachment.url
    }
    for pending in _recent_pending_vision_tasks(context):
        for attachment in pending.context.attachments:
            if attachment.attachment_type != "image" or not attachment.url:
                continue
            if attachment.url in seen_urls:
                continue
            seen_urls.add(attachment.url)
            attachments.append(attachment)
            if len(attachments) >= config.vision.max_images_per_message:
                break
        if len(attachments) >= config.vision.max_images_per_message:
            break
    if not attachments:
        return context
    note = f"[相关未解析图片 x{len(attachments)}：来自最近群聊，当前消息可能在询问这些图片]"
    plain_text = "\n".join(part for part in (context.plain_text, note) if part).strip()
    return replace(context, plain_text=plain_text, attachments=[*context.attachments, *attachments])


def _refresh_observation_buffer_image_descriptions(
    context: MessageContext,
    descriptions: list[str],
) -> None:
    if not descriptions:
        return
    buffer = observation_buffers.get(context.group_id)
    if not buffer:
        return
    for index, buffered in enumerate(buffer):
        if buffered.message_id != context.message_id:
            continue
        buffer[index] = _context_with_deferred_image_descriptions(buffered, descriptions)
        return


async def _wait_for_relevant_pending_vision(context: MessageContext) -> bool:
    if not _looks_like_image_context_followup(context):
        _cleanup_pending_vision_tasks()
        return True

    pending = _recent_pending_vision_tasks(context)
    if not pending:
        return True

    logger.info(
        "Using {} pending image analyses as inline multimodal context for group {} message {}",
        len(pending),
        context.group_id,
        context.message_id,
    )
    return True


def _recent_pending_vision_tasks(context: MessageContext) -> list[_PendingVision]:
    _cleanup_pending_vision_tasks()
    current_key = _pending_vision_key(context)
    candidates = [
        pending
        for key, pending in pending_vision_tasks.items()
        if key != current_key and key[0] == context.group_id and not pending.future.done()
    ]
    candidates.sort(
        key=lambda item: (
            item.context.user_id == context.user_id,
            item.created_at,
        ),
        reverse=True,
    )
    return candidates[:_PENDING_VISION_MAX_MESSAGES]


def _cleanup_pending_vision_tasks() -> None:
    now = time.time()
    for key, pending in list(pending_vision_tasks.items()):
        if pending.future.done() or now - pending.created_at > _PENDING_VISION_MAX_AGE_SECONDS:
            pending_vision_tasks.pop(key, None)


def _pending_vision_key(context: MessageContext) -> tuple[str, str]:
    return (context.group_id, context.message_id)


def _has_image_attachments(context: MessageContext) -> bool:
    return any(attachment.attachment_type == "image" for attachment in context.attachments)


def _looks_like_image_context_followup(context: MessageContext) -> bool:
    if _has_image_attachments(context):
        return False
    compact = "".join(context.plain_text.split()).lower()
    if not compact:
        return False
    explicit_image_terms = (
        "图",
        "图片",
        "截图",
        "照片",
        "新闻",
        "image",
        "screenshot",
    )
    generic_reference_terms = (
        "这事",
        "这个事",
        "这件事",
        "这个事情",
        "这条",
        "这个",
        "这",
    )
    question_terms = (
        "发生什么",
        "发生啥",
        "啥情况",
        "什么情况",
        "怎么回事",
        "怎么看",
        "看法",
        "评价",
        "解释",
        "总结",
        "真的假的",
        "是真的吗",
    )
    return any(term in compact for term in explicit_image_terms) or (
        any(term in compact for term in generic_reference_terms)
        and any(term in compact for term in question_terms)
    )


def _should_defer_realtime_pipeline(context: MessageContext, mode: ParticipationMode) -> bool:
    if not config.observation_batch.enabled:
        return False
    if context.is_direct:
        return False
    if context.bot_mentioned:
        return False
    if _has_recent_bot_reply_context(context):
        return False
    if mode == "silent":
        return True
    if mode == "passive":
        return True
    if mode == "active":
        return not _looks_like_active_realtime_candidate(context)
    return False


def _has_recent_bot_reply_context(context: MessageContext) -> bool:
    text = context.plain_text.strip()
    if not text or len(text) > 160:
        return False
    recent_reply, _ = storage.get_recent_bot_reply_to_user(
        context.group_id,
        context.user_id,
        config.bot.interaction_followup_seconds,
    )
    return bool(recent_reply)


def _looks_like_active_realtime_candidate(context: MessageContext) -> bool:
    text = " ".join(context.plain_text.split())
    compact = "".join(text.split())
    if len(compact) < 6:
        return False
    realtime_cues = (
        "?",
        "？",
        "吗",
        "么",
        "怎么",
        "如何",
        "为什么",
        "咋",
        "要不要",
        "该不该",
        "有没有",
        "谁",
        "哪",
        "啥",
        "求",
        "建议",
        "推荐",
        "帮",
        "分析",
        "看法",
        "方案",
    )
    return any(cue in compact for cue in realtime_cues)


async def _defer_observation(context: MessageContext, mode: ParticipationMode) -> None:
    _ensure_deferred_vision_task(context)
    image_descriptions = [
        attachment.summary
        for attachment in context.attachments
        if attachment.attachment_type == "image" and attachment.summary
    ]
    buffer = observation_buffers.setdefault(context.group_id, [])
    if context.group_id not in observation_last_flush_at:
        observation_last_flush_at[context.group_id] = int(time.time())
    buffer.append(_context_with_deferred_image_descriptions(context, image_descriptions))
    storage.apply_relationship_delta(
        context.group_id,
        context.user_id,
        RelationDelta(familiarity=1, reason="deferred batch observation"),
    )
    storage.record_decision(
        context,
        ParticipationDecision("observe", _deferred_observation_reason(mode), mode, 0.0),
        "",
    )
    await _flush_observation_batch(context.group_id)


async def _record_deferred_vision(context: MessageContext) -> list[str]:
    if not context.attachments:
        return []
    try:
        vision = await pipeline.observe_vision(context)
    except Exception as exc:  # pragma: no cover - image observation must never break chat handling
        logger.warning(
            "Deferred image observation failed for group {} message {}: {}",
            context.group_id,
            context.message_id,
            exc,
        )
        return []

    image_descriptions = list(vision.attachment_descriptions or tuple(vision.descriptions))
    storage.update_image_descriptions(context.group_id, context.message_id, image_descriptions)
    await _record_sticker_candidates(context, list(vision.sticker_candidates))
    fact_write = storage.record_fact_candidates(list(vision.fact_candidates))
    memory_write = storage.record_memory_candidates(list(vision.memory_candidates))
    if memory_write.conflicts:
        logger.info(
            "Deferred image observation recorded {} memory conflicts in group {}; waiting for manual review",
            len(memory_write.conflicts),
            context.group_id,
        )
    await _maybe_update_profiles(
        [fact.subject_user_id for fact in fact_write.accepted],
        force=bool(fact_write.accepted),
    )
    return image_descriptions


def _context_with_deferred_image_descriptions(
    context: MessageContext,
    image_descriptions: list[str],
) -> MessageContext:
    descriptions = [description.strip() for description in image_descriptions if description.strip()]
    if not descriptions:
        return context
    lines = []
    if context.plain_text:
        lines.append(context.plain_text)
    lines.extend(f"[图片解读] {description}" for description in descriptions)
    return replace(context, plain_text="\n".join(lines))


def _deferred_observation_reason(mode: ParticipationMode) -> str:
    if mode == "silent":
        return "silent mode deferred to batch observation"
    if mode == "passive":
        return "passive non-direct message deferred to batch observation"
    return "active low-priority message deferred to batch observation"


async def _flush_observation_batch(group_id: str, force: bool = False) -> None:
    if not config.observation_batch.enabled:
        return
    buffer = observation_buffers.get(group_id, [])
    if not buffer:
        return

    now = int(time.time())
    last_flush = observation_last_flush_at.get(group_id, now)
    due_by_size = len(buffer) >= config.observation_batch.batch_size
    due_by_time = now - last_flush >= config.observation_batch.max_interval_seconds
    if not force and not due_by_size and not due_by_time:
        return

    batch_size = min(len(buffer), config.observation_batch.max_messages_per_batch)
    batch = list(buffer[:batch_size])
    try:
        result = await pipeline.observe_batch(
            group_id,
            batch,
            storage.list_memories("group", group_id, limit=3),
            storage.list_group_lexicon_records(group_id, limit=10),
        )
    except Exception as exc:  # pragma: no cover - batch observation must never break chat handling
        observation_last_flush_at[group_id] = now
        logger.warning("Batch observation failed for group {}: {}", group_id, exc)
        return

    del buffer[:batch_size]
    if not buffer:
        observation_buffers.pop(group_id, None)
    observation_last_flush_at[group_id] = now

    memories = list(result.memories)
    if result.reflection is not None:
        memories.append(result.reflection)
    fact_write = storage.record_fact_candidates(result.facts)
    memory_write = storage.record_memory_candidates(memories)
    if memory_write.conflicts:
        logger.info(
            "Batch observation recorded {} memory conflicts in group {}; waiting for manual review",
            len(memory_write.conflicts),
            group_id,
        )
    await _maybe_update_profiles(
        [fact.subject_user_id for fact in fact_write.accepted],
        force=bool(fact_write.accepted),
    )


async def _handle_whitelist(rest: list[str]) -> None:
    if not rest or rest[0] == "list":
        groups = storage.list_enabled_groups()
        await _finish_command(admin_cmd, "已启用群：" + (", ".join(groups) if groups else "(none)"))

    action = rest[0].lower()
    if len(rest) < 2 or action not in {"add", "remove"}:
        await _finish_command(admin_cmd, "用法：#bot whitelist list|add <group_id>|remove <group_id>")

    group_id = rest[1]
    storage.set_group_enabled(group_id, action == "add")
    await _finish_command(admin_cmd, f"已{'启用' if action == 'add' else '停用'}群：{group_id}")


async def _handle_admin(rest: list[str], current_user_id: str) -> None:
    if not rest or rest[0] == "list":
        await _finish_command(admin_cmd, "管理员：" + (", ".join(storage.list_admins()) or "(none)"))

    action = rest[0].lower()
    if len(rest) < 2 or action not in {"add", "remove"}:
        await _finish_command(admin_cmd, "用法：#bot admin list|add <qq_id>|remove <qq_id>")

    target = rest[1]
    if action == "add":
        storage.add_admin(target)
        await _finish_command(admin_cmd, f"已添加管理员：{target}")

    if target == current_user_id:
        await _finish_command(admin_cmd, "不能移除当前正在操作的管理员。")

    storage.remove_admin(target)
    await _finish_command(admin_cmd, f"已移除管理员：{target}")


async def _handle_ignore(rest: list[str]) -> None:
    if not rest or rest[0] == "list":
        ignored_users = storage.list_ignored_users()
        await _finish_command(
            admin_cmd,
            "ignored users: " + (", ".join(ignored_users) if ignored_users else "(none)"),
        )

    action = rest[0].lower()
    if len(rest) < 2 or action not in {"add", "remove"}:
        await _finish_command(admin_cmd, "Usage: #bot ignore list|add <qq_id>|remove <qq_id>")

    target = rest[1]
    if action == "add":
        storage.add_ignored_user(target)
        await _finish_command(admin_cmd, f"ignored user added: {target}")

    storage.remove_ignored_user(target)
    await _finish_command(admin_cmd, f"ignored user removed: {target}")


async def _handle_memory(rest: list[str], group_id: str) -> None:
    if not rest:
        await _finish_command(admin_cmd, _memory_help_text())
    if rest[0] == "lexicon":
        term = " ".join(rest[1:]).strip()
        memories = storage.list_group_lexicon(group_id, term=term)
        await _finish_command(admin_cmd, "\n".join(memories) if memories else "暂无词条记忆。")
    if rest[0] == "pending":
        memories = storage.list_memories_by_status("pending_confirmation")
        await _finish_command(admin_cmd, "\n".join(memories) if memories else "暂无待确认记忆。")
    if rest[0] == "conflicts":
        memories = storage.list_memories_by_status("conflict")
        await _finish_command(admin_cmd, "\n".join(memories) if memories else "暂无冲突记忆。")
    if len(rest) >= 2 and rest[0] == "approve":
        memory_id = _parse_memory_id(rest[1])
        if memory_id is None:
            await _finish_command(admin_cmd, "memory_id 必须是数字，例如：#bot memory approve 12")
        ok = storage.approve_memory(memory_id)
        await _finish_command(admin_cmd, "已批准。" if ok else "没有找到可批准的记忆。")
    if len(rest) >= 2 and rest[0] == "reject":
        memory_id = _parse_memory_id(rest[1])
        if memory_id is None:
            await _finish_command(admin_cmd, "memory_id 必须是数字，例如：#bot memory reject 12")
        ok = storage.reject_memory(memory_id)
        await _finish_command(admin_cmd, "已拒绝。" if ok else "没有找到可拒绝的记忆。")
    await _finish_command(admin_cmd, _memory_help_text())


async def _handle_facts(rest: list[str]) -> None:
    if not rest:
        await _finish_command(admin_cmd, _facts_help_text())
    action = rest[0].lower()
    if len(rest) >= 2 and action == "user":
        facts = storage.list_user_facts_text(rest[1], limit=20)
        await _finish_command(admin_cmd, "\n".join(facts) if facts else "暂无该用户 FACT。")
    if action == "pending":
        facts = storage.list_pending_facts(limit=20)
        lines = [
            f"#{fact.id} [{fact.fact_type}/{fact.claim_scope}] {fact.claim_text} "
            f"(subject={fact.subject_user_id}, src={fact.source_user_id}, conf={fact.confidence:.2f})"
            for fact in facts
        ]
        await _finish_command(admin_cmd, "\n".join(lines) if lines else "暂无待确认 FACT。")
    if len(rest) >= 2 and action == "approve":
        fact_id = _parse_memory_id(rest[1])
        if fact_id is None:
            await _finish_command(admin_cmd, "fact_id 必须是数字，例如：#bot facts approve 12")
        record = storage.approve_fact(fact_id)
        if record is None:
            await _finish_command(admin_cmd, "没有找到可批准的 FACT。")
        await _maybe_update_profiles([record.subject_user_id], force=True)
        await _finish_command(admin_cmd, "已批准。")
    if len(rest) >= 2 and action == "reject":
        fact_id = _parse_memory_id(rest[1])
        if fact_id is None:
            await _finish_command(admin_cmd, "fact_id 必须是数字，例如：#bot facts reject 12")
        ok = storage.reject_fact(fact_id)
        await _finish_command(admin_cmd, "已拒绝。" if ok else "没有找到可拒绝的 FACT。")
    if len(rest) >= 2 and action == "forget":
        fact_id = _parse_memory_id(rest[1])
        if fact_id is None:
            await _finish_command(admin_cmd, "fact_id 必须是数字，例如：#bot facts forget 12")
        record = storage.forget_fact(fact_id)
        if record is None:
            await _finish_command(admin_cmd, "没有找到可遗忘的 FACT。")
        await _maybe_update_profiles([record.subject_user_id], force=True)
        await _finish_command(admin_cmd, "已遗忘。")
    await _finish_command(admin_cmd, _facts_help_text())


async def _handle_profile(rest: list[str]) -> None:
    if not rest:
        await _finish_command(admin_cmd, "用法：#bot profile <qq_id>")
    await _finish_command(admin_cmd, storage.format_user_profile(rest[0]))


async def _handle_stickers(rest: list[str], group_id: str) -> None:
    action = rest[0].lower() if rest else "list"
    if action == "list":
        limit = _parse_memory_id(rest[1]) if len(rest) >= 2 else 20
        stickers = storage.list_stickers(group_id, limit=limit or 20)
        await _finish_command(admin_cmd, "\n\n".join(stickers) if stickers else "本群暂无已保存表情包。")
    if len(rest) >= 2 and action in {"enable", "disable"}:
        sticker_id = _parse_memory_id(rest[1])
        if sticker_id is None:
            await _finish_command(admin_cmd, "sticker_id 必须是数字，例如：#bot stickers disable 12")
        ok = storage.set_sticker_enabled(sticker_id, action == "enable")
        await _finish_command(admin_cmd, "已更新表情状态。" if ok else "没有找到该表情。")
    if len(rest) >= 2 and action in {"delete", "remove", "del", "rm"}:
        sticker_id = _parse_memory_id(rest[1])
        if sticker_id is None:
            await _finish_command(admin_cmd, "sticker_id 必须是数字，例如：#bot stickers delete 12")
        asset = storage.delete_sticker_asset(sticker_id)
        if asset is None:
            await _finish_command(admin_cmd, "没有找到该表情。")
        deleted_file = sticker_store.delete_saved_file(asset.local_path)
        suffix = "，本地图片也已删除。" if deleted_file else "，但没有找到可删除的本地图片。"
        await _finish_command(admin_cmd, f"已删除表情 #{sticker_id}{suffix}")
    await _finish_command(admin_cmd, "用法：#bot stickers list [数量]|enable <id>|disable <id>|delete <id>")


async def _handle_persona(rest: list[str]) -> None:
    if not rest or rest[0] == "show":
        await _finish_command(admin_cmd, storage.format_persona())
    if rest[0] == "self":
        await _handle_persona_self(rest[1:])
    await _finish_command(admin_cmd, _persona_help_text())


async def _handle_persona_self(rest: list[str]) -> None:
    if not rest or rest[0] == "show":
        memories = storage.list_self_memories("active")
        await _finish_command(admin_cmd, "\n".join(memories) if memories else "暂无自我记忆。")
    if rest[0] == "pending":
        memories = storage.list_self_memories("pending_confirmation")
        await _finish_command(admin_cmd, "\n".join(memories) if memories else "暂无待确认自我记忆。")
    if rest[0] == "conflicts":
        memories = storage.list_self_memories("conflict")
        await _finish_command(admin_cmd, "\n".join(memories) if memories else "暂无冲突自我记忆。")
    if len(rest) >= 2 and rest[0] == "approve":
        memory_id = _parse_memory_id(rest[1])
        if memory_id is None:
            await _finish_command(admin_cmd, "memory_id 必须是数字，例如：#bot persona self approve 12")
        ok = storage.approve_memory(memory_id)
        await _finish_command(admin_cmd, "已批准。" if ok else "没有找到可批准的自我记忆。")
    if len(rest) >= 2 and rest[0] == "reject":
        memory_id = _parse_memory_id(rest[1])
        if memory_id is None:
            await _finish_command(admin_cmd, "memory_id 必须是数字，例如：#bot persona self reject 12")
        ok = storage.reject_memory(memory_id)
        await _finish_command(admin_cmd, "已拒绝。" if ok else "没有找到可拒绝的自我记忆。")
    if len(rest) >= 2 and rest[0] == "forget":
        memory_id = _parse_memory_id(rest[1])
        if memory_id is None:
            await _finish_command(admin_cmd, "memory_id 必须是数字，例如：#bot persona self forget 12")
        ok = storage.forget_memory(memory_id)
        await _finish_command(admin_cmd, "已遗忘。" if ok else "没有找到可遗忘的自我记忆。")
    await _finish_command(admin_cmd, _persona_help_text())


async def _handle_llm(bot: Bot, event: GroupMessageEvent, rest: list[str]) -> None:
    action = rest[0].lower() if rest else "status"
    if action == "status":
        provider = config.llm.provider
        configured = is_llm_configured(config.llm)
        chat_url = (
            normalize_chat_completions_url(config.llm.base_url)
            if config.llm.base_url
            else "(empty)"
        )
        responses_url = (
            normalize_responses_url(config.llm.base_url) if config.llm.base_url else "(empty)"
        )
        image_model = config.image_generation.model or "(empty; required)"
        await _finish_command(
            admin_cmd,
            "LLM 状态：\n"
            f"provider={provider}\n"
            f"configured={configured}\n"
            f"model={config.llm.model or '(empty)'}\n"
            f"routing_enabled={config.llm.routing.enabled}\n"
            f"routing_base_model={config.llm.routing.base_model or '(empty)'}\n"
            f"routing_flagship_model={config.llm.routing.flagship_model or '(empty)'}\n"
            f"routing_vision_base_model={config.llm.routing.vision_base_model or '(empty)'}\n"
            f"chat_url={chat_url}\n"
            f"responses_url={responses_url}\n"
            f"api_key_env={config.llm.api_key_env}\n"
            f"image_generation_enabled={config.image_generation.enabled}\n"
            f"image_generation_model={image_model}\n"
            f"image_generation_size={config.image_generation.size}\n"
            f"image_generation_quality={config.image_generation.quality}\n"
            f"image_generation_format={config.image_generation.output_format}\n"
            f"image_generation_compression={config.image_generation.output_compression}\n"
            f"image_generation_timeout={config.image_generation.timeout_seconds}\n"
            f"image_generation_max_send_dimension={config.image_generation.max_send_dimension}"
        )

    if action == "test":
        prompt = " ".join(rest[1:]).strip() or "用一句话自然地打个招呼。"
        reply = await llm.complete_text(
            "你是 QQ 群里的拟人角色，说话自然、简短。",
            prompt,
            purpose="llm_test",
        )
        await _finish_command(admin_cmd, reply or "LLM 没有返回内容，请检查 provider/base_url/model/key。")

    await _finish_command(admin_cmd, "用法：#bot llm status|test [prompt]")


async def _handle_token_usage() -> None:
    now = int(time.time())
    data = storage.list_dashboard_llm_usage(since=now - 24 * 3600, limit=1)
    await _finish_command(admin_cmd, format_llm_token_report(data, hours=24))


async def _handle_relation(rest: list[str], group_id: str) -> None:
    if not rest:
        await _finish_command(admin_cmd, "用法：#bot relation <qq_id>|top [数量]|rank [数量]")
    action = rest[0].lower()
    if action in {"top", "rank", "ranking", "排行", "排行榜"}:
        limit = _parse_memory_id(rest[1]) if len(rest) >= 2 else 5
        if limit is None:
            await _finish_command(admin_cmd, "数量必须是数字，例如：#bot relation top 5")
        await _finish_command(admin_cmd, storage.format_relationship_ranking(group_id, limit))
    await _finish_command(admin_cmd, storage.format_relationship(group_id, rest[0]))


async def _handle_forget(rest: list[str]) -> None:
    if not rest:
        await _finish_command(admin_cmd, "用法：#bot forget <memory_id>")
    memory_id = _parse_memory_id(rest[0])
    if memory_id is None:
        await _finish_command(admin_cmd, "memory_id 必须是数字，例如：#bot forget 12")
    ok = storage.forget_memory(memory_id)
    await _finish_command(admin_cmd, "已遗忘。" if ok else "没有找到可遗忘的记忆。")


async def _handle_user_fact_decision(
    matcher: Any,
    event: GroupMessageEvent,
    args: Message,
    *,
    approve: bool,
) -> None:
    user_id = str(event.user_id)
    if storage.is_user_ignored(user_id):
        return

    command = "#approval" if approve else "#reject"
    parts = args.extract_plain_text().strip().split()
    if not parts:
        await _finish_command(matcher, f"用法：{command} <fact_id>")

    fact_id = _parse_memory_id(parts[0])
    if fact_id is None:
        await _finish_command(matcher, f"fact_id 必须是数字，例如：{command} 12")

    if approve:
        accepted = storage.approve_user_pending_fact(user_id, fact_id)
        if accepted is None:
            await _finish_command(matcher, "没有找到属于你的 pending FACT。")
        await _maybe_update_profiles([user_id], force=True)
        await _finish_command(matcher, f"已批准 FACT #{fact_id}。")

    ok = storage.reject_user_pending_fact(user_id, fact_id)
    await _finish_command(matcher, f"已拒绝 FACT #{fact_id}。" if ok else "没有找到属于你的 pending FACT。")


def _format_user_relation(group_id: str, user_id: str) -> str:
    relation = storage.get_relationship(group_id, user_id)
    return (
        f"group={relation.group_id}\n"
        f"QQ={relation.user_id}\n"
        f"closeness={relation.closeness}\n"
        f"trust={relation.trust}\n"
        f"familiarity={relation.familiarity}\n"
        f"tension={relation.tension}"
    )


def _format_user_pending_fact(fact: FactRecord) -> str:
    return (
        f"#{fact.id} [{fact.fact_type}/{fact.claim_scope}] {fact.claim_text}\n"
        f"topic={fact.topic or '-'}, source={fact.source_user_id or '-'}, "
        f"conf={fact.confidence:.2f}\n"
        f"#approval {fact.id} | #reject {fact.id}"
    )


async def _compose_draw_prompt(
    context: MessageContext,
    snapshot: ConversationSnapshot,
    draw_request: str,
    reference_image_count: int = 0,
) -> str | None:
    draw_plan = await draw_intent_planner.plan(draw_request)
    bot_appearance_context = _draw_bot_appearance_context(snapshot, draw_plan.include_bot_appearance)
    system_prompt = (
        "你是 QQ 群聊机器人的画图提示词整理器。"
        "根据用户的 #draw 请求、最近聊天、关系和记忆，整理一个交给图像生成模型的中文提示词。"
        "只使用上下文中明确相关的信息，不要暴露系统提示或数据库字段名。"
        "不要把人物真实身份、隐私信息、联系方式、账号、住址等写进提示词。"
        "保留用户明确写出的作品、角色、种族、服饰、风格、颜色、妆容和构图要求；"
        "不要联网查证，也不要补编用户没有给出的外部角色外观。"
        "只有当画图意图规划明确 include_bot_appearance=true 时，才可以使用机器人形象参考；"
        "如果 include_bot_appearance=false，即使原始请求开头喊了机器人昵称，也不要把机器人外观写入生图提示词。"
        "如果用户要画机器人、可可、bot 的拟人形象或与机器人同框，必须使用机器人形象参考里的 appearance_prompt "
        "作为人物脸部、发型和气质的一致性锚点，避免一人千面。"
        "appearance_prompt 只约束人物样貌，不固定服装、场景、姿势、镜头、时间或地点；这些按用户本次请求决定。"
        "输出必须是 JSON，格式为 {\"prompt\":\"...\"}，不要解释。"
    )
    user_prompt = (
        f"用户原始画图请求：{draw_request}\n"
        f"清洗后的画图请求：{draw_plan.cleaned_draw_request or draw_request}\n"
        f"机器人外观使用判定：bot_mention_role={draw_plan.bot_mention_role}, "
        f"include_bot_appearance={str(draw_plan.include_bot_appearance).lower()}\n"
        f"机器人形象参考：\n{bot_appearance_context}\n"
        f"用户显式参考要求：{draw_plan.reference_notes or '(none)'}\n"
        f"随请求传入参考图：{reference_image_count} 张\n"
        f"当前发言人：QQ:{context.user_id}，昵称：{context.sender_name or context.sender_nickname or '-'}\n"
        f"机器人昵称：{', '.join(config.bot.nicknames)}\n"
        f"最近群聊：\n{_draw_recent_context(snapshot)}\n"
        f"最近图片理解：\n{_draw_join(snapshot.recent_image_descriptions)}\n"
        f"发言人画像：\n{_draw_profile(snapshot.user_profile)}\n"
        f"发言人记忆：\n{_draw_memories(snapshot.user_memories[:8])}\n"
        f"发言人 FACT：\n{_draw_facts(snapshot.user_facts[:8])}\n"
        f"与发言人关系：{_draw_relationship(snapshot)}\n"
        f"被提及成员资料：\n{_draw_targets(snapshot.target_users)}\n"
        f"群复盘：\n{_draw_memories(snapshot.group_reflections[:4])}\n"
        f"群内词条：\n{_draw_memories(snapshot.group_lexicon[:6])}\n"
        "请把这些信息转成一个明确、可画、单张图片的提示词。"
        "提示词应描述主体、场景、风格、构图、情绪、色彩和必要细节；"
        "如果用户提供了参考图，提示词里要明确要求图像生成模型参考这些输入图的主体、构图、服饰或风格，"
        "但仍以用户文字要求为准；"
        "如果 include_bot_appearance=true，把 appearance_prompt 中的脸部、发型、气质特征写入提示词，"
        "但不要把 appearance_prompt 明确排除的场景或穿着固化进去；"
        "如果 include_bot_appearance=false，不要使用机器人形象参考，不要把机器人外观混入主体；"
        "如果用户请求很简单，也不要过度添加不相关记忆。"
    )
    model_tier = "flagship" if _draw_prompt_requires_flagship(context, snapshot, draw_request) else ""
    text = await llm.complete_text(
        system_prompt,
        user_prompt,
        purpose="draw_prompt",
        model_tier=model_tier,
    )
    prompt = _extract_draw_prompt(text)
    if (
        prompt is None
        and model_tier != "flagship"
        and _can_retry_draw_prompt_with_flagship()
    ):
        text = await llm.complete_text(
            system_prompt,
            user_prompt,
            purpose="draw_prompt",
            model_tier="flagship",
        )
        prompt = _extract_draw_prompt(text)
    return prompt


async def _draw_reference_image_attachments(
    bot: Bot,
    event: GroupMessageEvent,
    context: MessageContext,
) -> list[MessageAttachment]:
    attachments = list(context.attachments)
    fetch_reply = _fetch_reply_message(bot, event)
    reply_ids = _reply_segment_ids(event.message)
    if reply_ids:
        for reply_id in reply_ids:
            payload = await fetch_reply(reply_id)
            attachments.extend(_image_attachments_from_payload(payload))
    else:
        payload = _event_reply_payload(event, "")
        attachments.extend(_image_attachments_from_payload(payload))
    return attachments


def _reply_segment_ids(message: Any) -> list[str]:
    ids: list[str] = []
    for segment in _coerce_message_segments(message):
        if _segment_type(segment) != "reply":
            continue
        data = _segment_data(segment)
        reply_id = str(data.get("id") or data.get("message_id") or "").strip()
        if reply_id:
            ids.append(reply_id)
    return ids


def _image_attachments_from_payload(payload: Any) -> list[MessageAttachment]:
    if payload is None:
        return []
    if isinstance(payload, dict):
        message = payload.get("message")
        if message is None:
            message = payload.get("content")
        if message is None:
            message = payload.get("raw_message")
        return _image_attachments_from_message(message)
    return _image_attachments_from_message(payload)


def _image_attachments_from_message(message: Any) -> list[MessageAttachment]:
    attachments: list[MessageAttachment] = []
    for segment in _coerce_message_segments(message):
        segment_type = _segment_type(segment)
        data = _segment_data(segment)
        if segment_type == "image":
            attachments.append(_image_attachment_from_data(data))
            continue
        if segment_type in {"node", "forward"}:
            for key in ("content", "message", "raw_message"):
                if key in data:
                    attachments.extend(_image_attachments_from_message(data.get(key)))
    return attachments


def _image_attachment_from_data(data: dict[str, Any]) -> MessageAttachment:
    return MessageAttachment(
        attachment_type="image",
        file=str(data.get("file", "") or ""),
        url=str(data.get("url", "") or data.get("file_url", "") or ""),
        summary=str(data.get("summary", "") or ""),
        raw_data=json.dumps(data, ensure_ascii=False),
    )


def _coerce_message_segments(message: Any) -> list[Any]:
    if message is None:
        return []
    if isinstance(message, str):
        try:
            return list(Message(message))
        except Exception:
            return [{"type": "text", "data": {"text": message}}]
    if isinstance(message, dict):
        if "type" in message:
            return [message]
        for key in ("message", "content", "raw_message"):
            if key in message:
                return _coerce_message_segments(message.get(key))
        return []
    try:
        return list(message)
    except TypeError:
        return [message]


def _segment_type(segment: Any) -> str:
    if isinstance(segment, dict):
        return str(segment.get("type", "") or "")
    return str(getattr(segment, "type", "") or "")


def _segment_data(segment: Any) -> dict[str, Any]:
    raw_data = segment.get("data", {}) if isinstance(segment, dict) else getattr(segment, "data", {})
    return dict(raw_data) if isinstance(raw_data, dict) else {}


def _draw_bot_appearance_context(
    snapshot: ConversationSnapshot,
    include_bot_appearance: bool,
) -> str:
    if not include_bot_appearance:
        return "include_bot_appearance=false；本次不要使用机器人 appearance_prompt 或自我人设作为主体外观。"
    appearance = _draw_appearance_prompt(snapshot.persona_lines)
    if not appearance:
        return "include_bot_appearance=true；但当前没有配置 appearance_prompt。"
    return f"include_bot_appearance=true\nappearance_prompt: {appearance}"


def _draw_appearance_prompt(persona_lines: list[str]) -> str:
    for line in persona_lines:
        key, sep, value = line.partition(":")
        if sep and key.strip() == "appearance_prompt":
            return value.strip()
    return ""


def _extract_draw_prompt(text: str | None) -> str | None:
    if not text:
        return None
    raw = text.strip()
    if raw.startswith("```"):
        raw = raw.strip("`").strip()
        if raw.lower().startswith("json"):
            raw = raw[4:].strip()
    start = raw.find("{")
    end = raw.rfind("}")
    if start >= 0 and end > start:
        raw = raw[start : end + 1]
    try:
        data = json.loads(raw)
    except ValueError:
        prompt = _extract_truncated_draw_prompt(raw)
        if prompt:
            return prompt[: max(1, config.image_generation.max_prompt_chars)]
        return None
    prompt = str(data.get("prompt", "")).strip()
    if not prompt:
        return None
    return prompt[: max(1, config.image_generation.max_prompt_chars)]


def _extract_truncated_draw_prompt(raw: str) -> str | None:
    match = re.search(r'"prompt"\s*:\s*"(?P<prompt>.*)', raw, re.S)
    if not match:
        return None
    prompt = match.group("prompt")
    if '"' in prompt:
        prompt = prompt.rsplit('"', 1)[0]
    prompt = prompt.rstrip("}` \n\r\t")
    try:
        prompt = json.loads(f'"{prompt}"')
    except ValueError:
        prompt = prompt.replace('\\"', '"').replace("\\n", "\n")
    prompt = str(prompt).strip()
    return prompt or None


def _can_retry_draw_prompt_with_flagship() -> bool:
    retry_checker = getattr(llm, "should_retry_with_flagship", None)
    if not callable(retry_checker):
        return False
    try:
        return bool(retry_checker("draw_prompt"))
    except Exception as exc:  # pragma: no cover - retry checks must not break draw
        logger.warning("Draw prompt flagship retry check failed: {}", exc)
        return False


def _draw_prompt_requires_flagship(
    context: MessageContext,
    snapshot: ConversationSnapshot,
    draw_request: str,
) -> bool:
    text = "\n".join(
        (
            draw_request,
            context.plain_text,
            _draw_join(snapshot.recent_image_descriptions),
        )
    ).lower()
    if any(name and name.lower() in text for name in config.bot.nicknames):
        return True
    if snapshot.persona_lines and any(token in text for token in ("bot", "机器人", "可可", "人设")):
        return True
    high_risk_cues = (
        "隐私",
        "身份证",
        "手机号",
        "住址",
        "账号",
        "真实",
        "本人",
        "合照",
        "照着",
        "参考截图",
    )
    return any(cue in text for cue in high_risk_cues)


def _draw_recent_context(snapshot: ConversationSnapshot) -> str:
    if not snapshot.speaker_recent_messages and not snapshot.other_recent_messages:
        return _draw_join(snapshot.recent_messages[-12:])
    parts = []
    if snapshot.speaker_recent_messages:
        parts.append("发言人最近消息：\n" + _draw_join(snapshot.speaker_recent_messages[-8:]))
    if snapshot.other_recent_messages:
        parts.append("其他群友最近消息：\n" + _draw_join(snapshot.other_recent_messages[-8:]))
    return "\n\n".join(parts) if parts else "(none)"


def _draw_memories(memories: list[MemoryRecord]) -> str:
    if not memories:
        return "(none)"
    return "\n".join(f"[{item.kind}] {item.content}" for item in memories)


def _draw_facts(facts: list[FactRecord]) -> str:
    if not facts:
        return "(none)"
    return "\n".join(
        f"[{fact.fact_type}] {fact.claim_text} (topic={fact.topic}, conf={fact.confidence:.2f})"
        for fact in facts
    )


def _draw_profile(profile: UserProfileRecord | None) -> str:
    if profile is None:
        return "(none)"
    traits = json.dumps(profile.traits, ensure_ascii=False) if profile.traits else "{}"
    return f"{profile.summary}\ntraits={traits}"


def _draw_relationship(snapshot: ConversationSnapshot) -> str:
    relation = snapshot.relationship
    if relation is None:
        return "(none)"
    return (
        f"closeness={relation.closeness}, trust={relation.trust}, "
        f"familiarity={relation.familiarity}, tension={relation.tension}, "
        f"summary={relation.summary or '(empty)'}"
    )


def _draw_targets(targets: list[TargetUserContext]) -> str:
    if not targets:
        return "(none)"
    lines = []
    for target in targets[:4]:
        aliases = ", ".join(target.aliases[:6]) or "(none)"
        lines.append(
            f"QQ:{target.user_id} status={target.resolution_status} aliases={aliases}\n"
            f"profile={_draw_profile(target.profile)}\n"
            f"facts=\n{_draw_facts(target.facts[:6])}"
        )
    return "\n\n".join(lines)


def _draw_join(lines: list[str]) -> str:
    return "\n".join(lines) if lines else "(none)"


def _draw_usage_date(now: int | None = None) -> str:
    timestamp = int(time.time() if now is None else now)
    return datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d")


def _maybe_cleanup_unused_stickers() -> None:
    if not config.stickers.enabled:
        return
    now = int(time.time())
    interval_seconds = config.stickers.cleanup_interval_hours * 60 * 60
    if not storage.claim_sticker_cleanup(interval_seconds, now=now):
        return
    unused_seconds = config.stickers.unused_ttl_hours * 60 * 60
    deleted_assets = storage.delete_unused_sticker_assets(unused_seconds, now=now)
    deleted_files = 0
    for asset in deleted_assets:
        if sticker_store.delete_saved_file(asset.local_path):
            deleted_files += 1
    if deleted_assets:
        logger.info(
            "Sticker cleanup deleted {} assets and {} files after {} unused hours",
            len(deleted_assets),
            deleted_files,
            config.stickers.unused_ttl_hours,
        )


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


async def _maybe_update_profiles(user_ids: list[str], force: bool = False) -> None:
    seen: set[str] = set()
    for raw_user_id in user_ids:
        user_id = str(raw_user_id).strip()
        if not user_id or user_id in seen:
            continue
        seen.add(user_id)
        if not force and not storage.should_update_user_profile(user_id, config.facts.profile_fact_threshold):
            continue
        facts = storage.list_user_facts(user_id, limit=0)
        if force and not facts:
            storage.clear_user_profile(user_id)
            continue
        draft = await pipeline.profile(user_id, facts, storage.get_user_profile(user_id))
        if draft is None:
            continue
        storage.maybe_update_user_profile(user_id, draft, facts, force=force)


async def _record_sticker_candidates(
    context: MessageContext,
    candidates: list[StickerCandidate],
) -> None:
    if not config.stickers.enabled or not candidates:
        return
    for candidate in candidates:
        existing = storage.find_existing_sticker_asset(context.group_id, candidate)
        if existing is not None:
            storage.upsert_sticker_asset(
                context,
                candidate,
                local_path=existing.local_path,
                sha256=existing.sha256,
            )
            continue
        saved = await sticker_store.save_candidate(context, candidate)
        if saved is None:
            continue
        asset = storage.upsert_sticker_asset(
            context,
            candidate,
            local_path=saved.local_path,
            sha256=saved.sha256,
        )
        if asset is not None and not _same_local_path(asset.local_path, saved.local_path):
            sticker_store.delete_saved_file(saved.local_path)


async def _send_group_reply(
    reply: str,
    sticker: StickerAssetRecord | None,
    reply_to_message_id: str | None,
    *,
    bot: Bot | None = None,
    context: MessageContext | None = None,
    decision: ParticipationDecision | None = None,
    allow_bubbles: bool = True,
) -> _ReplySendResult:
    parts = _prepare_reply_parts(reply, context, decision, allow_bubbles)
    if not parts and sticker is None:
        return _ReplySendResult(())

    delay_seconds = config.bot.reply_bubble_delay_seconds if allow_bubbles else 0
    queued = False
    sent_sticker: StickerAssetRecord | None = None
    if len(parts) > 1:
        for index, part in enumerate(parts[:-1]):
            first_reply_to = reply_to_message_id if index == 0 else None
            result = await _send_single_reply(part, None, first_reply_to, bot=bot, context=context)
            queued = queued or result.queued
            if result.queued:
                queued_sticker = await _queue_remaining_reply_parts(
                    parts[index + 1 :],
                    sticker,
                    bot=bot,
                    context=context,
                )
                return _ReplySendResult(parts, result.sticker or queued_sticker, queued=True)
            if delay_seconds > 0:
                await asyncio.sleep(delay_seconds)
        last_reply_to = None
        last_part = parts[-1]
    else:
        last_reply_to = reply_to_message_id
        last_part = parts[0] if parts else ""

    result = await _send_single_reply(last_part, None, last_reply_to, bot=bot, context=context)
    queued = queued or result.queued
    if result.queued:
        queued_sticker = await _queue_reply_sticker(
            sticker,
            bot=bot,
            context=context,
            source="group reply sticker",
            reason="previous reply text queued",
        )
        return _ReplySendResult(parts, queued_sticker, queued=True)

    sticker_result = await _send_reply_sticker(sticker, bot=bot, context=context)
    sent_sticker = sticker_result.sticker
    queued = queued or sticker_result.queued
    return _ReplySendResult(parts, sent_sticker, queued)


async def _send_single_reply(
    reply: str,
    sticker: StickerAssetRecord | None,
    reply_to_message_id: str | None,
    *,
    bot: Bot | None,
    context: MessageContext | None,
) -> _SingleReplySendResult:
    first_error: ActionFailed | None = None
    attempts = _reply_send_attempts(
        reply,
        sticker,
        reply_to_message_id,
    )
    for index, (message, attempted_sticker, used_reply) in enumerate(attempts):
        sticker_included = _message_contains_image(message)
        try:
            await _send_group_message(message, bot=bot, context=context)
        except ActionFailed as exc:
            if first_error is None:
                first_error = exc
            _log_reply_send_failure(
                exc,
                attempted_sticker if sticker_included else None,
                used_reply,
                reply_to_message_id,
            )
            continue
        except Exception as exc:
            if not _should_queue_send_error(exc):
                raise
            queued_sticker = attempted_sticker if sticker_included else None
            queued = await _queue_reply_attempts(
                attempts[index:],
                bot=bot,
                context=context,
                source="group reply",
                reason=_send_error_detail(exc),
            )
            if queued:
                logger.warning(
                    "Group reply send queued for group {} message {} after transient failure: {}",
                    context.group_id if context else "",
                    context.message_id if context else "",
                    exc,
                )
                return _SingleReplySendResult(queued_sticker, queued=True)
            raise
        return _SingleReplySendResult(attempted_sticker if sticker_included else None)
    if first_error is not None:
        raise first_error
    return _SingleReplySendResult()


async def _send_group_message(
    message: Message | str,
    *,
    bot: Bot | None,
    context: MessageContext | None,
) -> None:
    group_id = _onebot_group_id(context.group_id) if context is not None else None
    if bot is not None and group_id is not None:
        await bot.send_group_msg(group_id=group_id, message=message)
        return
    await group_message.send(message)


async def _queue_remaining_reply_parts(
    remaining_parts: tuple[str, ...],
    sticker: StickerAssetRecord | None,
    *,
    bot: Bot | None,
    context: MessageContext | None,
) -> StickerAssetRecord | None:
    if not remaining_parts:
        return None
    queued_sticker: StickerAssetRecord | None = None
    for part in remaining_parts:
        await _queue_reply_attempts(
            _reply_send_attempts(part, None, None),
            bot=bot,
            context=context,
            source="group reply bubble",
            reason="previous bubble queued",
        )
    if sticker is not None:
        queued_sticker = await _queue_reply_sticker(
            sticker,
            bot=bot,
            context=context,
            source="group reply sticker",
            reason="previous bubble queued",
        )
    return queued_sticker


async def _send_reply_sticker(
    sticker: StickerAssetRecord | None,
    *,
    bot: Bot | None,
    context: MessageContext | None,
) -> _SingleReplySendResult:
    if sticker is None:
        return _SingleReplySendResult()
    try:
        return await _send_single_reply("", sticker, None, bot=bot, context=context)
    except ActionFailed as exc:
        _log_reply_send_failure(exc, sticker, False, None)
        return _SingleReplySendResult()


async def _queue_reply_sticker(
    sticker: StickerAssetRecord | None,
    *,
    bot: Bot | None,
    context: MessageContext | None,
    source: str,
    reason: str,
) -> StickerAssetRecord | None:
    if sticker is None:
        return None
    queued = await _queue_reply_attempts(
        _reply_send_attempts("", sticker, None),
        bot=bot,
        context=context,
        source=source,
        reason=reason,
    )
    return sticker if queued else None


async def _queue_reply_attempts(
    attempts: list[tuple[Message | str, StickerAssetRecord | None, bool]],
    *,
    bot: Bot | None,
    context: MessageContext | None,
    source: str,
    reason: str,
) -> bool:
    if bot is None or context is None:
        return False
    queued_attempts = tuple(
        _QueuedSendAttempt(message, attempted_sticker if _message_contains_image(message) else None)
        for message, attempted_sticker, _ in attempts
    )
    return await _queue_outbound_group_attempts(
        bot,
        context.group_id,
        queued_attempts,
        source=source,
        reason=reason,
    )


def _prepare_reply_parts(
    reply: str,
    context: MessageContext | None,
    decision: ParticipationDecision | None,
    allow_bubbles: bool,
) -> tuple[str, ...]:
    text = str(reply or "").strip()
    if not text:
        return ()
    if not allow_bubbles:
        return (text,)

    settings = settings_from_bot_config(config.bot)
    if context is not None and decision is not None:
        recent_replies = storage.get_recent_bot_reply_texts(
            context.group_id,
            config.bot.reply_emoji_cooldown_messages,
        )
        text = style_reply_text(
            text,
            settings,
            action=decision.action,
            value_type=decision.value_type,
            trigger_text=context.plain_text,
            recent_bot_replies=recent_replies,
        )
    return split_reply_bubbles(text, settings)


def _reply_send_attempts(
    reply: str,
    sticker: StickerAssetRecord | None,
    reply_to_message_id: str | None,
) -> list[tuple[Message | str, StickerAssetRecord | None, bool]]:
    reply_to = str(reply_to_message_id or "").strip() or None
    sticker_refs = _sticker_file_refs(sticker) if sticker is not None else ()
    requested: list[tuple[str | None, StickerAssetRecord | None, str]] = []
    if reply_to:
        if sticker is not None:
            requested.extend((reply_to, sticker, file_ref) for file_ref in sticker_refs)
            requested.append((reply_to, None, ""))
            requested.extend((None, sticker, file_ref) for file_ref in sticker_refs)
        else:
            requested.append((reply_to, None, ""))
        requested.append((None, None, ""))
    else:
        if sticker is not None:
            requested.extend((None, sticker, file_ref) for file_ref in sticker_refs)
            requested.append((None, None, ""))
        else:
            requested.append((None, None, ""))

    attempts: list[tuple[Message | str, StickerAssetRecord | None, bool]] = []
    seen: set[tuple[bool, int | None, str]] = set()
    for attempt_reply_to, attempt_sticker, attempt_file_ref in requested:
        if not reply and attempt_sticker is None:
            continue
        key = (
            bool(attempt_reply_to),
            attempt_sticker.id if attempt_sticker is not None else None,
            attempt_file_ref if attempt_sticker is not None else "",
        )
        if key in seen:
            continue
        seen.add(key)
        attempts.append(
            (
                _reply_message(reply, attempt_sticker, attempt_reply_to, attempt_file_ref),
                attempt_sticker,
                bool(attempt_reply_to),
            )
        )
    return attempts


def _reply_message(
    reply: str,
    sticker: StickerAssetRecord | None,
    reply_to_message_id: str | None = None,
    file_ref: str = "",
) -> Message | str:
    text_message = _reply_text_message(reply)
    if sticker is not None and not file_ref:
        file_ref = _sticker_file_ref(sticker)
    reply_to = str(reply_to_message_id or "").strip()
    if not file_ref and not reply_to:
        return text_message
    message = Message()
    if reply_to:
        message += MessageSegment.reply(reply_to)
    if reply:
        if isinstance(text_message, Message):
            message += text_message
        else:
            message += MessageSegment.text(reply)
        if file_ref:
            message += MessageSegment.text("\n")
    if file_ref:
        message += MessageSegment.image(file=file_ref)
    return message


def _log_reply_send_failure(
    exc: ActionFailed,
    sticker: StickerAssetRecord | None,
    used_reply: bool,
    reply_to_message_id: str | None,
) -> None:
    if sticker is not None:
        logger.warning(
            "Group reply send failed for asset #{} ({}) reply_to={}: {}",
            sticker.id,
            sticker.url or sticker.local_path,
            reply_to_message_id if used_reply else "",
            exc,
        )
        return
    if used_reply:
        logger.warning("Group reply send failed reply_to={}: {}", reply_to_message_id, exc)


def _reply_text_message(reply: str) -> Message | str:
    parts = parse_outgoing_mention_parts(reply)
    if not any(part.kind == "at" for part in parts):
        return reply
    message = Message()
    for part in parts:
        if part.kind == "at":
            message += MessageSegment.at(part.user_id)
        elif part.text:
            message += MessageSegment.text(part.text)
    return message


def _message_contains_image(message: Message | str) -> bool:
    return isinstance(message, Message) and any(segment.type == "image" for segment in message)


def _message_has_reply(message: Message) -> bool:
    return any(segment.type == "reply" for segment in message)


def _start_outbound_retry_worker() -> None:
    global _outbound_retry_worker
    if not config.bot.send_retry_enabled:
        return
    if _outbound_retry_worker is not None and not _outbound_retry_worker.done():
        return
    _outbound_retry_worker = asyncio.create_task(_outbound_retry_loop())


async def _stop_outbound_retry_worker() -> None:
    global _outbound_retry_worker
    task = _outbound_retry_worker
    _outbound_retry_worker = None
    if task is None or task.done():
        return
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task


async def _outbound_retry_loop() -> None:
    while True:
        await asyncio.sleep(_OUTBOUND_RETRY_WORKER_INTERVAL_SECONDS)
        for bot in list(driver.bots.values()):
            if isinstance(bot, Bot):
                await _flush_outbound_queue(bot, "retry worker")


async def _outbound_queue_size(bot_self_id: str | None = None) -> int:
    async with _outbound_queue_lock:
        if bot_self_id is None:
            return len(_outbound_queue)
        return sum(1 for item in _outbound_queue if item.bot_self_id == str(bot_self_id))


async def _queue_outbound_group_attempts(
    bot: Bot,
    group_id: str,
    send_attempts: tuple[_QueuedSendAttempt, ...],
    *,
    source: str,
    reason: str,
) -> bool:
    if not config.bot.send_retry_enabled or not send_attempts:
        return False
    now = time.time()
    async with _outbound_queue_lock:
        _drop_expired_outbound_messages_locked(now)
        while len(_outbound_queue) >= config.bot.send_retry_queue_limit:
            dropped = _outbound_queue.pop(0)
            logger.warning(
                "Dropping queued outbound message #{} for bot {} group {}: queue limit reached",
                dropped.id,
                dropped.bot_self_id,
                dropped.group_id,
            )
        queued = _QueuedOutboundMessage(
            id=next(_outbound_queue_ids),
            bot_self_id=str(bot.self_id),
            group_id=str(group_id),
            send_attempts=send_attempts,
            created_at=now,
            next_attempt_at=now,
            source=source,
            reason=reason,
        )
        _outbound_queue.append(queued)
        logger.warning(
            "Queued outbound message #{} for bot {} group {} from {} after {} (queue_size={})",
            queued.id,
            queued.bot_self_id,
            queued.group_id,
            source,
            reason,
            len(_outbound_queue),
        )
    return True


async def _flush_outbound_queue(bot: Bot, reason: str) -> None:
    if not config.bot.send_retry_enabled:
        return
    async with _outbound_flush_lock:
        while True:
            queued = await _pop_next_due_outbound_message(bot.self_id)
            if queued is None:
                return
            try:
                status, sticker, error = await _try_send_queued_group_message(bot, queued)
            except Exception as exc:
                logger.warning(
                    "Dropping queued outbound message #{} after unexpected send error: {}",
                    queued.id,
                    exc,
                )
                continue
            if status == "sent":
                logger.info(
                    "Flushed queued outbound message #{} for bot {} group {} via {}",
                    queued.id,
                    queued.bot_self_id,
                    queued.group_id,
                    reason,
                )
                if sticker is not None:
                    storage.record_sticker_sent(sticker.id, usage_date=_draw_usage_date())
                    _maybe_cleanup_unused_stickers()
                continue
            if status == "retry":
                await _requeue_outbound_message(queued, error)
                return


async def _pop_next_due_outbound_message(bot_self_id: str) -> _QueuedOutboundMessage | None:
    now = time.time()
    async with _outbound_queue_lock:
        _drop_expired_outbound_messages_locked(now)
        for index, queued in enumerate(_outbound_queue):
            if queued.bot_self_id == str(bot_self_id) and queued.next_attempt_at <= now:
                return _outbound_queue.pop(index)
    return None


async def _try_send_queued_group_message(
    bot: Bot,
    queued: _QueuedOutboundMessage,
) -> tuple[str, StickerAssetRecord | None, BaseException | None]:
    group_id = _onebot_group_id(queued.group_id)
    if group_id is None:
        logger.warning(
            "Dropping queued outbound message #{}: invalid group id {}",
            queued.id,
            queued.group_id,
        )
        return ("drop", None, None)

    first_action_error: ActionFailed | None = None
    for attempt in queued.send_attempts:
        try:
            await bot.send_group_msg(group_id=group_id, message=attempt.message)
        except ActionFailed as exc:
            if first_action_error is None:
                first_action_error = exc
            logger.warning(
                "Queued outbound message #{} action attempt failed for group {}: {}",
                queued.id,
                queued.group_id,
                exc,
            )
            continue
        except Exception as exc:
            if _should_queue_send_error(exc):
                return ("retry", None, exc)
            raise
        return ("sent", attempt.sticker, None)

    logger.warning(
        "Dropping queued outbound message #{} after all action attempts failed: {}",
        queued.id,
        first_action_error,
    )
    return ("drop", None, first_action_error)


async def _requeue_outbound_message(
    queued: _QueuedOutboundMessage,
    error: BaseException | None,
) -> None:
    now = time.time()
    queued.attempts += 1
    queued.reason = _send_error_detail(error) if error is not None else queued.reason
    if (
        queued.attempts >= config.bot.send_retry_max_attempts
        or now - queued.created_at > config.bot.send_retry_max_age_seconds
    ):
        logger.warning(
            "Dropping queued outbound message #{} for bot {} group {} after {} attempts: {}",
            queued.id,
            queued.bot_self_id,
            queued.group_id,
            queued.attempts,
            queued.reason,
        )
        return
    queued.next_attempt_at = now + _outbound_retry_delay(queued.attempts)
    async with _outbound_queue_lock:
        _outbound_queue.append(queued)
    logger.warning(
        "Queued outbound message #{} retry {}/{} in {:.1f}s: {}",
        queued.id,
        queued.attempts,
        config.bot.send_retry_max_attempts,
        queued.next_attempt_at - now,
        queued.reason,
    )


def _drop_expired_outbound_messages_locked(now: float) -> None:
    kept: list[_QueuedOutboundMessage] = []
    for queued in _outbound_queue:
        expired = now - queued.created_at > config.bot.send_retry_max_age_seconds
        exhausted = queued.attempts >= config.bot.send_retry_max_attempts
        if expired or exhausted:
            logger.warning(
                "Dropping queued outbound message #{} for bot {} group {}: {}",
                queued.id,
                queued.bot_self_id,
                queued.group_id,
                "expired" if expired else "attempts exhausted",
            )
            continue
        kept.append(queued)
    _outbound_queue[:] = kept


def _outbound_retry_delay(attempts: int) -> float:
    base = config.bot.send_retry_base_delay_seconds
    maximum = config.bot.send_retry_max_delay_seconds
    return min(maximum, base * (2 ** max(0, attempts - 1)))


def _onebot_group_id(group_id: str) -> int | None:
    try:
        return int(str(group_id).strip())
    except ValueError:
        return None


def _should_queue_send_error(exc: BaseException) -> bool:
    if isinstance(exc, (ApiNotAvailable, WebSocketClosed, ConnectionClosed)):
        return True
    if isinstance(exc, NetworkError):
        detail = _send_error_detail(exc).lower()
        return "timeout" not in detail
    return isinstance(exc, (ConnectionError, OSError))


def _send_error_detail(exc: BaseException) -> str:
    message = getattr(exc, "msg", None) or str(exc) or repr(exc)
    return f"{type(exc).__name__}: {message}"


def _sticker_file_ref(sticker: StickerAssetRecord) -> str:
    return sticker_file_ref(sticker)


def _sticker_file_refs(sticker: StickerAssetRecord) -> tuple[str, ...]:
    return sticker_file_refs(sticker)


def _same_local_path(left: str, right: str) -> bool:
    try:
        return Path(left).resolve() == Path(right).resolve()
    except OSError:
        return str(left).strip() == str(right).strip()


def _reply_record_text(reply: str | Iterable[str] | None, sticker: StickerAssetRecord | None) -> str:
    if isinstance(reply, str) or reply is None:
        text = reply or ""
    else:
        text = "\n".join(part for part in reply if str(part or "").strip())
    if sticker is None:
        return text
    label = sticker.usage or sticker.description or sticker.local_path or sticker.url
    marker = f"[表情 #{sticker.id}: {label}]"
    return f"{text}\n{marker}".strip()


def _final_qa_blocked_decision(decision: Any, reason: str) -> Any:
    return replace(
        decision,
        action="observe",
        reason=f"{decision.reason}; final QA blocked reply: {reason}",
        score=min(decision.score, 0.49),
    )


async def _build_context(bot: Bot, event: GroupMessageEvent) -> MessageContext:
    top_level_text, _ = render_message_text_and_mentions(event.message, str(bot.self_id))
    plain_text, mentions = await render_message_text_and_mentions_with_forwards(
        event.message,
        str(bot.self_id),
        _fetch_forward_message(bot),
        reply_fetcher=_fetch_reply_message(bot, event),
    )
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
        is_direct=_is_direct_message(bot, event, top_level_text),
        bot_mentioned=_is_bot_mentioned(bot, event, top_level_text),
        timestamp=int(getattr(event, "time", 0) or time.time()),
        attachments=_extract_attachments(event),
        mentions=mentions,
    )


def _fetch_forward_message(bot: Bot):
    async def _fetch(forward_id: str) -> Any:
        try:
            return await bot.get_forward_msg(id=forward_id)
        except Exception as exc:
            logger.warning("Failed to fetch forwarded message {}: {}", forward_id, exc)
            return None

    return _fetch


def _fetch_reply_message(bot: Bot, event: GroupMessageEvent):
    async def _fetch(message_id: str) -> Any:
        event_reply = _event_reply_payload(event, message_id)
        if event_reply is not None:
            return event_reply
        target_message_id = _onebot_message_id(message_id)
        if target_message_id is None:
            return None
        try:
            return await bot.get_msg(message_id=target_message_id)
        except Exception as exc:
            logger.warning("Failed to fetch quoted message {}: {}", message_id, exc)
            return None

    return _fetch


def _event_reply_payload(event: GroupMessageEvent, message_id: str) -> dict[str, Any] | None:
    reply = getattr(event, "reply", None)
    if reply is None:
        return None
    reply_message_id = str(
        getattr(reply, "message_id", "") or getattr(reply, "id", "") or ""
    ).strip()
    if reply_message_id and message_id and reply_message_id != str(message_id):
        return None
    message = getattr(reply, "message", None)
    if message is None:
        message = getattr(reply, "content", None)
    raw_message = str(getattr(reply, "raw_message", "") or "")
    sender = getattr(reply, "sender", None)
    return {
        "message_id": reply_message_id or str(message_id),
        "message": message if message is not None else raw_message,
        "raw_message": raw_message,
        "sender": {
            "user_id": _sender_field(sender, "user_id"),
            "nickname": _sender_field(sender, "nickname") or _sender_field(sender, "card"),
            "card": _sender_field(sender, "card"),
        },
    }


def _is_direct_message(bot: Bot, event: GroupMessageEvent, plain_text: str) -> bool:
    if _has_explicit_bot_at(bot, event):
        return True
    if _is_reply_to_bot(bot, event):
        return True
    return looks_like_bot_address(plain_text, config.bot.nicknames)


def _is_bot_mentioned(bot: Bot, event: GroupMessageEvent, plain_text: str) -> bool:
    return (
        _has_explicit_bot_at(bot, event)
        or _is_reply_to_bot(bot, event)
        or text_mentions_bot_name(plain_text, config.bot.nicknames)
    )


def _has_explicit_bot_at(bot: Bot, event: GroupMessageEvent) -> bool:
    for segment in event.message:
        if segment.type == "at" and str(segment.data.get("qq")) == str(bot.self_id):
            return True
    return False


def _is_reply_to_bot(bot: Bot, event: GroupMessageEvent) -> bool:
    reply = getattr(event, "reply", None)
    if reply is None:
        return False
    sender = getattr(reply, "sender", None)
    return str(_sender_field(sender, "user_id")) == str(bot.self_id)


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
        "#bot ignore list|add <qq_id>|remove <qq_id>\n"
        "#bot memory lexicon [term]|pending|conflicts|approve <id>|reject <id>\n"
        "#bot facts user <qq_id>|pending|approve <id>|reject <id>|forget <id>\n"
        "#bot profile <qq_id>\n"
        "#bot stickers list [数量]|enable <id>|disable <id>|delete <id>\n"
        "#bot persona show|self [pending|conflicts|approve <id>|reject <id>|forget <id>]\n"
        "#bot llm status|test [prompt]\n"
        "#bot token\n"
        "#bot why\n"
        "#bot relation <qq_id>|top [数量]|rank [数量]\n"
        "#bot forget <memory_id>"
    )


def _memory_help_text() -> str:
    return (
        "用法：\n"
        "#bot memory lexicon [term]\n"
        "#bot memory pending\n"
        "#bot memory conflicts\n"
        "#bot memory approve <memory_id>\n"
        "#bot memory reject <memory_id>"
    )


def _facts_help_text() -> str:
    return (
        "用法：\n"
        "#bot facts user <qq_id>\n"
        "#bot facts pending\n"
        "#bot facts approve <fact_id>\n"
        "#bot facts reject <fact_id>\n"
        "#bot facts forget <fact_id>\n"
        "#bot profile <qq_id>"
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
