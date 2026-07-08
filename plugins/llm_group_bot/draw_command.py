from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from loguru import logger
from nonebot.adapters.onebot.v11 import Bot, GroupMessageEvent, Message
from nonebot.params import CommandArg

from plugins.llm_group_bot.draw_prompts import compose_draw_prompt as _compose_draw_prompt_impl
from plugins.llm_group_bot.generated_images import send_generated_image as _send_generated_image
from qq_llm_bot.draw_images import prepare_draw_reference_images
from qq_llm_bot.draw_reference import DrawIntentPlanner
from qq_llm_bot.image_generation import GeneratedImageStore
from qq_llm_bot.llm import is_llm_configured
from qq_llm_bot.models import ConversationSnapshot, MessageAttachment, MessageContext
from qq_llm_bot.onebot_context import (
    build_message_context,
    event_reply_payload,
    onebot_message_id,
    reply_message_fetcher,
)
from qq_llm_bot.onebot_messages import (
    image_attachments_from_payload as _image_attachments_from_payload,
    reply_segment_ids as _reply_segment_ids,
)
from qq_llm_bot.outbound_queue import OutboundGroupSendQueue

_PROCESSING_ACK_EMOJI_ID = "124"  # QQ [OK]
_DRAW_FAILURE_REPLY = "哎呀，图片不见了，我的我的~"

config: Any = None
storage: Any = None
llm: Any = None
outbound_queue: OutboundGroupSendQueue | None = None
maintenance: Any = None
draw_cmd: Any = None
generated_image_store: GeneratedImageStore | None = None
draw_intent_planner: DrawIntentPlanner | None = None
_finish_command_callback: Callable[[Any, Message | str], Awaitable[None]] | None = None
_remember_command_reply_callback: Callable[[GroupMessageEvent], None] | None = None


def configure(
    *,
    config_: Any,
    storage_: Any,
    llm_: Any,
    outbound_queue_: OutboundGroupSendQueue,
    maintenance_: Any,
    draw_cmd_: Any,
    finish_command: Callable[[Any, Message | str], Awaitable[None]],
    remember_command_reply: Callable[[GroupMessageEvent], None],
) -> None:
    global config, storage, llm, outbound_queue, maintenance, draw_cmd
    global generated_image_store, draw_intent_planner
    global _finish_command_callback, _remember_command_reply_callback

    config = config_
    storage = storage_
    llm = llm_
    outbound_queue = outbound_queue_
    maintenance = maintenance_
    draw_cmd = draw_cmd_
    generated_image_store = GeneratedImageStore(config)
    draw_intent_planner = DrawIntentPlanner(llm, bot_names=tuple(config.bot.nicknames))
    _finish_command_callback = finish_command
    _remember_command_reply_callback = remember_command_reply


async def handle_draw_command(
    bot: Bot,
    event: GroupMessageEvent,
    args: Message = CommandArg(),
) -> None:
    if _remember_command_reply_callback is None:  # pragma: no cover - setup invariant
        raise RuntimeError("draw command handlers are not configured")
    _remember_command_reply_callback(event)

    group_id = str(event.group_id)
    user_id = str(event.user_id)
    is_admin = storage.is_admin(user_id)
    if not storage.is_group_enabled(group_id):
        return
    if storage.is_user_ignored(user_id):
        return
    if not config.image_generation.enabled:
        await _finish_command("生图功能未开启。")
    if not is_llm_configured(config.llm):
        await _finish_command("生图模型未配置，请检查 provider/base_url/model/key。")

    if not config.llm.routing.image_generation_model.strip():
        await _finish_command("llm.router.image_generation_model is required for #draw.")

    prompt = args.extract_plain_text().strip()
    if not prompt:
        await _finish_command("用法：#draw <图片描述>")
    if len(prompt) > config.image_generation.max_prompt_chars:
        await _finish_command(
            f"图片描述太长了，最大 {config.image_generation.max_prompt_chars} 个字符。"
        )

    relation = storage.get_relationship(group_id, user_id)
    if not is_admin and relation.trust < config.image_generation.min_trust:
        await _finish_command(
            f"现在还不能生图：当前 trust={relation.trust}，需要 >= {config.image_generation.min_trust}。"
        )

    usage_date = maintenance.usage_date()
    used_count = storage.count_image_generation_usage(user_id, usage_date)
    if not is_admin and used_count >= config.image_generation.daily_limit:
        await _finish_command(
            f"今天的生图次数已经用完了（{used_count}/{config.image_generation.daily_limit}）。"
        )

    await _acknowledge_processing(bot, event.message_id, "draw")

    context = await build_message_context(bot, event, bot_names=config.bot.nicknames)
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
        await _finish_command(reference_images.error)

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
        await _finish_command(_draw_failure_reply("提示词整理失败", is_admin))

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
        await _finish_command(_draw_failure_reply("图片保存失败", is_admin))

    if not await _send_generated_image(
        bot,
        context,
        saved,
        command_sender=draw_cmd,
        outbound_queue=outbound_queue,
    ):
        detail = saved.local_path or saved.url or saved.file_ref
        await _finish_command(
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


async def _finish_command(message: Message | str) -> None:
    if _finish_command_callback is None or draw_cmd is None:  # pragma: no cover - setup invariant
        raise RuntimeError("draw command handlers are not configured")
    await _finish_command_callback(draw_cmd, message)


async def _acknowledge_processing(bot: Bot, message_id: object, reason: str) -> None:
    target_message_id = onebot_message_id(message_id)
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


async def _compose_draw_prompt(
    context: MessageContext,
    snapshot: ConversationSnapshot,
    draw_request: str,
    reference_image_count: int = 0,
) -> str | None:
    return await _compose_draw_prompt_impl(
        config,
        llm,
        draw_intent_planner,
        context,
        snapshot,
        draw_request,
        reference_image_count,
    )


async def _draw_reference_image_attachments(
    bot: Bot,
    event: GroupMessageEvent,
    context: MessageContext,
) -> list[MessageAttachment]:
    attachments = list(context.attachments)
    fetch_reply = reply_message_fetcher(bot, event)
    reply_ids = _reply_segment_ids(event.message)
    if reply_ids:
        for reply_id in reply_ids:
            payload = await fetch_reply(reply_id)
            attachments.extend(_image_attachments_from_payload(payload))
    else:
        payload = event_reply_payload(event, "")
        attachments.extend(_image_attachments_from_payload(payload))
    return attachments


def _draw_failure_reply(stage: str, is_admin: bool, detail: str = "") -> str:
    if not is_admin:
        return _DRAW_FAILURE_REPLY
    suffix = f"\n详情：{detail}" if detail else ""
    return f"{_DRAW_FAILURE_REPLY}\n管理员调试：{stage}{suffix}"
