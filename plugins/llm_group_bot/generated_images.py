from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

from loguru import logger
from nonebot.adapters.onebot.v11 import Bot, Message, MessageSegment
from nonebot.adapters.onebot.v11.exception import ActionFailed

from qq_llm_bot.models import MessageContext
from qq_llm_bot.outbound_queue import (
    OutboundGroupSendQueue,
    QueuedSendAttempt,
    send_error_detail,
    should_queue_send_error,
)


async def send_generated_image(
    bot: Bot,
    context: MessageContext,
    saved: Any,
    *,
    command_sender: Any,
    outbound_queue: OutboundGroupSendQueue,
) -> bool:
    send_refs = _generated_image_send_refs(saved)
    if not send_refs:
        logger.warning(
            "Generated image has no sendable non-local ref for {}",
            saved.local_path or saved.url or saved.file_ref,
        )
        return False

    prepared_attempts = [
        (ref_kind, include_reply, _generated_image_message(context.message_id if include_reply else "", file_ref))
        for ref_kind, file_ref in send_refs
        for include_reply in (True, False)
    ]
    for index, (ref_kind, include_reply, message) in enumerate(prepared_attempts):
        try:
            await command_sender.send(message)
            if ref_kind != "file" or not include_reply:
                logger.info(
                    "Generated image send succeeded via {} reply={} for {}",
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
            if not should_queue_send_error(exc):
                raise
            queued = await outbound_queue.queue_group_attempts(
                bot,
                context.group_id,
                tuple(
                    QueuedSendAttempt(attempt_message)
                    for _, _, attempt_message in prepared_attempts[index:]
                ),
                source="draw image",
                reason=send_error_detail(exc),
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


def _generated_image_send_refs(saved: Any) -> list[tuple[str, str]]:
    send_refs: list[tuple[str, str]] = []
    base64_ref = _generated_image_base64_ref(saved.local_path)
    _append_send_ref(send_refs, "base64", base64_ref)
    _append_send_ref(send_refs, "url", saved.url)
    _append_send_ref(send_refs, "file", _non_local_image_ref(saved.file_ref))
    return send_refs


def _append_send_ref(send_refs: list[tuple[str, str]], ref_kind: str, ref: str) -> None:
    cleaned = str(ref or "").strip()
    if cleaned and cleaned not in {existing_ref for _, existing_ref in send_refs}:
        send_refs.append((ref_kind, cleaned))


def _non_local_image_ref(file_ref: str) -> str:
    cleaned = str(file_ref or "").strip()
    if cleaned.lower().startswith(("http://", "https://", "base64://")):
        return cleaned
    return ""


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
