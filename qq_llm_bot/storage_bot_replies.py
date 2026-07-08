from __future__ import annotations

import time
from collections.abc import Iterable

from qq_llm_bot.models import MessageContext
from qq_llm_bot.storage_message_records import record_message


def record_bot_reply(storage: object, group_id: str, bot_id: str, reply: str) -> None:
    record_bot_reply_parts(storage, group_id, bot_id, [reply])


def record_bot_reply_parts(
    storage: object,
    group_id: str,
    bot_id: str,
    replies: Iterable[str],
) -> None:
    now = int(time.time())
    clean_replies = [str(reply or "").strip() for reply in replies if str(reply or "").strip()]
    for index, reply in enumerate(clean_replies, start=1):
        record_message(
            storage,
            MessageContext(
                group_id=str(group_id),
                user_id=str(bot_id),
                message_id=f"bot-{now}-{index}",
                plain_text=reply,
                raw_message=reply,
                sender_name="bot",
                sender_role="bot",
                timestamp=now,
            ),
        )
