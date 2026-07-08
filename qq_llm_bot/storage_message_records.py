from __future__ import annotations

import time

from qq_llm_bot.models import MessageContext
from qq_llm_bot.storage_message_parts import record_attachments, record_mentions
from qq_llm_bot.storage_message_profiles import upsert_user_profile


def record_message(storage: object, context: MessageContext) -> None:
    now = context.timestamp or int(time.time())
    with storage._connect() as conn:
        conn.execute(
            """
            INSERT INTO messages (
                time, group_id, user_id, message_id, raw_message, plain_text,
                sender_name, sender_role
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                now,
                context.group_id,
                context.user_id,
                context.message_id,
                context.raw_message,
                context.plain_text,
                context.sender_name,
                context.sender_role,
            ),
        )
        upsert_user_profile(conn, context, now)
        record_attachments(conn, context)
        record_mentions(conn, context)
