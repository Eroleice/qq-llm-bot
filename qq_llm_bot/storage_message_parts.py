from __future__ import annotations

import sqlite3
import time

from qq_llm_bot.models import MessageContext


def record_attachments(conn: sqlite3.Connection, context: MessageContext) -> None:
    if not context.attachments:
        return
    now = context.timestamp or int(time.time())
    conn.executemany(
        """
        INSERT INTO message_attachments (
            time, group_id, user_id, message_id, attachment_type,
            file, url, summary, raw_data
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                now,
                context.group_id,
                context.user_id,
                context.message_id,
                attachment.attachment_type,
                attachment.file,
                attachment.url,
                attachment.summary,
                attachment.raw_data,
            )
            for attachment in context.attachments
        ],
    )


def record_mentions(conn: sqlite3.Connection, context: MessageContext) -> None:
    if not context.mentions:
        return
    now = context.timestamp or int(time.time())
    conn.executemany(
        """
        INSERT INTO message_mentions (
            time, group_id, user_id, message_id, mentioned_user_id,
            display_name, is_bot, raw_data
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                now,
                context.group_id,
                context.user_id,
                context.message_id,
                mention.user_id,
                mention.display_name,
                1 if mention.is_bot else 0,
                mention.raw_data,
            )
            for mention in context.mentions
        ],
    )
