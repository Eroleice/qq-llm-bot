from __future__ import annotations

from qq_llm_bot.storage_message_activity_reads import get_recent_activity_counts
from qq_llm_bot.storage_message_recent_reads import (
    get_focused_recent_messages,
    get_recent_bot_reply_texts,
    get_recent_bot_reply_to_user,
    get_recent_messages,
)

__all__ = [
    "get_focused_recent_messages",
    "get_recent_activity_counts",
    "get_recent_bot_reply_texts",
    "get_recent_bot_reply_to_user",
    "get_recent_image_descriptions",
    "get_recent_messages",
]

def get_recent_image_descriptions(
    storage: object,
    group_id: str,
    limit: int = 8,
) -> list[str]:
    with storage._connect() as conn:
        rows = conn.execute(
            """
            SELECT sender_name, messages.user_id, message_attachments.summary
            FROM message_attachments
            LEFT JOIN messages
              ON messages.group_id = message_attachments.group_id
             AND messages.message_id = message_attachments.message_id
            WHERE message_attachments.group_id = ?
              AND message_attachments.attachment_type = 'image'
              AND message_attachments.summary != ''
            ORDER BY message_attachments.id DESC
            LIMIT ?
            """,
            (str(group_id), int(limit)),
        ).fetchall()
    lines = []
    for row in reversed(rows):
        name = str(row["sender_name"] or row["user_id"] or "unknown")
        lines.append(f"{name}: [图片] {row['summary']}")
    return lines
