from __future__ import annotations

import time

from qq_llm_bot.storage_record_serializers import _format_message_context_line


def get_recent_bot_reply_texts(
    storage: object,
    group_id: str,
    limit: int = 10,
) -> list[str]:
    with storage._connect() as conn:
        rows = conn.execute(
            """
            SELECT plain_text
            FROM messages
            WHERE group_id = ?
              AND sender_role = 'bot'
              AND plain_text != ''
            ORDER BY id DESC
            LIMIT ?
            """,
            (str(group_id), max(1, int(limit))),
        ).fetchall()
    return [str(row["plain_text"] or "") for row in rows if str(row["plain_text"] or "").strip()]


def get_recent_messages(storage: object, group_id: str, limit: int = 12) -> list[str]:
    with storage._connect() as conn:
        rows = conn.execute(
            """
            SELECT sender_name, user_id, plain_text
            FROM messages
            WHERE group_id = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (str(group_id), limit),
        ).fetchall()
    lines = []
    for row in reversed(rows):
        line = _format_message_context_line(row)
        if line:
            lines.append(line)
    return lines


def get_focused_recent_messages(
    storage: object,
    group_id: str,
    user_id: str,
    speaker_limit: int = 5,
    other_limit: int = 10,
) -> tuple[list[str], list[str]]:
    group_id = str(group_id)
    user_id = str(user_id)
    with storage._connect() as conn:
        speaker_rows = conn.execute(
            """
            SELECT sender_name, user_id, plain_text
            FROM messages
            WHERE group_id = ?
              AND user_id = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (group_id, user_id, int(speaker_limit)),
        ).fetchall()
        other_rows = conn.execute(
            """
            SELECT sender_name, user_id, plain_text
            FROM messages
            WHERE group_id = ?
              AND user_id != ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (group_id, user_id, int(other_limit)),
        ).fetchall()

    speaker_lines = [
        line for row in reversed(speaker_rows) if (line := _format_message_context_line(row))
    ]
    other_lines = [
        line for row in reversed(other_rows) if (line := _format_message_context_line(row))
    ]
    return speaker_lines, other_lines


def get_recent_bot_reply_to_user(
    storage: object,
    group_id: str,
    user_id: str,
    window_seconds: int,
) -> tuple[str, int]:
    now = int(time.time())
    cutoff = now - max(1, int(window_seconds))
    with storage._connect() as conn:
        row = conn.execute(
            """
            SELECT time, reply
            FROM bot_decisions
            WHERE group_id = ?
              AND user_id = ?
              AND reply != ''
              AND time >= ?
            ORDER BY id DESC
            LIMIT 1
            """,
            (str(group_id), str(user_id), cutoff),
        ).fetchone()
    if row is None:
        return "", 0
    reply = str(row["reply"] or "").strip()
    if not reply:
        return "", 0
    return reply, max(0, now - int(row["time"]))
