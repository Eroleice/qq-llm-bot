from __future__ import annotations

from typing import Any

from qq_llm_bot.storage_dashboard_final_qa import (
    list_dashboard_final_qa_blocks as list_dashboard_final_qa_blocks,
)


def list_dashboard_messages(
    storage: Any,
    group_id: str = "",
    user_id: str = "",
    start_time: int | None = None,
    end_time: int | None = None,
    limit: int = 200,
) -> list[dict[str, object]]:
    where = []
    params: list[object] = []
    if group_id:
        where.append("group_id = ?")
        params.append(str(group_id))
    if user_id:
        where.append("user_id = ?")
        params.append(str(user_id))
    if start_time is not None:
        where.append("time >= ?")
        params.append(int(start_time))
    if end_time is not None:
        where.append("time < ?")
        params.append(int(end_time))
    where_sql = "WHERE " + " AND ".join(where) if where else ""

    with storage._connect() as conn:
        rows = conn.execute(
            f"""
            SELECT id, time, group_id, user_id, message_id, raw_message, plain_text,
                   sender_name, sender_role
            FROM messages
            {where_sql}
            ORDER BY time DESC, id DESC
            LIMIT ?
            """,
            [*params, int(limit)],
        ).fetchall()
    messages = [
        {
            "id": int(row["id"]),
            "time": int(row["time"]),
            "group_id": str(row["group_id"]),
            "user_id": str(row["user_id"]),
            "message_id": str(row["message_id"]),
            "raw_message": str(row["raw_message"]),
            "plain_text": str(row["plain_text"]),
            "sender_name": str(row["sender_name"] or ""),
            "sender_role": str(row["sender_role"] or ""),
            "attachments": [],
            "mentions": [],
        }
        for row in rows
    ]
    attach_dashboard_attachments(storage, messages)
    attach_dashboard_mentions(storage, messages)
    return messages

def attach_dashboard_attachments(storage: Any, messages: list[dict[str, object]]) -> None:
    if not messages:
        return
    pairs = [(str(item["group_id"]), str(item["message_id"])) for item in messages]
    clauses = " OR ".join("(group_id = ? AND message_id = ?)" for _ in pairs)
    params: list[object] = []
    for group_id, message_id in pairs:
        params.extend([group_id, message_id])
    with storage._connect() as conn:
        rows = conn.execute(
            f"""
            SELECT group_id, message_id, attachment_type, file, url, summary, raw_data
            FROM message_attachments
            WHERE {clauses}
            ORDER BY id
            """,
            params,
        ).fetchall()
    by_key: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in rows:
        key = (str(row["group_id"]), str(row["message_id"]))
        by_key.setdefault(key, []).append(
            {
                "attachment_type": str(row["attachment_type"]),
                "file": str(row["file"] or ""),
                "url": str(row["url"] or ""),
                "summary": str(row["summary"] or ""),
                "raw_data": str(row["raw_data"] or ""),
            }
        )
    for item in messages:
        key = (str(item["group_id"]), str(item["message_id"]))
        item["attachments"] = by_key.get(key, [])

def attach_dashboard_mentions(storage: Any, messages: list[dict[str, object]]) -> None:
    if not messages:
        return
    pairs = [(str(item["group_id"]), str(item["message_id"])) for item in messages]
    clauses = " OR ".join("(group_id = ? AND message_id = ?)" for _ in pairs)
    params: list[object] = []
    for group_id, message_id in pairs:
        params.extend([group_id, message_id])
    with storage._connect() as conn:
        rows = conn.execute(
            f"""
            SELECT group_id, message_id, mentioned_user_id, display_name, is_bot, raw_data
            FROM message_mentions
            WHERE {clauses}
            ORDER BY id
            """,
            params,
        ).fetchall()
    by_key: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in rows:
        key = (str(row["group_id"]), str(row["message_id"]))
        by_key.setdefault(key, []).append(
            {
                "user_id": str(row["mentioned_user_id"] or ""),
                "display_name": str(row["display_name"] or ""),
                "is_bot": bool(row["is_bot"]),
                "raw_data": str(row["raw_data"] or ""),
            }
        )
    for item in messages:
        key = (str(item["group_id"]), str(item["message_id"]))
        item["mentions"] = by_key.get(key, [])
