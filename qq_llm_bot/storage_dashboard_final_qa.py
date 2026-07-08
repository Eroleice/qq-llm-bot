from __future__ import annotations

from typing import Any

from qq_llm_bot.storage_records import _decode_string_list


def list_dashboard_final_qa_blocks(
    storage: Any,
    group_id: str = "",
    user_id: str = "",
    start_time: int | None = None,
    end_time: int | None = None,
    limit: int = 100,
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
        where.append("created_at >= ?")
        params.append(int(start_time))
    if end_time is not None:
        where.append("created_at < ?")
        params.append(int(end_time))
    where_sql = "WHERE " + " AND ".join(where) if where else ""

    with storage._connect() as conn:
        rows = conn.execute(
            f"""
            SELECT id, created_at, message_time, group_id, user_id, message_id,
                   sender_name, sender_role, trigger_text, raw_message,
                   is_direct, bot_mentioned, mode, action, decision_reason,
                   score, value_type, value_score, traffic_level,
                   candidate_reply, qa_reason, qa_categories, qa_confidence,
                   recent_messages, speaker_recent_messages, other_recent_messages,
                   recent_image_descriptions
            FROM final_qa_blocks
            {where_sql}
            ORDER BY created_at DESC, id DESC
            LIMIT ?
            """,
            [*params, int(limit)],
        ).fetchall()

    return [
        {
            "id": int(row["id"]),
            "created_at": int(row["created_at"]),
            "message_time": int(row["message_time"] or 0),
            "group_id": str(row["group_id"]),
            "user_id": str(row["user_id"]),
            "message_id": str(row["message_id"]),
            "sender_name": str(row["sender_name"] or ""),
            "sender_role": str(row["sender_role"] or ""),
            "trigger_text": str(row["trigger_text"] or ""),
            "raw_message": str(row["raw_message"] or ""),
            "is_direct": bool(row["is_direct"]),
            "bot_mentioned": bool(row["bot_mentioned"]),
            "mode": str(row["mode"] or ""),
            "action": str(row["action"] or ""),
            "decision_reason": str(row["decision_reason"] or ""),
            "score": float(row["score"] or 0.0),
            "value_type": str(row["value_type"] or ""),
            "value_score": float(row["value_score"] or 0.0),
            "traffic_level": str(row["traffic_level"] or ""),
            "candidate_reply": str(row["candidate_reply"] or ""),
            "qa_reason": str(row["qa_reason"] or ""),
            "qa_categories": _decode_string_list(str(row["qa_categories"] or "[]"), limit=20),
            "qa_confidence": float(row["qa_confidence"] or 0.0),
            "recent_messages": _decode_string_list(
                str(row["recent_messages"] or "[]"),
                limit=50,
            ),
            "speaker_recent_messages": _decode_string_list(
                str(row["speaker_recent_messages"] or "[]"),
                limit=50,
            ),
            "other_recent_messages": _decode_string_list(
                str(row["other_recent_messages"] or "[]"),
                limit=50,
            ),
            "recent_image_descriptions": _decode_string_list(
                str(row["recent_image_descriptions"] or "[]"),
                limit=50,
            ),
        }
        for row in rows
    ]
