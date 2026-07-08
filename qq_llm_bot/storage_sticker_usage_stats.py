from __future__ import annotations

import time
from typing import Any

from qq_llm_bot.storage_record_values import _local_usage_date


def record_sticker_sent(
    storage: Any,
    sticker_id: int,
    usage_date: str = "",
    sent_at: int | None = None,
) -> None:
    now = int(sent_at or time.time())
    usage_day = str(usage_date).strip() or _local_usage_date(now)
    with storage._connect() as conn:
        conn.execute(
            """
            UPDATE sticker_assets
            SET last_seen_at = ?,
                hit_count = hit_count + 1,
                send_count = send_count + 1,
                last_sent_at = ?
            WHERE id = ?
            """,
            (now, now, int(sticker_id)),
        )
        row = conn.execute(
            """
            SELECT group_id
            FROM sticker_assets
            WHERE id = ?
            """,
            (int(sticker_id),),
        ).fetchone()
        if row is None:
            return
        conn.execute(
            """
            INSERT INTO sticker_usage_daily (
                sticker_id, group_id, usage_date, send_count, first_sent_at, last_sent_at
            )
            VALUES (?, ?, ?, 1, ?, ?)
            ON CONFLICT(sticker_id, usage_date) DO UPDATE SET
                group_id = excluded.group_id,
                send_count = sticker_usage_daily.send_count + 1,
                last_sent_at = excluded.last_sent_at
            """,
            (int(sticker_id), str(row["group_id"]), usage_day, now, now),
        )


def count_sticker_usage(storage: Any, sticker_id: int, usage_date: str = "") -> int:
    where = ["sticker_id = ?"]
    params: list[object] = [int(sticker_id)]
    if usage_date:
        where.append("usage_date = ?")
        params.append(str(usage_date))
    with storage._connect() as conn:
        row = conn.execute(
            f"""
            SELECT COALESCE(SUM(send_count), 0) AS count
            FROM sticker_usage_daily
            WHERE {' AND '.join(where)}
            """,
            params,
        ).fetchone()
    return int(row["count"]) if row is not None else 0


def list_sticker_usage_daily(
    storage: Any,
    group_id: str = "",
    usage_date: str = "",
    limit: int = 200,
) -> list[dict[str, object]]:
    where: list[str] = []
    params: list[object] = []
    if group_id:
        where.append("group_id = ?")
        params.append(str(group_id))
    if usage_date:
        where.append("usage_date = ?")
        params.append(str(usage_date))
    params.append(int(limit))
    where_sql = f"WHERE {' AND '.join(where)}" if where else ""
    with storage._connect() as conn:
        rows = conn.execute(
            f"""
            SELECT sticker_id, group_id, usage_date, send_count, first_sent_at, last_sent_at
            FROM sticker_usage_daily
            {where_sql}
            ORDER BY usage_date DESC, send_count DESC, last_sent_at DESC
            LIMIT ?
            """,
            params,
        ).fetchall()
    return [
        {
            "sticker_id": int(row["sticker_id"]),
            "group_id": str(row["group_id"]),
            "usage_date": str(row["usage_date"]),
            "send_count": int(row["send_count"] or 0),
            "first_sent_at": int(row["first_sent_at"] or 0),
            "last_sent_at": int(row["last_sent_at"] or 0),
        }
        for row in rows
    ]
