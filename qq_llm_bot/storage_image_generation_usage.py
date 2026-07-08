from __future__ import annotations

import time


def count_image_generation_usage(storage: object, user_id: str, usage_date: str) -> int:
    with storage._connect() as conn:
        row = conn.execute(
            """
            SELECT COUNT(1) AS count
            FROM image_generation_usage
            WHERE user_id = ? AND usage_date = ?
            """,
            (str(user_id), str(usage_date)),
        ).fetchone()
    return int(row["count"]) if row is not None else 0

def record_image_generation_usage(
    storage: object,
    group_id: str,
    user_id: str,
    usage_date: str,
    prompt: str,
    image_ref: str,
    created_at: int | None = None,
) -> None:
    now = int(created_at or time.time())
    with storage._connect() as conn:
        conn.execute(
            """
            INSERT INTO image_generation_usage (
                usage_date, group_id, user_id, prompt, image_ref, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                str(usage_date),
                str(group_id),
                str(user_id),
                str(prompt)[:1000],
                str(image_ref)[:1000],
                now,
            ),
        )
