from __future__ import annotations

import time
from typing import Any


def forget_memory(storage: Any, memory_id: int) -> bool:
    with storage._connect() as conn:
        cursor = conn.execute(
            """
            UPDATE memory_items
            SET status = 'forgotten', updated_at = ?
            WHERE id = ? AND status != 'forgotten'
            """,
            (int(time.time()), int(memory_id)),
        )
    return cursor.rowcount > 0


def approve_memory(storage: Any, memory_id: int) -> bool:
    now = int(time.time())
    with storage._connect() as conn:
        row = conn.execute(
            """
            SELECT conflict_of
            FROM memory_items
            WHERE id = ? AND status IN ('pending_confirmation', 'conflict')
            """,
            (int(memory_id),),
        ).fetchone()
        if row is None:
            return False

        conflict_of = row["conflict_of"]
        if conflict_of:
            conn.execute(
                """
                UPDATE memory_items
                SET status = 'forgotten',
                    verification_status = 'rejected',
                    updated_at = ?
                WHERE id = ? AND status = 'active'
                """,
                (now, int(conflict_of)),
            )

        cursor = conn.execute(
            """
            UPDATE memory_items
            SET status = 'active',
                verification_status = 'accepted',
                conflict_of = NULL,
                updated_at = ?,
                last_seen_at = ?
            WHERE id = ? AND status IN ('pending_confirmation', 'conflict')
            """,
            (now, now, int(memory_id)),
        )
    return cursor.rowcount > 0


def reject_memory(storage: Any, memory_id: int) -> bool:
    with storage._connect() as conn:
        cursor = conn.execute(
            """
            UPDATE memory_items
            SET status = 'rejected',
                verification_status = 'rejected',
                updated_at = ?
            WHERE id = ? AND status IN ('pending_confirmation', 'conflict')
            """,
            (int(time.time()), int(memory_id)),
        )
    return cursor.rowcount > 0
