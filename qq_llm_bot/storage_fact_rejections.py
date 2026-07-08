from __future__ import annotations

import time
from typing import Any

from qq_llm_bot.storage_records import _dashboard_user_id


def reject_fact(storage: Any, fact_id: int) -> bool:
    with storage._connect() as conn:
        cursor = conn.execute(
            """
            UPDATE member_facts
            SET status = 'rejected', updated_at = ?
            WHERE id = ? AND status = 'pending_confirmation'
            """,
            (int(time.time()), int(fact_id)),
        )
    return cursor.rowcount > 0


def reject_user_pending_fact(storage: Any, user_id: str, fact_id: int) -> bool:
    subject = _dashboard_user_id(user_id)
    if not subject:
        return False
    with storage._connect() as conn:
        cursor = conn.execute(
            """
            UPDATE member_facts
            SET status = 'rejected', updated_at = ?
            WHERE id = ? AND subject_user_id = ? AND status = 'pending_confirmation'
            """,
            (int(time.time()), int(fact_id), subject),
        )
    return cursor.rowcount > 0
