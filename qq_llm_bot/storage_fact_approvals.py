from __future__ import annotations

import time
from typing import Any

from qq_llm_bot.models import FactRecord
from qq_llm_bot.storage_fact_action_records import fetch_fact_record
from qq_llm_bot.storage_fact_rules import sync_aliases_for_fact
from qq_llm_bot.storage_records import _dashboard_user_id


def approve_fact(storage: Any, fact_id: int) -> FactRecord | None:
    now = int(time.time())
    with storage._connect() as conn:
        cursor = conn.execute(
            """
            UPDATE member_facts
            SET status = 'accepted', updated_at = ?, last_seen_at = ?
            WHERE id = ? AND status = 'pending_confirmation'
            """,
            (now, now, int(fact_id)),
        )
        if cursor.rowcount <= 0:
            return None
        record = fetch_fact_record(conn, fact_id)
        if record is not None:
            sync_aliases_for_fact(conn, record)
    return record


def approve_user_pending_fact(storage: Any, user_id: str, fact_id: int) -> FactRecord | None:
    subject = _dashboard_user_id(user_id)
    if not subject:
        return None
    now = int(time.time())
    with storage._connect() as conn:
        cursor = conn.execute(
            """
            UPDATE member_facts
            SET status = 'accepted', updated_at = ?, last_seen_at = ?
            WHERE id = ? AND subject_user_id = ? AND status = 'pending_confirmation'
            """,
            (now, now, int(fact_id), subject),
        )
        if cursor.rowcount <= 0:
            return None
        record = fetch_fact_record(conn, fact_id)
        if record is not None:
            sync_aliases_for_fact(conn, record)
    return record
