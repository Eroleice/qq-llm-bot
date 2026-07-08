from __future__ import annotations

import time
from dataclasses import replace
from typing import Any

from qq_llm_bot.models import FactRecord
from qq_llm_bot.storage_fact_action_records import fetch_fact_record
from qq_llm_bot.storage_helpers import clean_fact_field as _clean_fact_field


def forget_fact(storage: Any, fact_id: int, reason: str = "manual") -> FactRecord | None:
    now = int(time.time())
    with storage._connect() as conn:
        record = fetch_fact_record(conn, fact_id)
        if record is None:
            return None
        cursor = conn.execute(
            """
            UPDATE member_facts
            SET status = 'forgotten',
                forget_reason = ?,
                updated_at = ?
            WHERE id = ?
              AND status IN ('accepted', 'pending_confirmation', 'superseded')
            """,
            (_clean_fact_field(reason, 120) or "manual", now, int(fact_id)),
        )
        if cursor.rowcount <= 0:
            return None
        conn.execute(
            """
            UPDATE member_aliases
            SET status = 'forgotten',
                updated_at = ?
            WHERE source_fact_id = ?
              AND status = 'active'
            """,
            (now, int(fact_id)),
        )
    return replace(record, status="forgotten", updated_at=now, forget_reason=reason or "manual")
