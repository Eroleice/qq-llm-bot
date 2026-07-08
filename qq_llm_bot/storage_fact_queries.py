from __future__ import annotations

import time
from typing import Any

from qq_llm_bot.models import FactRecord
from qq_llm_bot.storage_fact_rules import PROTECTED_FACT_TYPES
from qq_llm_bot.storage_helpers import (
    is_unreasonable_alias_fact as _is_unreasonable_alias_fact,
    safe_fact_status as _safe_fact_status,
)
from qq_llm_bot.storage_records import (
    format_fact_record,
    _dashboard_user_id,
    _fact_record,
)


def list_user_facts(
    storage: Any,
    user_id: str,
    limit: int = 20,
    status: str = "accepted",
    group_id: str = "",
    include_faded: bool = False,
) -> list[FactRecord]:
    subject = _dashboard_user_id(user_id)
    if not subject:
        return []
    where = ["subject_user_id = ?", "status = ?"]
    params: list[object] = [subject, _safe_fact_status(status)]
    if group_id:
        where.append("source_group_id = ?")
        params.append(str(group_id))
    if status == "accepted" and not include_faded:
        cutoff = int(time.time()) - storage.fact_context_ttl_seconds
        protected = ", ".join("?" for _ in PROTECTED_FACT_TYPES)
        where.append(f"(importance >= ? OR fact_type IN ({protected}) OR last_seen_at >= ?)")
        params.extend([storage.low_importance_threshold, *sorted(PROTECTED_FACT_TYPES), cutoff])
    limit_value = int(limit)
    limit_sql = "LIMIT ?" if limit_value > 0 else ""
    if limit_value > 0:
        params.append(limit_value)
    with storage._connect() as conn:
        rows = conn.execute(
            f"""
            SELECT id, subject_user_id, fact_type, claim_text, topic, stance,
                   confidence, status, claim_scope, source_user_id, source_group_id,
                   evidence_message_id, evidence_text, created_at, updated_at,
                   importance, last_seen_at, superseded_by_fact_id, forget_reason
            FROM member_facts
            WHERE {' AND '.join(where)}
            ORDER BY
                CASE WHEN fact_type IN ('identity', 'alias', 'boundary') THEN 0 ELSE 1 END,
                importance DESC, updated_at DESC, id DESC
            {limit_sql}
            """,
            params,
        ).fetchall()
    records = [_fact_record(row) for row in rows]
    if status == "accepted":
        records = [record for record in records if not _is_unreasonable_alias_fact(record)]
    return records


def list_user_facts_text(
    storage: Any,
    user_id: str,
    limit: int = 20,
    status: str = "accepted",
) -> list[str]:
    return [format_fact_record(record) for record in list_user_facts(storage, user_id, limit, status)]


def list_pending_facts(storage: Any, limit: int = 20) -> list[FactRecord]:
    with storage._connect() as conn:
        rows = conn.execute(
            """
            SELECT id, subject_user_id, fact_type, claim_text, topic, stance,
                   confidence, status, claim_scope, source_user_id, source_group_id,
                   evidence_message_id, evidence_text, created_at, updated_at,
                   importance, last_seen_at, superseded_by_fact_id, forget_reason
            FROM member_facts
            WHERE status = 'pending_confirmation'
            ORDER BY updated_at DESC, id DESC
            LIMIT ?
            """,
            (int(limit),),
        ).fetchall()
    return [_fact_record(row) for row in rows]


def get_fact_record(storage: Any, fact_id: int) -> FactRecord | None:
    with storage._connect() as conn:
        row = conn.execute(
            """
            SELECT id, subject_user_id, fact_type, claim_text, topic, stance,
                   confidence, status, claim_scope, source_user_id, source_group_id,
                   evidence_message_id, evidence_text, created_at, updated_at,
                   importance, last_seen_at, superseded_by_fact_id, forget_reason
            FROM member_facts
            WHERE id = ?
            """,
            (int(fact_id),),
        ).fetchone()
    return _fact_record(row) if row else None
