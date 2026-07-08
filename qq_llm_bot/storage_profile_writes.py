from __future__ import annotations

import json
import time
from typing import Any

from qq_llm_bot.models import FactRecord, UserProfileDraft, UserProfileRecord
from qq_llm_bot.storage_records import _compact_int_list, _dashboard_user_id


def maybe_update_user_profile(
    storage: Any,
    user_id: str,
    draft: UserProfileDraft,
    facts: list[FactRecord],
    force: bool = False,
) -> UserProfileRecord | None:
    subject = _dashboard_user_id(user_id)
    if not subject or not draft.summary.strip():
        return None
    if not force and not storage.should_update_user_profile(subject):
        return None
    accepted_facts = [fact for fact in facts if fact.status == "accepted"]
    fact_count = len(accepted_facts)
    supporting_ids = draft.supporting_fact_ids or tuple(fact.id for fact in accepted_facts[-20:])
    now = int(time.time())
    summary = " ".join(draft.summary.split())[:500]
    with storage._connect() as conn:
        current = conn.execute(
            "SELECT version FROM member_profiles WHERE user_id = ?",
            (subject,),
        ).fetchone()
        version = int(current["version"]) + 1 if current else 1
        conn.execute(
            """
            INSERT INTO member_profiles (
                user_id, summary, traits_json, supporting_fact_ids,
                fact_count, version, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(user_id) DO UPDATE SET
                summary = excluded.summary,
                traits_json = excluded.traits_json,
                supporting_fact_ids = excluded.supporting_fact_ids,
                fact_count = excluded.fact_count,
                version = excluded.version,
                updated_at = excluded.updated_at
            """,
            (
                subject,
                summary,
                json.dumps(draft.traits, ensure_ascii=False),
                json.dumps(_compact_int_list(supporting_ids, limit=80), ensure_ascii=False),
                fact_count,
                version,
                now,
            ),
        )
    return UserProfileRecord(
        user_id=subject,
        summary=summary,
        traits=draft.traits,
        supporting_fact_ids=tuple(_compact_int_list(supporting_ids, limit=80)),
        fact_count=fact_count,
        version=version,
        updated_at=now,
    )


def clear_user_profile(storage: Any, user_id: str) -> bool:
    subject = _dashboard_user_id(user_id)
    if not subject:
        return False
    with storage._connect() as conn:
        cursor = conn.execute("DELETE FROM member_profiles WHERE user_id = ?", (subject,))
    return cursor.rowcount > 0
