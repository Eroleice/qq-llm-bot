from __future__ import annotations

from typing import Any

from qq_llm_bot.models import UserProfileRecord
from qq_llm_bot.storage_records import _dashboard_user_id, _user_profile_record


def get_user_profile(storage: Any, user_id: str) -> UserProfileRecord | None:
    subject = _dashboard_user_id(user_id)
    if not subject:
        return None
    with storage._connect() as conn:
        row = conn.execute(
            """
            SELECT user_id, summary, traits_json, supporting_fact_ids,
                   fact_count, version, updated_at
            FROM member_profiles
            WHERE user_id = ?
            """,
            (subject,),
        ).fetchone()
    return _user_profile_record(row) if row else None


def should_update_user_profile(storage: Any, user_id: str, threshold: int | None = None) -> bool:
    subject = _dashboard_user_id(user_id)
    if not subject or not subject.isdigit():
        return False
    threshold = max(1, int(threshold or storage.profile_fact_threshold))
    with storage._connect() as conn:
        accepted_count = int(
            conn.execute(
                """
                SELECT COUNT(*) AS count
                FROM member_facts
                WHERE subject_user_id = ? AND status = 'accepted'
                """,
                (subject,),
            ).fetchone()["count"]
        )
        row = conn.execute(
            "SELECT fact_count FROM member_profiles WHERE user_id = ?",
            (subject,),
        ).fetchone()
    profiled_count = int(row["fact_count"]) if row else 0
    return accepted_count - profiled_count >= threshold
