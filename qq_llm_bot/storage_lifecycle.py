from __future__ import annotations

import sqlite3
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from qq_llm_bot.config import AppConfig
from qq_llm_bot.storage_helpers import is_reasonable_member_alias as _is_reasonable_member_alias
from qq_llm_bot.storage_records import (
    _compact_persona_items,
    _fact_record,
)
from qq_llm_bot.storage_schema import (
    create_storage_schema,
    migrate_storage_schema,
    seed_storage_config,
)


def build_initial_persona(config: AppConfig) -> dict[str, str]:
    persona = {
        "self_name": config.persona.self_name,
        "core_traits": "、".join(config.persona.core_traits),
        "speech_style": "、".join(config.persona.speech_style),
        "boundaries": "、".join(config.persona.boundaries),
        "current_mood": config.persona.current_mood,
        "relationship_tendency": config.persona.relationship_tendency,
        "activity_level": str(config.persona.activity_level),
    }
    persona.update(
        _compact_persona_items(
            {
                "full_name": config.persona.full_name,
                "gender": config.persona.gender,
                "age": str(config.persona.age) if config.persona.age else "",
                "city": config.persona.city,
                "education_school": config.persona.education_school,
                "education_major": config.persona.education_major,
                "education_degree": config.persona.education_degree,
                "employer": config.persona.employer,
                "occupation": config.persona.occupation,
                "work_years": str(config.persona.work_years) if config.persona.work_years else "",
                "relationship_status": config.persona.relationship_status,
                "background_summary": config.persona.background_summary,
                "appearance_prompt": config.persona.appearance_prompt,
            }
        )
    )
    return persona


def storage_from_config(storage_cls: type[Any], config: AppConfig) -> Any:
    return storage_cls(
        config.resolve_path(config.storage.sqlite_path),
        config.bot.admin_ids,
        config.bot.ignored_user_ids,
        config.bot.enabled_groups,
        build_initial_persona(config),
        config.stickers.max_context_stickers,
        config.facts.fact_confidence_threshold,
        config.facts.third_party_trust_threshold,
        config.facts.third_party_confidence_threshold,
        config.facts.profile_fact_threshold,
        config.facts.context_fact_limit,
        config.facts.target_user_limit,
        config.facts.low_importance_threshold,
        config.facts.fact_context_ttl_days,
        config.bot.interaction_followup_seconds,
    )


def setup_storage(storage: Any) -> None:
    storage.db_path.parent.mkdir(parents=True, exist_ok=True)
    with storage._connect() as conn:
        create_storage_schema(conn)
        migrate_storage_schema(conn)
        backfill_member_aliases(storage, conn)
        reject_unreasonable_member_aliases(conn)
        seed_storage_config(
            conn,
            admins=storage.initial_admins,
            enabled_groups=storage.initial_groups,
            ignored_users=storage.initial_ignored_users,
            persona=storage.initial_persona,
        )


def backfill_member_aliases(storage: Any, conn: sqlite3.Connection) -> None:
    rows = conn.execute(
        """
        SELECT id, subject_user_id, fact_type, claim_text, topic, stance,
               confidence, status, claim_scope, source_user_id, source_group_id,
               evidence_message_id, evidence_text, created_at, updated_at,
               importance, last_seen_at, superseded_by_fact_id, forget_reason
        FROM member_facts
        WHERE status = 'accepted'
          AND fact_type IN ('identity', 'alias')
        ORDER BY updated_at ASC, id ASC
        """
    ).fetchall()
    for row in rows:
        storage._sync_aliases_for_fact(conn, _fact_record(row))


def reject_unreasonable_member_aliases(conn: sqlite3.Connection) -> None:
    rows = conn.execute(
        """
        SELECT id, alias
        FROM member_aliases
        WHERE status = 'active'
        """
    ).fetchall()
    ids = [
        int(row["id"])
        for row in rows
        if not _is_reasonable_member_alias(str(row["alias"] or ""))
    ]
    if not ids:
        return
    placeholders = ", ".join("?" for _ in ids)
    conn.execute(
        f"""
        UPDATE member_aliases
        SET status = 'rejected',
            updated_at = ?
        WHERE id IN ({placeholders})
        """,
        [int(time.time()), *ids],
    )


@contextmanager
def connect_storage(storage: Any) -> Iterator[sqlite3.Connection]:
    with storage._lock:
        conn = sqlite3.connect(storage.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()
