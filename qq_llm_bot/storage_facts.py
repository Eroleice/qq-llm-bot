from __future__ import annotations

import time
from collections.abc import Iterable
from dataclasses import replace
from typing import Any

from qq_llm_bot.models import FactCandidate, FactRecord, FactWriteSet
from qq_llm_bot.storage_fact_actions import (
    approve_fact,
    approve_user_pending_fact,
    forget_fact,
    reject_fact,
    reject_user_pending_fact,
)
from qq_llm_bot.storage_fact_queries import (
    get_fact_record,
    list_pending_facts,
    list_user_facts,
    list_user_facts_text,
)
from qq_llm_bot.storage_fact_rules import (
    CONFLICT_SENSITIVE_KINDS,
    FACT_INACTIVE_STATUSES,
    PROTECTED_FACT_TYPES,
    SENSITIVE_CONFIRMATION_KINDS,
    TRUSTED_THIRD_PARTY_THRESHOLD,
    fact_acceptance_status,
    find_conflicting_facts,
    find_duplicate_fact,
    insert_fact,
    normalize_fact_candidate,
    supersede_facts,
    sync_aliases_for_fact,
)


__all__ = [
    "CONFLICT_SENSITIVE_KINDS",
    "FACT_INACTIVE_STATUSES",
    "PROTECTED_FACT_TYPES",
    "SENSITIVE_CONFIRMATION_KINDS",
    "TRUSTED_THIRD_PARTY_THRESHOLD",
    "approve_fact",
    "approve_user_pending_fact",
    "fact_acceptance_status",
    "find_conflicting_facts",
    "find_duplicate_fact",
    "forget_fact",
    "get_fact_record",
    "insert_fact",
    "list_pending_facts",
    "list_user_facts",
    "list_user_facts_text",
    "normalize_fact_candidate",
    "record_fact_candidates",
    "reject_fact",
    "reject_user_pending_fact",
    "supersede_facts",
    "sync_aliases_for_fact",
]


def record_fact_candidates(storage: Any, facts: Iterable[FactCandidate]) -> FactWriteSet:
    accepted: list[FactRecord] = []
    pending: list[FactRecord] = []
    rejected: list[FactCandidate] = []
    now = int(time.time())

    with storage._connect() as conn:
        for item in facts:
            normalized = normalize_fact_candidate(item)
            acceptance_status = fact_acceptance_status(storage, conn, normalized)
            if acceptance_status == "rejected":
                rejected.append(replace(normalized, status="rejected"))
                continue

            conflicting_facts = find_conflicting_facts(conn, normalized)
            if conflicting_facts and normalized.claim_scope != "self_report":
                acceptance_status = "pending_confirmation"

            duplicate = find_duplicate_fact(conn, normalized, acceptance_status)
            if duplicate:
                conn.execute(
                    """
                    UPDATE member_facts
                    SET confidence = MAX(confidence, ?),
                        importance = MAX(importance, ?),
                        updated_at = ?,
                        last_seen_at = ?
                    WHERE id = ?
                    """,
                    (normalized.confidence, normalized.importance, now, now, duplicate.id),
                )
                updated = replace(
                    duplicate,
                    confidence=max(duplicate.confidence, normalized.confidence),
                    importance=max(duplicate.importance, normalized.importance),
                    updated_at=now,
                    last_seen_at=now,
                )
                if updated.status == "accepted":
                    sync_aliases_for_fact(conn, updated)
                if updated.status == "accepted":
                    accepted.append(updated)
                elif updated.status == "pending_confirmation":
                    pending.append(updated)
                continue

            inserted = insert_fact(
                conn,
                replace(
                    normalized,
                    status="accepted" if acceptance_status == "accepted" else "pending_confirmation",
                ),
                now,
            )
            if inserted.status == "accepted":
                if conflicting_facts and normalized.claim_scope == "self_report":
                    supersede_facts(conn, conflicting_facts, inserted.id, now)
                sync_aliases_for_fact(conn, inserted)
                accepted.append(inserted)
            elif inserted.status == "pending_confirmation":
                pending.append(inserted)

    return FactWriteSet(accepted=accepted, pending=pending, rejected=rejected)
