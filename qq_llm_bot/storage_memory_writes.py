from __future__ import annotations

import time
from collections.abc import Iterable
from dataclasses import replace
from typing import Any

from qq_llm_bot.models import MemoryCandidate, MemoryWriteSet
from qq_llm_bot.storage_memory_rules import (
    acceptance_status,
    find_conflicting_memory,
    find_duplicate_memory,
    insert_memory,
    normalize_memory_candidate,
)


def record_memories(storage: Any, memories: Iterable[MemoryCandidate]) -> None:
    record_memory_candidates(storage, memories)


def record_memory_candidates(
    storage: Any,
    memories: Iterable[MemoryCandidate],
    confidence_threshold: float = 0.75,
) -> MemoryWriteSet:
    accepted: list[MemoryCandidate] = []
    pending: list[MemoryCandidate] = []
    conflicts: list[MemoryCandidate] = []
    rejected: list[MemoryCandidate] = []
    now = int(time.time())

    with storage._connect() as conn:
        for item in memories:
            normalized = normalize_memory_candidate(item)
            acceptance = acceptance_status(storage, conn, normalized, confidence_threshold)
            if acceptance == "rejected":
                rejected.append(replace(normalized, status="rejected", verification_status="rejected"))
                continue

            if acceptance == "pending_confirmation":
                pending_item = replace(
                    normalized,
                    status="pending_confirmation",
                    verification_status="pending_confirmation",
                )
                insert_memory(conn, pending_item, now)
                pending.append(pending_item)
                continue

            duplicate = find_duplicate_memory(conn, normalized)
            if duplicate:
                conn.execute(
                    """
                    UPDATE memory_items
                    SET confidence = MAX(confidence, ?),
                        importance = MAX(importance, ?),
                        source_text = CASE WHEN source_text = '' THEN ? ELSE source_text END,
                        updated_at = ?,
                        last_seen_at = ?
                    WHERE id = ?
                    """,
                    (
                        normalized.confidence,
                        normalized.importance,
                        normalized.source_text,
                        now,
                        now,
                        duplicate.id,
                    ),
                )
                accepted.append(replace(normalized, status="active", verification_status="accepted"))
                continue

            conflict = find_conflicting_memory(conn, normalized)
            if conflict:
                conflict_item = replace(
                    normalized,
                    status="conflict",
                    verification_status="conflict",
                    conflict_of=conflict.id,
                )
                insert_memory(conn, conflict_item, now)
                conflicts.append(conflict_item)
                continue

            active_item = replace(normalized, status="active", verification_status="accepted")
            insert_memory(conn, active_item, now)
            accepted.append(active_item)

    return MemoryWriteSet(
        accepted=accepted,
        pending=pending,
        conflicts=conflicts,
        rejected=rejected,
    )
