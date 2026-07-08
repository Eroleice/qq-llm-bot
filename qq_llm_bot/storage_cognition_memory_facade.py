from __future__ import annotations

import sqlite3
from typing import Iterable

import qq_llm_bot.storage_memories as _storage_memories
from qq_llm_bot.models import MemoryCandidate, MemoryRecord, MemoryWriteSet


class StorageCognitionMemoryFacadeMixin:
    def record_memories(self, memories: Iterable[MemoryCandidate]) -> None:
        _storage_memories.record_memories(self, memories)

    def record_memory_candidates(
        self,
        memories: Iterable[MemoryCandidate],
        confidence_threshold: float = 0.75,
    ) -> MemoryWriteSet:
        return _storage_memories.record_memory_candidates(self, memories, confidence_threshold)

    def list_memories(
        self,
        owner_type: str,
        owner_id: str,
        limit: int = 8,
        status: str = "active",
    ) -> list[MemoryRecord]:
        return _storage_memories.list_memories(self, owner_type, owner_id, limit, status)

    def list_user_memories(self, user_id: str, limit: int = 8) -> list[str]:
        return _storage_memories.list_user_memories(self, user_id, limit)

    def list_self_memories(self, status: str = "active", limit: int = 16) -> list[str]:
        return _storage_memories.list_self_memories(self, status, limit)

    def list_memories_by_status(self, status: str, limit: int = 12) -> list[str]:
        return _storage_memories.list_memories_by_status(self, status, limit)

    def forget_memory(self, memory_id: int) -> bool:
        return _storage_memories.forget_memory(self, memory_id)

    def approve_memory(self, memory_id: int) -> bool:
        return _storage_memories.approve_memory(self, memory_id)

    def reject_memory(self, memory_id: int) -> bool:
        return _storage_memories.reject_memory(self, memory_id)

    def get_memory_record(self, memory_id: int) -> MemoryRecord | None:
        return _storage_memories.get_memory_record(self, memory_id)

    def _normalize_memory_candidate(self, item: MemoryCandidate) -> MemoryCandidate:
        return _storage_memories.normalize_memory_candidate(item)

    def _acceptance_status(
        self,
        conn: sqlite3.Connection,
        item: MemoryCandidate,
        confidence_threshold: float,
    ) -> str:
        return _storage_memories.acceptance_status(self, conn, item, confidence_threshold)

    def _source_trust(self, conn: sqlite3.Connection, group_id: str, user_id: str) -> int:
        return _storage_memories.source_trust(conn, group_id, user_id)

    def _is_admin_conn(self, conn: sqlite3.Connection, user_id: str) -> bool:
        return _storage_memories.is_admin_conn(self, conn, user_id)

    def _find_duplicate_memory(
        self,
        conn: sqlite3.Connection,
        item: MemoryCandidate,
    ) -> MemoryRecord | None:
        return _storage_memories.find_duplicate_memory(conn, item)

    def _find_conflicting_memory(
        self,
        conn: sqlite3.Connection,
        item: MemoryCandidate,
    ) -> MemoryRecord | None:
        return _storage_memories.find_conflicting_memory(conn, item)

    def _find_conflicting_self_memory(
        self,
        conn: sqlite3.Connection,
        item: MemoryCandidate,
    ) -> MemoryRecord | None:
        return _storage_memories.find_conflicting_self_memory(conn, item)

    def _insert_memory(self, conn: sqlite3.Connection, item: MemoryCandidate, now: int) -> None:
        _storage_memories.insert_memory(conn, item, now)
