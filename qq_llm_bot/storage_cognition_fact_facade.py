from __future__ import annotations

import sqlite3
from typing import Iterable

import qq_llm_bot.storage_facts as _storage_facts
from qq_llm_bot.models import FactCandidate, FactRecord, FactWriteSet


class StorageCognitionFactFacadeMixin:
    def record_fact_candidates(self, facts: Iterable[FactCandidate]) -> FactWriteSet:
        return _storage_facts.record_fact_candidates(self, facts)

    def list_user_facts(
        self,
        user_id: str,
        limit: int = 20,
        status: str = "accepted",
        group_id: str = "",
        include_faded: bool = False,
    ) -> list[FactRecord]:
        return _storage_facts.list_user_facts(
            self,
            user_id,
            limit,
            status,
            group_id,
            include_faded,
        )

    def list_user_facts_text(self, user_id: str, limit: int = 20, status: str = "accepted") -> list[str]:
        return _storage_facts.list_user_facts_text(self, user_id, limit, status)

    def list_pending_facts(self, limit: int = 20) -> list[FactRecord]:
        return _storage_facts.list_pending_facts(self, limit)

    def get_fact_record(self, fact_id: int) -> FactRecord | None:
        return _storage_facts.get_fact_record(self, fact_id)

    def approve_fact(self, fact_id: int) -> FactRecord | None:
        return _storage_facts.approve_fact(self, fact_id)

    def approve_user_pending_fact(self, user_id: str, fact_id: int) -> FactRecord | None:
        return _storage_facts.approve_user_pending_fact(self, user_id, fact_id)

    def reject_fact(self, fact_id: int) -> bool:
        return _storage_facts.reject_fact(self, fact_id)

    def reject_user_pending_fact(self, user_id: str, fact_id: int) -> bool:
        return _storage_facts.reject_user_pending_fact(self, user_id, fact_id)

    def forget_fact(self, fact_id: int, reason: str = "manual") -> FactRecord | None:
        return _storage_facts.forget_fact(self, fact_id, reason)

    def _normalize_fact_candidate(self, item: FactCandidate) -> FactCandidate:
        return _storage_facts.normalize_fact_candidate(item)

    def _fact_acceptance_status(
        self,
        conn: sqlite3.Connection,
        item: FactCandidate,
    ) -> str:
        return _storage_facts.fact_acceptance_status(self, conn, item)

    def _find_duplicate_fact(
        self,
        conn: sqlite3.Connection,
        item: FactCandidate,
        acceptance_status: str,
    ) -> FactRecord | None:
        return _storage_facts.find_duplicate_fact(conn, item, acceptance_status)

    def _find_conflicting_facts(
        self,
        conn: sqlite3.Connection,
        item: FactCandidate,
    ) -> list[FactRecord]:
        return _storage_facts.find_conflicting_facts(conn, item)

    def _supersede_facts(
        self,
        conn: sqlite3.Connection,
        records: list[FactRecord],
        replacement_fact_id: int,
        now: int,
    ) -> None:
        _storage_facts.supersede_facts(conn, records, replacement_fact_id, now)

    def _sync_aliases_for_fact(self, conn: sqlite3.Connection, fact: FactRecord) -> None:
        _storage_facts.sync_aliases_for_fact(conn, fact)

    def _insert_fact(
        self,
        conn: sqlite3.Connection,
        item: FactCandidate,
        now: int,
    ) -> FactRecord:
        return _storage_facts.insert_fact(conn, item, now)
