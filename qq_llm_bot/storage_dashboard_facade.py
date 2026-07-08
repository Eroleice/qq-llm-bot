from __future__ import annotations

import sqlite3

import qq_llm_bot.storage_dashboard as _storage_dashboard
from qq_llm_bot.models import FactRecord, UserProfileRecord


class StorageDashboardFacadeMixin:
    def list_dashboard_llm_usage(
        self,
        since: int,
        limit: int = 100,
    ) -> dict[str, object]:
        return _storage_dashboard.list_dashboard_llm_usage(self, since, limit=limit)

    def get_dashboard_persona(self) -> dict[str, object]:
        return _storage_dashboard.get_dashboard_persona(self)

    def list_dashboard_groups(self) -> list[str]:
        return _storage_dashboard.list_dashboard_groups(self)

    def list_dashboard_user_cognition(
        self,
        group_id: str = "",
        user_id: str = "",
        limit: int = 100,
    ) -> list[dict[str, object]]:
        return _storage_dashboard.list_dashboard_user_cognition(
            self,
            group_id=group_id,
            user_id=user_id,
            limit=limit,
        )

    def _list_dashboard_relationship_rows(
        self,
        conn: sqlite3.Connection,
        user_id: str,
        limit: int,
    ) -> list[sqlite3.Row]:
        return _storage_dashboard.list_dashboard_relationship_rows(conn, user_id, limit)

    def _list_dashboard_user_fact_records(
        self,
        conn: sqlite3.Connection,
        user_id: str,
        limit: int,
    ) -> list[FactRecord]:
        return _storage_dashboard.list_dashboard_user_fact_records(conn, user_id, limit)

    def _dashboard_member_profile(
        self,
        conn: sqlite3.Connection,
        user_id: str,
    ) -> UserProfileRecord | None:
        return _storage_dashboard.dashboard_member_profile(conn, user_id)

    def _dashboard_user_profile(
        self,
        conn: sqlite3.Connection,
        user_id: str,
    ) -> dict[str, object]:
        return _storage_dashboard.dashboard_user_profile(conn, user_id)

    def list_dashboard_messages(
        self,
        group_id: str = "",
        user_id: str = "",
        start_time: int | None = None,
        end_time: int | None = None,
        limit: int = 200,
    ) -> list[dict[str, object]]:
        return _storage_dashboard.list_dashboard_messages(
            self,
            group_id=group_id,
            user_id=user_id,
            start_time=start_time,
            end_time=end_time,
            limit=limit,
        )

    def list_dashboard_final_qa_blocks(
        self,
        group_id: str = "",
        user_id: str = "",
        start_time: int | None = None,
        end_time: int | None = None,
        limit: int = 100,
    ) -> list[dict[str, object]]:
        return _storage_dashboard.list_dashboard_final_qa_blocks(
            self,
            group_id=group_id,
            user_id=user_id,
            start_time=start_time,
            end_time=end_time,
            limit=limit,
        )

    def _attach_dashboard_attachments(self, messages: list[dict[str, object]]) -> None:
        _storage_dashboard.attach_dashboard_attachments(self, messages)

    def _attach_dashboard_mentions(self, messages: list[dict[str, object]]) -> None:
        _storage_dashboard.attach_dashboard_mentions(self, messages)

    def list_dashboard_stickers(
        self,
        group_id: str = "",
        limit: int = 200,
    ) -> list[dict[str, object]]:
        return _storage_dashboard.list_dashboard_stickers(self, group_id=group_id, limit=limit)

    def list_dashboard_pending(self, limit: int = 100) -> list[dict[str, object]]:
        return _storage_dashboard.list_dashboard_pending(self, limit=limit)
