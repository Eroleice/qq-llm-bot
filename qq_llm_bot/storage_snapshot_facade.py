from __future__ import annotations

import sqlite3

import qq_llm_bot.storage_snapshots as _storage_snapshots
from qq_llm_bot.models import ConversationSnapshot, MessageContext, TargetUserContext
from qq_llm_bot.storage_helpers import NameResolutionMatch as _NameResolutionMatch


class StorageSnapshotFacadeMixin:
    def build_snapshot(self, context: MessageContext) -> ConversationSnapshot:
        return _storage_snapshots.build_snapshot(self, context)

    def _resolve_target_user_contexts(
        self,
        context: MessageContext,
    ) -> tuple[list[TargetUserContext], list[str], dict[str, list[str]]]:
        return _storage_snapshots.resolve_target_user_contexts(self, context)

    def _lookup_alias_users(self, conn: sqlite3.Connection, name: str) -> list[str]:
        return _storage_snapshots.lookup_alias_users(self, conn, name)

    def _lookup_display_name_users(
        self,
        conn: sqlite3.Connection,
        group_id: str,
        name: str,
    ) -> list[_NameResolutionMatch]:
        return _storage_snapshots.lookup_display_name_users(self, conn, group_id, name)

    def _recent_group_display_name_candidates(
        self,
        conn: sqlite3.Connection,
        group_id: str,
    ) -> list[tuple[str, str, str]]:
        return _storage_snapshots.recent_group_display_name_candidates(conn, group_id)

    def _latest_profile_name_candidates(
        self,
        conn: sqlite3.Connection,
    ) -> list[tuple[str, str, str]]:
        return _storage_snapshots.latest_profile_name_candidates(conn)

    def _rank_name_candidates(
        self,
        ref: str,
        candidates: list[tuple[str, str, str]],
        *,
        reason_prefix: str,
        source_bonus: float,
    ) -> list[_NameResolutionMatch]:
        return _storage_snapshots.rank_name_candidates(
            ref,
            candidates,
            reason_prefix=reason_prefix,
            source_bonus=source_bonus,
        )

    def _list_active_aliases(
        self,
        conn: sqlite3.Connection,
        user_id: str,
        limit: int = 12,
    ) -> list[str]:
        return _storage_snapshots.list_active_aliases(self, conn, user_id, limit=limit)
