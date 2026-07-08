from __future__ import annotations

import sqlite3
from pathlib import Path
from threading import RLock
from typing import Iterable

import qq_llm_bot.storage_facts as _storage_facts
import qq_llm_bot.storage_lifecycle as _storage_lifecycle
from qq_llm_bot.config import AppConfig
from qq_llm_bot.storage_access_facade import StorageAccessFacadeMixin
from qq_llm_bot.storage_cognition_facade import StorageCognitionFacadeMixin
from qq_llm_bot.storage_dashboard_facade import StorageDashboardFacadeMixin
from qq_llm_bot.storage_helpers import clamp_float as _clamp_float
from qq_llm_bot.storage_message_facade import StorageMessageFacadeMixin
from qq_llm_bot.storage_records import (
    _clamp_score,
    _dashboard_user_id,
)
from qq_llm_bot.storage_snapshot_facade import StorageSnapshotFacadeMixin
from qq_llm_bot.storage_sticker_facade import StorageStickerFacadeMixin

CONFLICT_SENSITIVE_KINDS = _storage_facts.CONFLICT_SENSITIVE_KINDS
SENSITIVE_CONFIRMATION_KINDS = _storage_facts.SENSITIVE_CONFIRMATION_KINDS
TRUSTED_THIRD_PARTY_THRESHOLD = _storage_facts.TRUSTED_THIRD_PARTY_THRESHOLD


class BotStorage(
    StorageMessageFacadeMixin,
    StorageStickerFacadeMixin,
    StorageCognitionFacadeMixin,
    StorageDashboardFacadeMixin,
    StorageSnapshotFacadeMixin,
    StorageAccessFacadeMixin,
):
    def __init__(
        self,
        db_path: Path,
        initial_admins: Iterable[str],
        initial_ignored_users: Iterable[str],
        initial_groups: Iterable[str],
        initial_persona: dict[str, str],
        sticker_context_limit: int = 24,
        fact_confidence_threshold: float = 0.75,
        third_party_trust_threshold: int = 70,
        third_party_confidence_threshold: float = 0.85,
        profile_fact_threshold: int = 5,
        context_fact_limit: int = 8,
        target_user_limit: int = 5,
        low_importance_threshold: float = 0.35,
        fact_context_ttl_days: int = 30,
        interaction_followup_seconds: int = 180,
    ) -> None:
        self.db_path = db_path
        self.initial_admins = {str(item) for item in initial_admins}
        self.initial_ignored_users = {_dashboard_user_id(str(item)) for item in initial_ignored_users}
        self.initial_ignored_users.discard("")
        self.initial_groups = {str(item) for item in initial_groups}
        self.initial_persona = initial_persona
        self.sticker_context_limit = max(1, int(sticker_context_limit))
        self.fact_confidence_threshold = _clamp_float(fact_confidence_threshold)
        self.third_party_trust_threshold = _clamp_score(third_party_trust_threshold)
        self.third_party_confidence_threshold = _clamp_float(third_party_confidence_threshold)
        self.profile_fact_threshold = max(1, int(profile_fact_threshold))
        self.context_fact_limit = max(1, int(context_fact_limit))
        self.target_user_limit = max(1, int(target_user_limit))
        self.low_importance_threshold = _clamp_float(low_importance_threshold)
        self.fact_context_ttl_seconds = max(1, int(fact_context_ttl_days)) * 24 * 60 * 60
        self.interaction_followup_seconds = max(1, int(interaction_followup_seconds))
        self._lock = RLock()

    @classmethod
    def from_config(cls, config: AppConfig) -> "BotStorage":
        return _storage_lifecycle.storage_from_config(cls, config)

    def setup(self) -> None:
        _storage_lifecycle.setup_storage(self)

    def _backfill_member_aliases(self, conn: sqlite3.Connection) -> None:
        _storage_lifecycle.backfill_member_aliases(self, conn)

    def _reject_unreasonable_member_aliases(self, conn: sqlite3.Connection) -> None:
        _storage_lifecycle.reject_unreasonable_member_aliases(conn)

    def _connect(self):
        return _storage_lifecycle.connect_storage(self)
