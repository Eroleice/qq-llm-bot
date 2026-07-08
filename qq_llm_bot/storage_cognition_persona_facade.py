from __future__ import annotations

from typing import Iterable

import qq_llm_bot.storage_lexicon as _storage_lexicon
import qq_llm_bot.storage_persona as _storage_persona
from qq_llm_bot.models import MemoryRecord


class StorageCognitionPersonaFacadeMixin:
    def list_group_lexicon_records(
        self,
        group_id: str,
        term: str = "",
        limit: int = 12,
        status: str = "active",
    ) -> list[MemoryRecord]:
        return _storage_lexicon.list_group_lexicon_records(self, group_id, term, limit, status)

    def list_group_lexicon(self, group_id: str, term: str = "", limit: int = 12) -> list[str]:
        return _storage_lexicon.list_group_lexicon(self, group_id, term, limit)

    def has_group_lexicon(
        self,
        group_id: str,
        term: str,
        statuses: Iterable[str] = ("active", "pending_confirmation", "conflict"),
    ) -> bool:
        return _storage_lexicon.has_group_lexicon(self, group_id, term, statuses)

    def get_persona_lines(self) -> list[str]:
        return _storage_persona.get_persona_lines(self)

    def format_persona(self) -> str:
        return _storage_persona.format_persona(self)

    def should_reflect(self, group_id: str, threshold: int, min_interval_seconds: int) -> bool:
        return _storage_persona.should_reflect(self, group_id, threshold, min_interval_seconds)
