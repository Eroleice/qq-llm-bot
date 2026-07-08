from __future__ import annotations

import sqlite3
from typing import Any

from qq_llm_bot.models import FactCandidate
from qq_llm_bot.storage_helpers import (
    is_complete_fact as _is_complete_fact,
    is_unreasonable_alias_fact as _is_unreasonable_alias_fact,
    looks_low_value_fact as _looks_low_value_fact,
)


def fact_acceptance_status(
    storage: Any,
    conn: sqlite3.Connection,
    item: FactCandidate,
) -> str:
    if not _is_complete_fact(item):
        return "rejected"
    if item.confidence < storage.fact_confidence_threshold:
        return "rejected"
    if _looks_low_value_fact(item):
        return "rejected"
    if _is_unreasonable_alias_fact(item):
        return "rejected"
    if item.claim_scope == "third_party":
        source_trust = storage._source_trust(conn, item.source_group_id, item.source_user_id)
        if (
            source_trust >= storage.third_party_trust_threshold
            and item.confidence >= storage.third_party_confidence_threshold
        ):
            return "accepted"
        return "pending_confirmation"
    return "accepted"
