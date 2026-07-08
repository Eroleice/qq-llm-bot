from __future__ import annotations

from typing import Any

from qq_llm_bot.storage_dashboard_cognition_candidates import collect_dashboard_cognition_candidates
from qq_llm_bot.storage_dashboard_cognition_items import build_dashboard_cognition_items
from qq_llm_bot.storage_dashboard_user_records import (
    dashboard_member_profile,
    dashboard_user_profile,
    list_dashboard_relationship_rows,
    list_dashboard_user_fact_records,
)


__all__ = [
    "dashboard_member_profile",
    "dashboard_user_profile",
    "list_dashboard_relationship_rows",
    "list_dashboard_user_cognition",
    "list_dashboard_user_fact_records",
]


def list_dashboard_user_cognition(
    storage: Any,
    group_id: str = "",
    user_id: str = "",
    limit: int = 100,
) -> list[dict[str, object]]:
    requested_group_id = str(group_id).strip()
    requested_user_id = str(user_id).strip()
    limit = max(1, int(limit))
    query_limit = max(limit * 4, limit)

    with storage._connect() as conn:
        candidates = collect_dashboard_cognition_candidates(
            conn,
            requested_group_id,
            requested_user_id,
            query_limit,
        )
        items = build_dashboard_cognition_items(conn, candidates, query_limit)

    items.sort(key=lambda item: int(item.get("updated_at", 0)), reverse=True)
    return items[:limit]
