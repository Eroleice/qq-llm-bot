from __future__ import annotations

import sqlite3

from qq_llm_bot.storage_dashboard_user_records import (
    dashboard_member_profile,
    dashboard_user_profile,
    list_dashboard_relationship_rows,
    list_dashboard_user_fact_records,
)
from qq_llm_bot.storage_records import (
    _dashboard_relationship_to_dict,
    _fact_to_dict,
    _user_profile_to_dict,
)


def build_dashboard_cognition_items(
    conn: sqlite3.Connection,
    candidates: dict[str, dict[str, object]],
    query_limit: int,
) -> list[dict[str, object]]:
    items: list[dict[str, object]] = []
    for entry in candidates.values():
        dashboard_user_id = str(entry["user_id"])
        relationships = list_dashboard_relationship_rows(conn, dashboard_user_id, query_limit)
        fact_records = list_dashboard_user_fact_records(conn, dashboard_user_id, 20)
        member_profile = dashboard_member_profile(conn, dashboard_user_id)
        user_profile = dashboard_user_profile(conn, dashboard_user_id)
        group_ids = entry["group_ids"]
        assert isinstance(group_ids, set)
        for row in relationships:
            relationship_group_id = str(row["group_id"] or "")
            if relationship_group_id:
                group_ids.add(relationship_group_id)
            entry["sort_at"] = max(int(entry["sort_at"]), int(row["updated_at"]))
        for record in fact_records:
            if record.source_group_id:
                group_ids.add(record.source_group_id)
            entry["sort_at"] = max(int(entry["sort_at"]), record.updated_at)
        if member_profile is not None:
            entry["sort_at"] = max(int(entry["sort_at"]), member_profile.updated_at)
        sorted_group_ids = sorted(group_ids)
        items.append(
            {
                "group_id": ", ".join(sorted_group_ids),
                "group_ids": sorted_group_ids,
                "user_id": dashboard_user_id,
                "nickname": user_profile["nickname"],
                "display_name": user_profile["display_name"],
                "relationship": _dashboard_relationship_to_dict(
                    dashboard_user_id,
                    relationships,
                    sorted_group_ids,
                ),
                "profile": _user_profile_to_dict(member_profile),
                "facts": [_fact_to_dict(record) for record in fact_records],
                "updated_at": int(entry["sort_at"]),
            }
        )
    return items
