from __future__ import annotations

from typing import Any


def list_dashboard_groups(storage: Any) -> list[str]:
    with storage._connect() as conn:
        rows = conn.execute(
            """
            SELECT group_id FROM group_whitelist
            UNION
            SELECT group_id FROM messages
            UNION
            SELECT group_id FROM relationships
            UNION
            SELECT source_group_id AS group_id FROM memory_items WHERE source_group_id != ''
            UNION
            SELECT source_group_id AS group_id FROM member_facts WHERE source_group_id != ''
            UNION
            SELECT group_id FROM sticker_assets
            ORDER BY group_id
            """
        ).fetchall()
    return [str(row["group_id"]) for row in rows if str(row["group_id"] or "").strip()]
