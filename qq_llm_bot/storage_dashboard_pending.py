from __future__ import annotations

from typing import Any

from qq_llm_bot.storage_records import _fact_record, _fact_to_dict, _memory_record, _memory_to_dict


def list_dashboard_pending(storage: Any, limit: int = 100) -> list[dict[str, object]]:
    with storage._connect() as conn:
        rows = conn.execute(
            """
            SELECT id, owner_type, owner_id, kind, content, confidence, importance, status,
                   updated_at, source_user_id, source_group_id, subject_user_id,
                   claim_scope, verification_status
            FROM memory_items
            WHERE status IN ('pending_confirmation', 'conflict')
            ORDER BY updated_at DESC, id DESC
            LIMIT ?
            """,
            (int(limit),),
        ).fetchall()
        fact_rows = conn.execute(
            """
            SELECT id, subject_user_id, fact_type, claim_text, topic, stance,
                   confidence, status, claim_scope, source_user_id, source_group_id,
                   evidence_message_id, evidence_text, created_at, updated_at
            FROM member_facts
            WHERE status = 'pending_confirmation'
            ORDER BY updated_at DESC, id DESC
            LIMIT ?
            """,
            (int(limit),),
        ).fetchall()

    items = []
    for row in rows:
        record = _memory_record(row)
        data = _memory_to_dict(record)
        data["item_type"] = "memory"
        prefix = "#bot persona self" if record.owner_type == "self" else "#bot memory"
        data["approve_command"] = f"{prefix} approve {record.id}"
        data["reject_command"] = f"{prefix} reject {record.id}"
        items.append(data)
    for row in fact_rows:
        record = _fact_record(row)
        data = _fact_to_dict(record)
        data["item_type"] = "fact"
        data["approve_command"] = f"#bot facts approve {record.id}"
        data["reject_command"] = f"#bot facts reject {record.id}"
        items.append(data)
    items.sort(key=lambda item: int(item.get("updated_at", 0)), reverse=True)
    return items[:limit]
