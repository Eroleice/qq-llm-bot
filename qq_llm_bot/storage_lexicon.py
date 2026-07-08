from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from qq_llm_bot.models import MemoryRecord
from qq_llm_bot.storage_records import format_memory_record, _memory_record
from qq_llm_bot.text_utils import lexicon_subject, normalize_lexicon_term


def list_group_lexicon_records(
    storage: Any,
    group_id: str,
    term: str = "",
    limit: int = 12,
    status: str = "active",
) -> list[MemoryRecord]:
    normalized_term = normalize_lexicon_term(term)
    params: list[object] = [str(group_id), str(status)]
    where = [
        "owner_type = 'group'",
        "owner_id = ?",
        "kind = 'lexicon'",
        "status = ?",
    ]
    if normalized_term:
        where.append("(subject_user_id = ? OR content LIKE ?)")
        params.extend([lexicon_subject(normalized_term), f"%{term}%"])
    params.append(int(limit))

    with storage._connect() as conn:
        rows = conn.execute(
            f"""
            SELECT id, owner_type, owner_id, kind, content, confidence, importance, status,
                   updated_at, source_user_id, source_group_id, subject_user_id,
                   claim_scope, verification_status
            FROM memory_items
            WHERE {' AND '.join(where)}
            ORDER BY importance DESC, updated_at DESC, id DESC
            LIMIT ?
            """,
            params,
        ).fetchall()
    return [_memory_record(row) for row in rows]


def list_group_lexicon(
    storage: Any,
    group_id: str,
    term: str = "",
    limit: int = 12,
) -> list[str]:
    return [
        format_memory_record(record)
        for record in storage.list_group_lexicon_records(group_id, term=term, limit=limit)
    ]


def has_group_lexicon(
    storage: Any,
    group_id: str,
    term: str,
    statuses: Iterable[str] = ("active", "pending_confirmation", "conflict"),
) -> bool:
    normalized_term = normalize_lexicon_term(term)
    if not normalized_term:
        return False
    status_list = [str(status) for status in statuses]
    placeholders = ",".join("?" for _ in status_list)
    with storage._connect() as conn:
        row = conn.execute(
            f"""
            SELECT 1
            FROM memory_items
            WHERE owner_type = 'group'
              AND owner_id = ?
              AND kind = 'lexicon'
              AND subject_user_id = ?
              AND status IN ({placeholders})
            LIMIT 1
            """,
            [str(group_id), lexicon_subject(normalized_term), *status_list],
        ).fetchone()
    return row is not None
