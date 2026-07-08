from __future__ import annotations

import time
from typing import Any

from qq_llm_bot.models import RelationDelta, RelationshipState
from qq_llm_bot.relationship_summary import merge_relationship_summary
from qq_llm_bot.storage_helpers import merge_summary as _merge_summary
from qq_llm_bot.storage_records import _clamp_score


def get_relationship(storage: Any, group_id: str, user_id: str) -> RelationshipState:
    with storage._connect() as conn:
        row = conn.execute(
            """
            SELECT group_id, user_id, closeness, trust, familiarity, tension, summary
            FROM relationships
            WHERE group_id = ? AND user_id = ?
            """,
            (str(group_id), str(user_id)),
        ).fetchone()
    if row is None:
        return RelationshipState(group_id=str(group_id), user_id=str(user_id))
    return RelationshipState(
        group_id=str(row["group_id"]),
        user_id=str(row["user_id"]),
        closeness=int(row["closeness"]),
        trust=int(row["trust"]),
        familiarity=int(row["familiarity"]),
        tension=int(row["tension"]),
        summary=merge_relationship_summary("", str(row["summary"] or "")),
    )


def apply_relationship_delta(
    storage: Any,
    group_id: str,
    user_id: str,
    delta: RelationDelta,
) -> RelationshipState:
    current = get_relationship(storage, group_id, user_id)
    updated = RelationshipState(
        group_id=str(group_id),
        user_id=str(user_id),
        closeness=_clamp_score(current.closeness + delta.closeness),
        trust=_clamp_score(current.trust + delta.trust),
        familiarity=_clamp_score(current.familiarity + delta.familiarity),
        tension=_clamp_score(current.tension + delta.tension),
        summary=_merge_summary(current.summary, delta.summary_patch),
    )
    with storage._connect() as conn:
        conn.execute(
            """
            INSERT INTO relationships (
                group_id, user_id, closeness, trust, familiarity, tension, summary, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(group_id, user_id) DO UPDATE SET
                closeness = excluded.closeness,
                trust = excluded.trust,
                familiarity = excluded.familiarity,
                tension = excluded.tension,
                summary = excluded.summary,
                updated_at = excluded.updated_at
            """,
            (
                updated.group_id,
                updated.user_id,
                updated.closeness,
                updated.trust,
                updated.familiarity,
                updated.tension,
                updated.summary,
                int(time.time()),
            ),
        )
    return updated


def touch_relationship(
    storage: Any,
    group_id: str,
    user_id: str,
    familiarity_delta: int = 1,
) -> None:
    apply_relationship_delta(
        storage,
        group_id,
        user_id,
        RelationDelta(familiarity=familiarity_delta, reason="message observed"),
    )
