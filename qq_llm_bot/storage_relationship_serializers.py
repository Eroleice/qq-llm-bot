from __future__ import annotations

import sqlite3

from qq_llm_bot.models import RelationshipState
from qq_llm_bot.relationship_summary import merge_relationship_summary
from qq_llm_bot.storage_record_values import _clamp_score


def _relationship_to_dict(relation: RelationshipState) -> dict[str, object]:
    return {
        "group_id": relation.group_id,
        "user_id": relation.user_id,
        "closeness": relation.closeness,
        "trust": relation.trust,
        "familiarity": relation.familiarity,
        "tension": relation.tension,
        "summary": relation.summary,
    }


def _relationship_row_to_dict(row: sqlite3.Row) -> dict[str, object]:
    data = _relationship_to_dict(
        RelationshipState(
            group_id=str(row["group_id"]),
            user_id=str(row["user_id"]),
            closeness=int(row["closeness"]),
            trust=int(row["trust"]),
            familiarity=int(row["familiarity"]),
            tension=int(row["tension"]),
            summary=str(row["summary"] or ""),
        )
    )
    data["updated_at"] = int(row["updated_at"])
    return data


def _relationship_rank_label(user_id: str, profile: dict[str, object] | None) -> str:
    name = ""
    if profile:
        name = str(profile.get("display_name") or profile.get("nickname") or "")
    name = _compact_display_text(name, 24)
    return f"{name}(QQ:{user_id})" if name else f"QQ:{user_id}"


def _compact_display_text(value: str, limit: int) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return f"{text[: max(0, limit - 3)].rstrip()}..."


def _dashboard_relationship_to_dict(
    user_id: str,
    rows: list[sqlite3.Row],
    group_ids: list[str],
) -> dict[str, object] | None:
    if not rows:
        return None
    summary = ""
    updated_at = 0
    closeness = 0
    trust = 0
    familiarity = 0
    tension = 0
    for row in rows:
        closeness += int(row["closeness"])
        trust += int(row["trust"])
        familiarity += int(row["familiarity"])
        tension += int(row["tension"])
        summary = merge_relationship_summary(summary, str(row["summary"] or ""))
        updated_at = max(updated_at, int(row["updated_at"]))
    return {
        "group_id": ", ".join(group_ids),
        "group_ids": group_ids,
        "user_id": user_id,
        "closeness": _clamp_score(closeness),
        "trust": _clamp_score(trust),
        "familiarity": _clamp_score(familiarity),
        "tension": _clamp_score(tension),
        "summary": summary,
        "updated_at": updated_at,
    }
