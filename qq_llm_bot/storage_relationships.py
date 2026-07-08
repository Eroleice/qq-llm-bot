from __future__ import annotations

from typing import Any

from qq_llm_bot.storage_records import _relationship_rank_label
from qq_llm_bot.storage_relationship_state import (
    apply_relationship_delta,
    get_relationship,
    touch_relationship,
)

__all__ = [
    "apply_relationship_delta",
    "format_relationship",
    "format_relationship_ranking",
    "get_relationship",
    "touch_relationship",
]


def format_relationship(storage: Any, group_id: str, user_id: str) -> str:
    relation = get_relationship(storage, group_id, user_id)
    return (
        f"QQ={user_id}\n"
        f"closeness={relation.closeness}\n"
        f"trust={relation.trust}\n"
        f"familiarity={relation.familiarity}\n"
        f"tension={relation.tension}\n"
        f"summary={relation.summary or '(empty)'}"
    )


def format_relationship_ranking(storage: Any, group_id: str, limit: int = 5) -> str:
    limit = max(1, min(50, int(limit)))
    with storage._connect() as conn:
        rows = conn.execute(
            """
            SELECT group_id, user_id, closeness, trust, familiarity, tension, updated_at,
                   (closeness + familiarity) AS relationship_score
            FROM relationships
            WHERE group_id = ?
            ORDER BY relationship_score DESC,
                     closeness DESC,
                     familiarity DESC,
                     trust DESC,
                     updated_at DESC
            LIMIT ?
            """,
            (str(group_id), limit),
        ).fetchall()
        profiles = {
            str(row["user_id"]): storage._dashboard_user_profile(conn, str(row["user_id"]))
            for row in rows
        }

    if not rows:
        return "本群暂无关系记录。"

    lines = [f"本群亲密/了解程度 TOP {limit}（按 亲密+了解 排序）："]
    for index, row in enumerate(rows, start=1):
        user_id = str(row["user_id"])
        closeness = int(row["closeness"])
        familiarity = int(row["familiarity"])
        trust = int(row["trust"])
        tension = int(row["tension"])
        score = int(row["relationship_score"])
        label = _relationship_rank_label(user_id, profiles.get(user_id))
        lines.append(
            f"{index}. {label} "
            f"亲密={closeness} 了解={familiarity} 信任={trust} 紧张={tension} 综合={score}"
        )
    return "\n".join(lines)
