from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from qq_llm_bot.config import ParticipationMode
from qq_llm_bot.models import (
    MemoryCandidate,
    MessageContext,
)
from qq_llm_bot.storage_audit_writes import (
    record_decision,
    record_final_qa_block,
    record_llm_usage,
)
from qq_llm_bot.storage_facts import SENSITIVE_CONFIRMATION_KINDS

__all__ = [
    "build_conflict_confirmation",
    "get_last_decision",
    "record_decision",
    "record_final_qa_block",
    "record_llm_usage",
]


def get_last_decision(storage: Any, group_id: str) -> str:
    with storage._connect() as conn:
        row = conn.execute(
            """
            SELECT time, user_id, message_id, mode, action, reason, score,
                   value_type, value_score, traffic_level, reply
            FROM bot_decisions
            WHERE group_id = ?
            ORDER BY id DESC
            LIMIT 1
            """,
            (str(group_id),),
        ).fetchone()
    if row is None:
        return "暂无决策记录。"
    return (
        f"time={row['time']}\n"
        f"user={row['user_id']}\n"
        f"message={row['message_id']}\n"
        f"mode={row['mode']}\n"
        f"action={row['action']}\n"
        f"score={row['score']:.2f}\n"
        f"value={row['value_type'] or 'none'}:{row['value_score']:.2f}\n"
        f"traffic={row['traffic_level'] or 'normal'}\n"
        f"reason={row['reason']}\n"
        f"reply={row['reply'] or '(none)'}"
    )


def build_conflict_confirmation(
    storage: Any,
    conflicts: Iterable[MemoryCandidate],
    context: MessageContext,
    mode: ParticipationMode,
) -> str | None:
    if mode == "silent" or not context.is_direct:
        return None
    for item in conflicts:
        if item.claim_scope not in {"self_report", "bot_directed"}:
            continue
        if item.owner_type != "user" or item.subject_user_id != context.user_id:
            continue
        if item.kind in SENSITIVE_CONFIRMATION_KINDS:
            continue
        if not item.conflict_of:
            continue
        old = storage.get_memory_record(item.conflict_of)
        if old is None:
            continue
        return f"我之前记得你说过「{old.content}」，现在是改成「{item.content}」了吗？"
    return None
