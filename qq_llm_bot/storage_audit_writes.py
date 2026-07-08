from __future__ import annotations

import json
import time
from collections.abc import Iterable
from typing import Any

from qq_llm_bot.llm import LLMUsageRecord
from qq_llm_bot.models import ConversationSnapshot, MessageContext, ParticipationDecision
from qq_llm_bot.storage_helpers import clamp_float
from qq_llm_bot.storage_records import _compact_string_list


def record_llm_usage(storage: Any, record: LLMUsageRecord) -> None:
    created_at = int(record.created_at or time.time())
    with storage._connect() as conn:
        conn.execute(
            """
            INSERT INTO llm_usage (
                created_at, purpose, model, prompt_chars, completion_chars,
                prompt_tokens, completion_tokens, total_tokens
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                created_at,
                str(record.purpose)[:80],
                str(record.model)[:120],
                max(0, int(record.prompt_chars)),
                max(0, int(record.completion_chars)),
                max(0, int(record.prompt_tokens)),
                max(0, int(record.completion_tokens)),
                max(0, int(record.total_tokens)),
            ),
        )


def record_decision(
    storage: Any,
    context: MessageContext,
    decision: ParticipationDecision,
    reply: str | None,
) -> None:
    with storage._connect() as conn:
        conn.execute(
            """
            INSERT INTO bot_decisions (
                time, group_id, user_id, message_id, mode, action, reason, score,
                value_type, value_score, traffic_level, reply
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                int(time.time()),
                context.group_id,
                context.user_id,
                context.message_id,
                decision.mode,
                decision.action,
                decision.reason,
                decision.score,
                decision.value_type,
                decision.value_score,
                decision.traffic_level,
                reply or "",
            ),
        )


def record_final_qa_block(
    storage: Any,
    context: MessageContext,
    decision: ParticipationDecision,
    snapshot: ConversationSnapshot,
    *,
    candidate_reply: str,
    qa_reason: str,
    qa_categories: Iterable[str] = (),
    qa_confidence: float = 0.0,
) -> None:
    candidate = str(candidate_reply or "").strip()
    if not candidate:
        return
    now = int(time.time())
    with storage._connect() as conn:
        conn.execute(
            """
            INSERT INTO final_qa_blocks (
                created_at, message_time, group_id, user_id, message_id,
                sender_name, sender_role, trigger_text, raw_message,
                is_direct, bot_mentioned, mode, action, decision_reason,
                score, value_type, value_score, traffic_level,
                candidate_reply, qa_reason, qa_categories, qa_confidence,
                recent_messages, speaker_recent_messages, other_recent_messages,
                recent_image_descriptions
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                now,
                int(context.timestamp or 0),
                context.group_id,
                context.user_id,
                context.message_id,
                context.sender_name,
                context.sender_role,
                context.plain_text,
                context.raw_message,
                1 if context.is_direct else 0,
                1 if context.bot_mentioned else 0,
                decision.mode,
                decision.action,
                decision.reason,
                decision.score,
                decision.value_type,
                decision.value_score,
                decision.traffic_level,
                candidate,
                str(qa_reason or "").strip(),
                json.dumps(_compact_string_list(qa_categories, limit=20), ensure_ascii=False),
                clamp_float(float(qa_confidence or 0.0)),
                json.dumps(_compact_string_list(snapshot.recent_messages, limit=30), ensure_ascii=False),
                json.dumps(_compact_string_list(snapshot.speaker_recent_messages, limit=30), ensure_ascii=False),
                json.dumps(_compact_string_list(snapshot.other_recent_messages, limit=30), ensure_ascii=False),
                json.dumps(_compact_string_list(snapshot.recent_image_descriptions, limit=30), ensure_ascii=False),
            ),
        )
