from __future__ import annotations

from typing import Iterable

import qq_llm_bot.storage_audit as _storage_audit
from qq_llm_bot.config import ParticipationMode
from qq_llm_bot.llm import LLMUsageRecord
from qq_llm_bot.models import (
    ConversationSnapshot,
    MemoryCandidate,
    MessageContext,
    ParticipationDecision,
)


class StorageCognitionAuditFacadeMixin:
    def record_llm_usage(self, record: LLMUsageRecord) -> None:
        _storage_audit.record_llm_usage(self, record)

    def record_decision(
        self,
        context: MessageContext,
        decision: ParticipationDecision,
        reply: str | None,
    ) -> None:
        _storage_audit.record_decision(self, context, decision, reply)

    def record_final_qa_block(
        self,
        context: MessageContext,
        decision: ParticipationDecision,
        snapshot: ConversationSnapshot,
        *,
        candidate_reply: str,
        qa_reason: str,
        qa_categories: Iterable[str] = (),
        qa_confidence: float = 0.0,
    ) -> None:
        _storage_audit.record_final_qa_block(
            self,
            context,
            decision,
            snapshot,
            candidate_reply=candidate_reply,
            qa_reason=qa_reason,
            qa_categories=qa_categories,
            qa_confidence=qa_confidence,
        )

    def get_last_decision(self, group_id: str) -> str:
        return _storage_audit.get_last_decision(self, group_id)

    def build_conflict_confirmation(
        self,
        conflicts: Iterable[MemoryCandidate],
        context: MessageContext,
        mode: ParticipationMode,
    ) -> str | None:
        return _storage_audit.build_conflict_confirmation(self, conflicts, context, mode)
