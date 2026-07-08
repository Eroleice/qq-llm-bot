from __future__ import annotations

from dataclasses import replace

from qq_llm_bot.models import MemoryCandidate
from qq_llm_bot.storage_helpers import (
    clamp_float as _clamp_float,
    safe_claim_scope as _safe_claim_scope,
    safe_verification_status as _safe_verification_status,
)


def normalize_memory_candidate(item: MemoryCandidate) -> MemoryCandidate:
    return replace(
        item,
        owner_id=str(item.owner_id),
        kind=item.kind.strip(),
        content=" ".join(item.content.split()),
        confidence=_clamp_float(item.confidence),
        importance=_clamp_float(item.importance),
        source_text=item.source_text.strip(),
        source_user_id=str(item.source_user_id or item.owner_id),
        source_group_id=str(item.source_group_id),
        subject_user_id=str(item.subject_user_id or item.owner_id),
        claim_scope=_safe_claim_scope(item.claim_scope),
        verification_status=_safe_verification_status(item.verification_status),
    )
