from __future__ import annotations

from dataclasses import replace

from qq_llm_bot.models import FactCandidate
from qq_llm_bot.storage_helpers import (
    clean_fact_field as _clean_fact_field,
    clamp_float as _clamp_float,
    fact_importance as _fact_importance,
    safe_claim_scope as _safe_claim_scope,
    safe_fact_status as _safe_fact_status,
)
from qq_llm_bot.storage_records import _dashboard_user_id


def normalize_fact_candidate(item: FactCandidate) -> FactCandidate:
    source_user_id = _dashboard_user_id(item.source_user_id)
    subject_user_id = _dashboard_user_id(item.subject_user_id)
    claim_scope = _safe_claim_scope(item.claim_scope)
    if claim_scope == "self_report" and subject_user_id and source_user_id and subject_user_id != source_user_id:
        claim_scope = "third_party"
    importance = _fact_importance(item)
    return replace(
        item,
        subject_user_id=subject_user_id,
        fact_type=_clean_fact_field(item.fact_type, 40),
        claim_text=_clean_fact_field(item.claim_text, 300),
        topic=_clean_fact_field(item.topic, 120),
        stance=_clean_fact_field(item.stance, 60),
        confidence=_clamp_float(item.confidence),
        importance=importance,
        status=_safe_fact_status(item.status),
        claim_scope=claim_scope,
        source_user_id=source_user_id,
        source_group_id=str(item.source_group_id).strip(),
        evidence_message_id=str(item.evidence_message_id).strip(),
        evidence_text=_clean_fact_field(item.evidence_text, 1000),
    )
