from __future__ import annotations

import sqlite3

from qq_llm_bot.models import (
    FactRecord,
    MemoryRecord,
    StickerAssetRecord,
    UserProfileRecord,
)
from qq_llm_bot.storage_llm_usage_serializers import (
    _llm_usage_group_to_dict as _llm_usage_group_to_dict,
    _llm_usage_row_to_dict as _llm_usage_row_to_dict,
    _llm_usage_summary_to_dict as _llm_usage_summary_to_dict,
)
from qq_llm_bot.storage_relationship_serializers import (
    _compact_display_text as _compact_display_text,
    _dashboard_relationship_to_dict as _dashboard_relationship_to_dict,
    _relationship_rank_label as _relationship_rank_label,
    _relationship_row_to_dict as _relationship_row_to_dict,
    _relationship_to_dict as _relationship_to_dict,
)


def _sticker_asset_to_dict(asset: StickerAssetRecord) -> dict[str, object]:
    usage = asset.usage or asset.description or "(no usage)"
    return {
        "id": asset.id,
        "group_id": asset.group_id,
        "source_user_id": asset.source_user_id,
        "source_message_id": asset.source_message_id,
        "url": asset.url,
        "file": asset.file,
        "local_path": asset.local_path,
        "sha256": asset.sha256,
        "description": asset.description,
        "ocr_text": asset.ocr_text,
        "mood": asset.mood,
        "usage": usage,
        "trigger": usage,
        "tags": list(asset.tags),
        "confidence": asset.confidence,
        "enabled": asset.enabled,
        "created_at": asset.created_at,
        "updated_at": asset.updated_at,
        "last_seen_at": asset.last_seen_at,
        "hit_count": asset.hit_count,
        "send_count": asset.send_count,
        "last_sent_at": asset.last_sent_at,
        "delete_command": f"#bot stickers delete {asset.id}",
    }

def _compact_persona_items(items: dict[str, str]) -> dict[str, str]:
    return {key: value.strip() for key, value in items.items() if value.strip()}

def _memory_to_dict(record: MemoryRecord) -> dict[str, object]:
    return {
        "id": record.id,
        "owner_type": record.owner_type,
        "owner_id": record.owner_id,
        "kind": record.kind,
        "content": record.content,
        "confidence": record.confidence,
        "importance": record.importance,
        "status": record.status,
        "updated_at": record.updated_at,
        "source_user_id": record.source_user_id,
        "source_group_id": record.source_group_id,
        "subject_user_id": record.subject_user_id,
        "claim_scope": record.claim_scope,
        "verification_status": record.verification_status,
    }

def _fact_to_dict(record: FactRecord) -> dict[str, object]:
    return {
        "id": record.id,
        "subject_user_id": record.subject_user_id,
        "fact_type": record.fact_type,
        "claim_text": record.claim_text,
        "topic": record.topic,
        "stance": record.stance,
        "confidence": record.confidence,
        "status": record.status,
        "claim_scope": record.claim_scope,
        "source_user_id": record.source_user_id,
        "source_group_id": record.source_group_id,
        "evidence_message_id": record.evidence_message_id,
        "evidence_text": record.evidence_text,
        "created_at": record.created_at,
        "updated_at": record.updated_at,
        "importance": record.importance,
        "last_seen_at": record.last_seen_at,
        "superseded_by_fact_id": record.superseded_by_fact_id,
        "forget_reason": record.forget_reason,
    }

def _user_profile_to_dict(record: UserProfileRecord | None) -> dict[str, object] | None:
    if record is None:
        return None
    return {
        "user_id": record.user_id,
        "summary": record.summary,
        "traits": record.traits,
        "supporting_fact_ids": list(record.supporting_fact_ids),
        "fact_count": record.fact_count,
        "version": record.version,
        "updated_at": record.updated_at,
    }

def _format_message_context_line(row: sqlite3.Row) -> str:
    name = str(row["sender_name"] or row["user_id"])
    text = str(row["plain_text"] or "").strip()
    if not text:
        return ""
    return f"{name}: {text}"

