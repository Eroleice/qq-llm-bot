from __future__ import annotations

import sqlite3

from qq_llm_bot.models import (
    FactRecord,
    ImageVisionCacheRecord,
    MemoryRecord,
    StickerAssetRecord,
    UserProfileRecord,
)
from qq_llm_bot.storage_record_values import (
    _decode_int_list,
    _decode_string_list,
    _decode_traits,
    _row_float,
    _row_int,
    _row_value,
)


def _image_vision_cache_record(row: sqlite3.Row) -> ImageVisionCacheRecord:
    return ImageVisionCacheRecord(
        url=str(row["url"]),
        description=str(row["description"] or ""),
        ocr_text=str(row["ocr_text"] or ""),
        topics=tuple(_decode_string_list(str(row["topics"] or "[]"))),
        memory=str(row["memory"] or ""),
        confidence=float(row["confidence"]),
        importance=float(row["importance"]),
        model=str(row["model"] or ""),
        created_at=int(row["created_at"]),
        updated_at=int(row["updated_at"]),
        last_seen_at=int(row["last_seen_at"]),
        hit_count=int(row["hit_count"]),
    )

def _sticker_asset_record(row: sqlite3.Row) -> StickerAssetRecord:
    return StickerAssetRecord(
        id=int(row["id"]),
        group_id=str(row["group_id"]),
        source_user_id=str(row["source_user_id"] or ""),
        source_message_id=str(row["source_message_id"] or ""),
        url=str(row["url"] or ""),
        file=str(row["file"] or ""),
        local_path=str(row["local_path"] or ""),
        sha256=str(row["sha256"] or ""),
        description=str(row["description"] or ""),
        ocr_text=str(row["ocr_text"] or ""),
        mood=str(row["mood"] or ""),
        usage=str(row["usage"] or ""),
        tags=tuple(_decode_string_list(str(row["tags"] or "[]"))),
        confidence=float(row["confidence"]),
        enabled=bool(int(row["enabled"])),
        created_at=int(row["created_at"]),
        updated_at=int(row["updated_at"]),
        last_seen_at=int(row["last_seen_at"]),
        hit_count=int(row["hit_count"]),
        send_count=_row_int(row, "send_count", 0),
        last_sent_at=_row_int(row, "last_sent_at", 0),
    )

def _memory_record(row: sqlite3.Row) -> MemoryRecord:
    return MemoryRecord(
        id=int(row["id"]),
        owner_type=str(row["owner_type"]),
        owner_id=str(row["owner_id"]),
        kind=str(row["kind"]),
        content=str(row["content"]),
        confidence=float(row["confidence"]),
        importance=float(row["importance"]),
        status=str(row["status"]),
        updated_at=int(row["updated_at"]),
        source_user_id=_row_value(row, "source_user_id", ""),
        source_group_id=_row_value(row, "source_group_id", ""),
        subject_user_id=_row_value(row, "subject_user_id", str(row["owner_id"])),
        claim_scope=_row_value(row, "claim_scope", "self_report"),
        verification_status=_row_value(row, "verification_status", "accepted"),
    )

def _fact_record(row: sqlite3.Row) -> FactRecord:
    superseded_by = _row_value(row, "superseded_by_fact_id", "")
    return FactRecord(
        id=int(row["id"]),
        subject_user_id=str(row["subject_user_id"]),
        fact_type=str(row["fact_type"]),
        claim_text=str(row["claim_text"]),
        topic=str(row["topic"]),
        stance=str(row["stance"] or ""),
        confidence=float(row["confidence"]),
        status=str(row["status"]),
        claim_scope=str(row["claim_scope"]),
        source_user_id=str(row["source_user_id"] or ""),
        source_group_id=str(row["source_group_id"] or ""),
        evidence_message_id=str(row["evidence_message_id"] or ""),
        evidence_text=str(row["evidence_text"] or ""),
        created_at=int(row["created_at"]),
        updated_at=int(row["updated_at"]),
        importance=_row_float(row, "importance", 0.5),
        last_seen_at=_row_int(row, "last_seen_at", int(row["updated_at"])),
        superseded_by_fact_id=int(superseded_by) if superseded_by.strip() else None,
        forget_reason=_row_value(row, "forget_reason", ""),
    )

def _user_profile_record(row: sqlite3.Row) -> UserProfileRecord:
    return UserProfileRecord(
        user_id=str(row["user_id"]),
        summary=str(row["summary"] or ""),
        traits=_decode_traits(str(row["traits_json"] or "{}")),
        supporting_fact_ids=tuple(_decode_int_list(str(row["supporting_fact_ids"] or "[]"))),
        fact_count=int(row["fact_count"]),
        version=int(row["version"]),
        updated_at=int(row["updated_at"]),
    )
