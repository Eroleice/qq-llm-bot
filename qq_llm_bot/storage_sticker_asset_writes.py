from __future__ import annotations

import json
import time
from typing import Any

from qq_llm_bot.models import MessageContext, StickerAssetRecord, StickerCandidate
from qq_llm_bot.storage_record_values import _compact_string_list
from qq_llm_bot.storage_sticker_asset_queries import sticker_asset_by_id
from qq_llm_bot.storage_sticker_matching import (
    find_similar_sticker_asset_row,
    find_sticker_asset_row,
)


def upsert_sticker_asset(
    storage: Any,
    context: MessageContext,
    candidate: StickerCandidate,
    local_path: str,
    sha256: str = "",
) -> StickerAssetRecord | None:
    group_id = str(context.group_id)
    url = str(candidate.url).strip()
    file = str(candidate.file).strip()
    local_path = str(local_path).strip()
    sha256 = str(sha256).strip()
    if not group_id or not local_path:
        return None

    now = int(time.time())
    tags_json = json.dumps(_compact_string_list(candidate.tags, limit=12), ensure_ascii=False)
    with storage._connect() as conn:
        existing = find_sticker_asset_row(conn, group_id, sha256=sha256, url=url)
        preserve_existing_file = False
        if existing is None:
            existing = find_similar_sticker_asset_row(conn, group_id, candidate)
            preserve_existing_file = existing is not None
        if existing is not None:
            asset_id = int(existing["id"])
            next_local_path = "" if preserve_existing_file else local_path
            next_sha256 = "" if preserve_existing_file else sha256
            conn.execute(
                """
                UPDATE sticker_assets
                SET source_user_id = ?,
                    source_message_id = ?,
                    url = CASE WHEN ? != '' THEN ? ELSE url END,
                    file = CASE WHEN ? != '' THEN ? ELSE file END,
                    local_path = CASE WHEN ? != '' THEN ? ELSE local_path END,
                    sha256 = CASE WHEN ? != '' THEN ? ELSE sha256 END,
                    description = ?,
                    ocr_text = ?,
                    mood = ?,
                    usage = ?,
                    tags = ?,
                    confidence = MAX(confidence, ?),
                    updated_at = ?,
                    last_seen_at = ?,
                    hit_count = hit_count + 1
                WHERE id = ?
                """,
                (
                    context.user_id,
                    context.message_id,
                    url,
                    url,
                    file,
                    file,
                    next_local_path,
                    next_local_path,
                    next_sha256,
                    next_sha256,
                    candidate.description[:1000],
                    candidate.ocr_text[:1000],
                    candidate.mood[:80],
                    candidate.usage[:500],
                    tags_json,
                    float(candidate.confidence),
                    now,
                    now,
                    asset_id,
                ),
            )
        else:
            cursor = conn.execute(
                """
                INSERT INTO sticker_assets (
                    group_id, source_user_id, source_message_id, url, file,
                    local_path, sha256, description, ocr_text, mood, usage,
                    tags, confidence, enabled, created_at, updated_at,
                    last_seen_at, hit_count
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?, ?, 1)
                """,
                (
                    group_id,
                    context.user_id,
                    context.message_id,
                    url,
                    file,
                    local_path,
                    sha256,
                    candidate.description[:1000],
                    candidate.ocr_text[:1000],
                    candidate.mood[:80],
                    candidate.usage[:500],
                    tags_json,
                    float(candidate.confidence),
                    now,
                    now,
                    now,
                ),
            )
            asset_id = int(cursor.lastrowid)
        return sticker_asset_by_id(conn, asset_id)
