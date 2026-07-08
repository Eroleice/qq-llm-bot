from __future__ import annotations

import sqlite3
from typing import Any

from qq_llm_bot.models import StickerAssetRecord, StickerCandidate
from qq_llm_bot.storage_record_formatters import format_sticker_asset
from qq_llm_bot.storage_record_rows import _sticker_asset_record
from qq_llm_bot.storage_sticker_matching import (
    find_similar_sticker_asset_row,
    find_sticker_asset_row,
)

STICKER_ASSET_COLUMNS = """
    id, group_id, source_user_id, source_message_id, url, file,
    local_path, sha256, description, ocr_text, mood, usage,
    tags, confidence, enabled, created_at, updated_at,
    last_seen_at, hit_count, send_count, last_sent_at
"""


def sticker_asset_by_id(
    conn: sqlite3.Connection,
    sticker_id: int,
) -> StickerAssetRecord | None:
    row = conn.execute(
        f"""
        SELECT {STICKER_ASSET_COLUMNS}
        FROM sticker_assets
        WHERE id = ?
        """,
        (int(sticker_id),),
    ).fetchone()
    return _sticker_asset_record(row) if row is not None else None


def find_existing_sticker_asset(
    storage: Any,
    group_id: str,
    candidate: StickerCandidate,
) -> StickerAssetRecord | None:
    group_id = str(group_id)
    if not group_id:
        return None
    url = str(candidate.url).strip()
    with storage._connect() as conn:
        row = find_sticker_asset_row(conn, group_id, url=url)
        if row is None:
            row = find_similar_sticker_asset_row(conn, group_id, candidate)
        if row is None:
            return None
        return sticker_asset_by_id(conn, int(row["id"]))


def list_sticker_assets(
    storage: Any,
    group_id: str,
    limit: int = 24,
    enabled_only: bool = True,
) -> list[StickerAssetRecord]:
    where = ["group_id = ?", "local_path != ''"]
    params: list[object] = [str(group_id)]
    if enabled_only:
        where.append("enabled = 1")
    params.append(int(limit))
    with storage._connect() as conn:
        rows = conn.execute(
            f"""
            SELECT {STICKER_ASSET_COLUMNS}
            FROM sticker_assets
            WHERE {' AND '.join(where)}
            ORDER BY confidence DESC, hit_count DESC, updated_at DESC, id DESC
            LIMIT ?
            """,
            params,
        ).fetchall()
    return [_sticker_asset_record(row) for row in rows]


def list_stickers(storage: Any, group_id: str, limit: int = 20) -> list[str]:
    assets = list_sticker_assets(storage, group_id, limit=limit, enabled_only=False)
    return [format_sticker_asset(asset) for asset in assets]


def get_sticker_asset(storage: Any, sticker_id: int) -> StickerAssetRecord | None:
    with storage._connect() as conn:
        return sticker_asset_by_id(conn, sticker_id)
