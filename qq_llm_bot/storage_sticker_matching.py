from __future__ import annotations

import sqlite3

from qq_llm_bot.models import StickerCandidate
from qq_llm_bot.storage_record_values import (
    _sticker_text_key,
    _useful_sticker_text_key,
)


def find_sticker_asset_row(
    conn: sqlite3.Connection,
    group_id: str,
    sha256: str = "",
    url: str = "",
) -> sqlite3.Row | None:
    if sha256:
        row = conn.execute(
            """
            SELECT id
            FROM sticker_assets
            WHERE group_id = ? AND sha256 = ?
            LIMIT 1
            """,
            (str(group_id), str(sha256)),
        ).fetchone()
        if row is not None:
            return row
    if url:
        return conn.execute(
            """
            SELECT id
            FROM sticker_assets
            WHERE group_id = ? AND url = ?
            LIMIT 1
            """,
            (str(group_id), str(url)),
        ).fetchone()
    return None

def find_similar_sticker_asset_row(
    conn: sqlite3.Connection,
    group_id: str,
    candidate: StickerCandidate,
) -> sqlite3.Row | None:
    ocr_key = _sticker_text_key(candidate.ocr_text)
    if not _useful_sticker_text_key(ocr_key):
        return None
    rows = conn.execute(
        """
        SELECT id, ocr_text
        FROM sticker_assets
        WHERE group_id = ?
          AND ocr_text != ''
        ORDER BY updated_at DESC, id DESC
        LIMIT 300
        """,
        (str(group_id),),
    ).fetchall()
    for row in rows:
        if _sticker_text_key(str(row["ocr_text"] or "")) == ocr_key:
            return row
    return None
