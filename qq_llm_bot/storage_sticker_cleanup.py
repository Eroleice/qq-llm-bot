from __future__ import annotations

import time
from typing import Any

from qq_llm_bot.models import StickerAssetRecord
from qq_llm_bot.storage_record_rows import _sticker_asset_record
from qq_llm_bot.storage_record_values import _safe_int
from qq_llm_bot.storage_sticker_asset_queries import STICKER_ASSET_COLUMNS


def claim_sticker_cleanup(
    storage: Any,
    interval_seconds: int,
    now: int | None = None,
) -> bool:
    timestamp = int(now or time.time())
    interval = max(1, int(interval_seconds))
    with storage._connect() as conn:
        row = conn.execute(
            """
            SELECT value
            FROM bot_maintenance_state
            WHERE key = 'sticker_cleanup.last_run_at'
            """
        ).fetchone()
        last_run_at = _safe_int(str(row["value"] or "0"), 0) if row is not None else 0
        if row is not None and timestamp - last_run_at < interval:
            return False
        conn.execute(
            """
            INSERT INTO bot_maintenance_state (key, value, updated_at)
            VALUES ('sticker_cleanup.last_run_at', ?, ?)
            ON CONFLICT(key) DO UPDATE SET
                value = excluded.value,
                updated_at = excluded.updated_at
            """,
            (str(timestamp), timestamp),
        )
    return True


def delete_unused_sticker_assets(
    storage: Any,
    unused_seconds: int,
    now: int | None = None,
) -> list[StickerAssetRecord]:
    timestamp = int(now or time.time())
    cutoff = timestamp - max(1, int(unused_seconds))
    with storage._connect() as conn:
        rows = conn.execute(
            f"""
            SELECT {STICKER_ASSET_COLUMNS}
            FROM sticker_assets
            WHERE local_path != ''
              AND (
                (last_sent_at > 0 AND last_sent_at <= ?)
                OR (last_sent_at = 0 AND created_at <= ?)
              )
            ORDER BY COALESCE(NULLIF(last_sent_at, 0), created_at) ASC, id ASC
            """,
            (cutoff, cutoff),
        ).fetchall()
        assets = [_sticker_asset_record(row) for row in rows]
        if assets:
            ids = [asset.id for asset in assets]
            placeholders = ", ".join("?" for _ in ids)
            conn.execute(f"DELETE FROM sticker_assets WHERE id IN ({placeholders})", ids)
            conn.execute(
                f"DELETE FROM sticker_usage_daily WHERE sticker_id IN ({placeholders})",
                ids,
            )
    return assets
