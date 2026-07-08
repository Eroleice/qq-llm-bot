from __future__ import annotations

import time
from typing import Any

from qq_llm_bot.models import StickerAssetRecord
from qq_llm_bot.storage_sticker_asset_queries import get_sticker_asset


def set_sticker_enabled(storage: Any, sticker_id: int, enabled: bool) -> bool:
    with storage._connect() as conn:
        cursor = conn.execute(
            """
            UPDATE sticker_assets
            SET enabled = ?, updated_at = ?
            WHERE id = ?
            """,
            (1 if enabled else 0, int(time.time()), int(sticker_id)),
        )
    return cursor.rowcount > 0


def delete_sticker_asset(storage: Any, sticker_id: int) -> StickerAssetRecord | None:
    asset = get_sticker_asset(storage, sticker_id)
    if asset is None:
        return None
    with storage._connect() as conn:
        conn.execute("DELETE FROM sticker_assets WHERE id = ?", (int(sticker_id),))
        conn.execute("DELETE FROM sticker_usage_daily WHERE sticker_id = ?", (int(sticker_id),))
    return asset
