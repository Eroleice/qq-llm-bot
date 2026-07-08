from __future__ import annotations

from typing import Any

from qq_llm_bot.storage_record_rows import _sticker_asset_record
from qq_llm_bot.storage_record_serializers import _sticker_asset_to_dict
from qq_llm_bot.storage_sticker_asset_queries import STICKER_ASSET_COLUMNS


def list_dashboard_stickers(
    storage: Any,
    group_id: str = "",
    limit: int = 200,
) -> list[dict[str, object]]:
    where = ["enabled = 1", "local_path != ''"]
    params: list[object] = []
    if group_id:
        where.append("group_id = ?")
        params.append(str(group_id))
    params.append(int(limit))
    with storage._connect() as conn:
        rows = conn.execute(
            f"""
            SELECT {STICKER_ASSET_COLUMNS}
            FROM sticker_assets
            WHERE {' AND '.join(where)}
            ORDER BY group_id, confidence DESC, hit_count DESC, updated_at DESC, id DESC
            LIMIT ?
            """,
            params,
        ).fetchall()
    return [_sticker_asset_to_dict(_sticker_asset_record(row)) for row in rows]
