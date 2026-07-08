from __future__ import annotations

from dataclasses import asdict
from typing import Any

from fastapi import HTTPException, Request
from fastapi.responses import FileResponse

from qq_llm_bot.cognitive_storage import BotStorage
from qq_llm_bot.config import AppConfig
from qq_llm_bot.dashboard_utils import (
    clamp_limit,
    ensure_dashboard_authorized,
    resolve_sticker_file,
)
from qq_llm_bot.stickers import StickerLocalStore


def register_dashboard_sticker_routes(app: Any, storage: BotStorage, config: AppConfig) -> None:
    api_prefix = config.dashboard.api_prefix
    sticker_store = StickerLocalStore(config)

    @app.get(f"{api_prefix}/stickers")
    async def dashboard_stickers(
        request: Request,
        group_id: str = "",
        limit: int = 200,
    ) -> dict[str, object]:
        ensure_dashboard_authorized(request, config)
        items = storage.list_dashboard_stickers(
            group_id=group_id.strip(),
            limit=clamp_limit(limit, 10, 500),
        )
        return {
            "items": [
                item
                for item in items
                if resolve_sticker_file(config, str(item.get("local_path", ""))) is not None
            ]
        }

    @app.get(f"{api_prefix}/stickers/{{sticker_id}}/image")
    async def dashboard_sticker_image(request: Request, sticker_id: int) -> FileResponse:
        ensure_dashboard_authorized(request, config)
        asset = storage.get_sticker_asset(sticker_id)
        if asset is None or not asset.enabled:
            raise HTTPException(status_code=404, detail="sticker not found")
        image_path = resolve_sticker_file(config, asset.local_path)
        if image_path is None:
            raise HTTPException(status_code=404, detail="sticker image not found")
        return FileResponse(image_path)

    @app.post(f"{api_prefix}/stickers/{{sticker_id}}/enable")
    async def dashboard_enable_sticker(request: Request, sticker_id: int) -> dict[str, object]:
        ensure_dashboard_authorized(request, config)
        if not storage.set_sticker_enabled(sticker_id, True):
            raise HTTPException(status_code=404, detail="sticker not found")
        return {"ok": True}

    @app.post(f"{api_prefix}/stickers/{{sticker_id}}/disable")
    async def dashboard_disable_sticker(request: Request, sticker_id: int) -> dict[str, object]:
        ensure_dashboard_authorized(request, config)
        if not storage.set_sticker_enabled(sticker_id, False):
            raise HTTPException(status_code=404, detail="sticker not found")
        return {"ok": True}

    @app.delete(f"{api_prefix}/stickers/{{sticker_id}}")
    async def dashboard_delete_sticker(request: Request, sticker_id: int) -> dict[str, object]:
        ensure_dashboard_authorized(request, config)
        asset = storage.delete_sticker_asset(sticker_id)
        if asset is None:
            raise HTTPException(status_code=404, detail="sticker not found")
        deleted_file = sticker_store.delete_saved_file(asset.local_path)
        return {"ok": True, "deleted_file": deleted_file, "item": asdict(asset)}
