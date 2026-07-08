from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import asdict
from typing import Any

from fastapi import HTTPException, Request

from qq_llm_bot.cognitive_storage import BotStorage
from qq_llm_bot.config import AppConfig
from qq_llm_bot.dashboard_utils import (
    ensure_dashboard_authorized,
    notify_fact_changed,
    request_json_object,
)


def register_dashboard_pending_routes(
    app: Any,
    storage: BotStorage,
    config: AppConfig,
    on_fact_changed: Callable[[list[str]], Awaitable[None]] | None = None,
) -> None:
    api_prefix = config.dashboard.api_prefix

    @app.post(f"{api_prefix}/pending/bulk")
    async def dashboard_bulk_pending(request: Request) -> dict[str, object]:
        ensure_dashboard_authorized(request, config)
        payload = await request_json_object(request)
        action = str(payload.get("action", "")).strip().lower()
        if action not in {"approve", "reject"}:
            raise HTTPException(status_code=400, detail="action must be approve or reject")
        raw_items = payload.get("items", [])
        if not isinstance(raw_items, list) or not raw_items:
            raise HTTPException(status_code=400, detail="items are required")

        changed_user_ids: list[str] = []
        results: list[dict[str, object]] = []
        for raw_item in raw_items:
            if not isinstance(raw_item, dict):
                results.append({"ok": False, "detail": "invalid item"})
                continue
            item_type = str(raw_item.get("item_type") or raw_item.get("type") or "memory").strip()
            try:
                item_id = int(raw_item.get("id", 0))
            except (TypeError, ValueError):
                results.append({"ok": False, "item_type": item_type, "detail": "invalid id"})
                continue

            if item_type == "fact":
                if action == "approve":
                    record = storage.approve_fact(item_id)
                    ok = record is not None
                    if record is not None and record.subject_user_id:
                        changed_user_ids.append(record.subject_user_id)
                else:
                    ok = storage.reject_fact(item_id)
            elif item_type == "memory":
                ok = storage.approve_memory(item_id) if action == "approve" else storage.reject_memory(item_id)
            else:
                ok = False
            results.append({"ok": ok, "item_type": item_type, "id": item_id})

        if changed_user_ids and on_fact_changed is not None:
            await on_fact_changed(changed_user_ids)
        success_count = sum(1 for item in results if item.get("ok"))
        return {"ok": success_count > 0, "count": success_count, "results": results}

    @app.post(f"{api_prefix}/facts/{{fact_id}}/approve")
    async def dashboard_approve_fact(request: Request, fact_id: int) -> dict[str, object]:
        ensure_dashboard_authorized(request, config)
        record = storage.approve_fact(fact_id)
        if record is None:
            raise HTTPException(status_code=404, detail="pending fact not found")
        await notify_fact_changed(on_fact_changed, record)
        return {"ok": True, "item": asdict(record)}

    @app.post(f"{api_prefix}/facts/{{fact_id}}/reject")
    async def dashboard_reject_fact(request: Request, fact_id: int) -> dict[str, object]:
        ensure_dashboard_authorized(request, config)
        if not storage.reject_fact(fact_id):
            raise HTTPException(status_code=404, detail="pending fact not found")
        return {"ok": True}

    @app.post(f"{api_prefix}/facts/{{fact_id}}/forget")
    async def dashboard_forget_fact(request: Request, fact_id: int) -> dict[str, object]:
        ensure_dashboard_authorized(request, config)
        record = storage.forget_fact(fact_id, reason="dashboard")
        if record is None:
            raise HTTPException(status_code=404, detail="fact not found")
        await notify_fact_changed(on_fact_changed, record)
        return {"ok": True, "item": asdict(record)}

    @app.delete(f"{api_prefix}/facts/{{fact_id}}")
    async def dashboard_delete_fact(request: Request, fact_id: int) -> dict[str, object]:
        return await dashboard_forget_fact(request, fact_id)

    @app.post(f"{api_prefix}/memories/{{memory_id}}/approve")
    async def dashboard_approve_memory(request: Request, memory_id: int) -> dict[str, object]:
        ensure_dashboard_authorized(request, config)
        if not storage.approve_memory(memory_id):
            raise HTTPException(status_code=404, detail="pending memory not found")
        return {"ok": True}

    @app.post(f"{api_prefix}/memories/{{memory_id}}/reject")
    async def dashboard_reject_memory(request: Request, memory_id: int) -> dict[str, object]:
        ensure_dashboard_authorized(request, config)
        if not storage.reject_memory(memory_id):
            raise HTTPException(status_code=404, detail="pending memory not found")
        return {"ok": True}

    @app.post(f"{api_prefix}/memories/{{memory_id}}/forget")
    async def dashboard_forget_memory(request: Request, memory_id: int) -> dict[str, object]:
        ensure_dashboard_authorized(request, config)
        if not storage.forget_memory(memory_id):
            raise HTTPException(status_code=404, detail="memory not found")
        return {"ok": True}
