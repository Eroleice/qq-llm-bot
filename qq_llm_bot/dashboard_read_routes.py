from __future__ import annotations

from datetime import datetime
from typing import Any

from fastapi import Request
from fastapi.responses import HTMLResponse

from qq_llm_bot.cognitive_storage import BotStorage
from qq_llm_bot.config import AppConfig
from qq_llm_bot.dashboard_assets import DASHBOARD_HTML
from qq_llm_bot.dashboard_utils import (
    clamp_limit,
    date_to_timestamp,
    ensure_dashboard_authorized,
)


def register_dashboard_read_routes(app: Any, storage: BotStorage, config: AppConfig) -> None:
    route_prefix = config.dashboard.route_prefix
    api_prefix = config.dashboard.api_prefix

    @app.get(route_prefix, response_class=HTMLResponse)
    async def dashboard_page(request: Request) -> HTMLResponse:
        ensure_dashboard_authorized(request, config)
        return HTMLResponse(DASHBOARD_HTML.replace("__API_PREFIX__", api_prefix))

    @app.get(f"{api_prefix}/groups")
    async def dashboard_groups(request: Request) -> dict[str, object]:
        ensure_dashboard_authorized(request, config)
        return {"groups": storage.list_dashboard_groups()}

    @app.get(f"{api_prefix}/persona")
    async def dashboard_persona(request: Request) -> dict[str, object]:
        ensure_dashboard_authorized(request, config)
        return storage.get_dashboard_persona()

    @app.get(f"{api_prefix}/users")
    async def dashboard_users(
        request: Request,
        group_id: str = "",
        user_id: str = "",
        limit: int = 100,
    ) -> dict[str, object]:
        ensure_dashboard_authorized(request, config)
        return {
            "items": storage.list_dashboard_user_cognition(
                group_id=group_id.strip(),
                user_id=user_id.strip(),
                limit=clamp_limit(limit, 10, 300),
            )
        }

    @app.get(f"{api_prefix}/messages")
    async def dashboard_messages(
        request: Request,
        group_id: str = "",
        user_id: str = "",
        date_from: str = "",
        date_to: str = "",
        limit: int = 200,
    ) -> dict[str, object]:
        ensure_dashboard_authorized(request, config)
        start_time = date_to_timestamp(date_from.strip(), end=False)
        end_time = date_to_timestamp(date_to.strip(), end=True)
        return {
            "items": storage.list_dashboard_messages(
                group_id=group_id.strip(),
                user_id=user_id.strip(),
                start_time=start_time,
                end_time=end_time,
                limit=clamp_limit(limit, 10, 1000),
            )
        }

    @app.get(f"{api_prefix}/pending")
    async def dashboard_pending(request: Request, limit: int = 100) -> dict[str, object]:
        ensure_dashboard_authorized(request, config)
        return {"items": storage.list_dashboard_pending(limit=clamp_limit(limit, 10, 300))}

    @app.get(f"{api_prefix}/qa-blocks")
    async def dashboard_qa_blocks(
        request: Request,
        group_id: str = "",
        user_id: str = "",
        date_from: str = "",
        date_to: str = "",
        limit: int = 100,
    ) -> dict[str, object]:
        ensure_dashboard_authorized(request, config)
        start_time = date_to_timestamp(date_from.strip(), end=False)
        end_time = date_to_timestamp(date_to.strip(), end=True)
        return {
            "items": storage.list_dashboard_final_qa_blocks(
                group_id=group_id.strip(),
                user_id=user_id.strip(),
                start_time=start_time,
                end_time=end_time,
                limit=clamp_limit(limit, 10, 500),
            )
        }

    @app.get(f"{api_prefix}/llm-usage")
    async def dashboard_llm_usage(
        request: Request,
        hours: int = 24,
        limit: int = 100,
    ) -> dict[str, object]:
        ensure_dashboard_authorized(request, config)
        safe_hours = clamp_limit(hours, 1, 24 * 90)
        now = int(datetime.now().timestamp())
        data = storage.list_dashboard_llm_usage(
            since=now - safe_hours * 3600,
            limit=clamp_limit(limit, 10, 500),
        )
        data["hours"] = safe_hours
        data["now"] = now
        return data
