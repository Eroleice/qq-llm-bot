from __future__ import annotations

import os
from collections.abc import Awaitable, Callable
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from fastapi import HTTPException, Request

from qq_llm_bot.config import AppConfig
from qq_llm_bot.models import FactRecord


def ensure_dashboard_authorized(request: Request, config: AppConfig) -> None:
    expected = dashboard_token(config)
    if not expected:
        return
    supplied = request.query_params.get("token") or request.headers.get("x-dashboard-token", "")
    if supplied != expected:
        raise HTTPException(status_code=401, detail="dashboard token is required")


def dashboard_token(config: AppConfig) -> str:
    return config.dashboard.access_token or os.getenv(config.dashboard.access_token_env, "")


async def request_json_object(request: Request) -> dict[str, Any]:
    try:
        payload = await request.json()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="invalid json body") from exc
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="json body must be an object")
    return payload


async def notify_fact_changed(
    on_fact_changed: Callable[[list[str]], Awaitable[None]] | None,
    record: FactRecord,
) -> None:
    if on_fact_changed is None or not record.subject_user_id:
        return
    await on_fact_changed([record.subject_user_id])


def date_to_timestamp(value: str, end: bool) -> int | None:
    if not value:
        return None
    try:
        day = datetime.strptime(value, "%Y-%m-%d")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"invalid date: {value}") from exc
    if end:
        day = day + timedelta(days=1)
    return int(day.timestamp())


def clamp_limit(value: int, minimum: int, maximum: int) -> int:
    return max(minimum, min(maximum, int(value)))


def resolve_sticker_file(config: AppConfig, local_path: str) -> Path | None:
    raw_path = str(local_path).strip()
    if not raw_path:
        return None
    try:
        path = Path(raw_path).resolve()
        root = config.resolve_path(config.stickers.storage_dir).resolve()
    except OSError:
        return None
    if not path.is_relative_to(root):
        return None
    if not path.exists() or not path.is_file():
        return None
    return path
