from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from qq_llm_bot.cognitive_storage import BotStorage
from qq_llm_bot.config import AppConfig
from qq_llm_bot.dashboard_pending_routes import register_dashboard_pending_routes
from qq_llm_bot.dashboard_read_routes import register_dashboard_read_routes
from qq_llm_bot.dashboard_sticker_routes import register_dashboard_sticker_routes


def register_dashboard_routes(
    driver: Any,
    storage: BotStorage,
    config: AppConfig,
    on_fact_changed: Callable[[list[str]], Awaitable[None]] | None = None,
) -> None:
    app = getattr(driver, "server_app", None)
    if app is None:
        return
    if getattr(app.state, "qq_llm_bot_dashboard_registered", False):
        return
    app.state.qq_llm_bot_dashboard_registered = True

    register_dashboard_read_routes(app, storage, config)
    register_dashboard_pending_routes(app, storage, config, on_fact_changed)
    register_dashboard_sticker_routes(app, storage, config)
