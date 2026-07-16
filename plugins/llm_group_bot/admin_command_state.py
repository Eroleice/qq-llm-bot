from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from nonebot.adapters.onebot.v11 import Message

storage: Any = None
config: Any = None
admin_cmd: Any = None
guess_cmd: Any = None
sticker_store: Any = None
llm: Any = None
_finish_command_callback: Callable[[Any, Message | str], Awaitable[None]] | None = None
_update_profiles_callback: Callable[..., Awaitable[None]] | None = None


def configure(
    *,
    storage_: Any,
    config_: Any,
    admin_cmd_: Any,
    guess_cmd_: Any,
    sticker_store_: Any,
    llm_: Any,
    finish_command: Callable[[Any, Message | str], Awaitable[None]],
    update_profiles: Callable[..., Awaitable[None]],
) -> None:
    global storage, config, admin_cmd, guess_cmd, sticker_store, llm
    global _finish_command_callback, _update_profiles_callback
    storage = storage_
    config = config_
    admin_cmd = admin_cmd_
    guess_cmd = guess_cmd_
    sticker_store = sticker_store_
    llm = llm_
    _finish_command_callback = finish_command
    _update_profiles_callback = update_profiles


async def finish_command(matcher: Any, message: Message | str) -> None:
    if _finish_command_callback is None:  # pragma: no cover - plugin setup invariant
        raise RuntimeError("admin command handlers are not configured")
    await _finish_command_callback(matcher, message)


async def maybe_update_profiles(user_ids: list[str], force: bool = False) -> None:
    if _update_profiles_callback is None:  # pragma: no cover - plugin setup invariant
        raise RuntimeError("admin command handlers are not configured")
    await _update_profiles_callback(user_ids, force=force)
