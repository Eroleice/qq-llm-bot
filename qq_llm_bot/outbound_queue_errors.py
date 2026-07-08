from __future__ import annotations

from nonebot.adapters.onebot.v11.exception import ApiNotAvailable, NetworkError
from nonebot.exception import WebSocketClosed
from websockets.exceptions import ConnectionClosed


def onebot_group_id(group_id: str) -> int | None:
    try:
        return int(str(group_id).strip())
    except ValueError:
        return None


def is_retryable_send_error(exc: BaseException) -> bool:
    if isinstance(exc, (ApiNotAvailable, WebSocketClosed, ConnectionClosed)):
        return True
    if isinstance(exc, NetworkError):
        detail = send_error_detail(exc).lower()
        return "timeout" not in detail
    return isinstance(exc, (ConnectionError, OSError))


def send_error_detail(exc: BaseException) -> str:
    message = getattr(exc, "msg", None) or str(exc) or repr(exc)
    return f"{type(exc).__name__}: {message}"
