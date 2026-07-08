from __future__ import annotations

import re

from qq_llm_bot.onebot_message_types import (
    FORWARDED_RECORD_END,
    FORWARDED_RECORD_START,
    QUOTED_MESSAGE_END,
    QUOTED_MESSAGE_START,
)


def strip_forwarded_records(text: str) -> str:
    return _strip_marked_block(text, FORWARDED_RECORD_START, FORWARDED_RECORD_END)


def strip_quoted_messages(text: str) -> str:
    return _strip_marked_block(text, QUOTED_MESSAGE_START, QUOTED_MESSAGE_END)


def _strip_marked_block(text: str, start_marker: str, end_marker: str) -> str:
    pattern = re.compile(
        rf"{re.escape(start_marker)}.*?{re.escape(end_marker)}",
        re.S,
    )
    return "\n".join(line for line in pattern.sub("", text).splitlines() if line.strip()).strip()
