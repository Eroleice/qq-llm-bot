from __future__ import annotations

import time
import unicodedata
from dataclasses import dataclass

from qq_llm_bot.models import MessageContext


@dataclass
class _RepeatState:
    last_seen_at: float
    count: int = 1


class GroupTextRepeatGuard:
    def __init__(
        self,
        *,
        window_seconds: float = 60.0,
        max_entries_per_group: int = 128,
    ) -> None:
        self.window_seconds = max(1.0, float(window_seconds))
        self.max_entries_per_group = max(1, int(max_entries_per_group))
        self._recent: dict[str, dict[str, _RepeatState]] = {}

    def is_repeat(self, context: MessageContext) -> bool:
        key = normalize_repeat_text(context)
        if not key:
            return False

        now = float(context.timestamp or time.time())
        entries = self._recent.setdefault(context.group_id, {})
        self._prune(entries, now)

        state = entries.get(key)
        if state is None or now - state.last_seen_at > self.window_seconds:
            entries[key] = _RepeatState(last_seen_at=now)
            self._trim(entries)
            return False

        state.last_seen_at = now
        state.count += 1
        return True

    def _prune(self, entries: dict[str, _RepeatState], now: float) -> None:
        expired = [
            key
            for key, state in entries.items()
            if now - state.last_seen_at > self.window_seconds
        ]
        for key in expired:
            entries.pop(key, None)

    def _trim(self, entries: dict[str, _RepeatState]) -> None:
        overflow = len(entries) - self.max_entries_per_group
        if overflow <= 0:
            return
        oldest_keys = sorted(entries, key=lambda key: entries[key].last_seen_at)
        for key in oldest_keys[:overflow]:
            entries.pop(key, None)


def normalize_repeat_text(context: MessageContext) -> str:
    if context.sender_role == "bot" or context.attachments:
        return ""
    text = unicodedata.normalize("NFKC", context.plain_text or "")
    return " ".join(text.split()).casefold()
