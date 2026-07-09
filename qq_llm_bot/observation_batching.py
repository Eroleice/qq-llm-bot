from __future__ import annotations

from collections.abc import Sequence

from qq_llm_bot.models import MessageContext


def select_observation_batch_size(
    buffer: Sequence[MessageContext],
    *,
    batch_size: int,
    max_messages_per_batch: int,
    max_interval_seconds: int,
) -> int:
    if not buffer:
        return 0

    hard_limit = min(len(buffer), max(1, int(max_messages_per_batch)))
    if hard_limit >= len(buffer):
        return hard_limit

    natural_cut = _natural_cut_index(
        buffer,
        hard_limit=hard_limit,
        min_size=min(hard_limit, max(1, int(batch_size) // 2)),
        gap_threshold_seconds=_natural_gap_threshold_seconds(max_interval_seconds),
    )
    return natural_cut or hard_limit


def _natural_cut_index(
    buffer: Sequence[MessageContext],
    *,
    hard_limit: int,
    min_size: int,
    gap_threshold_seconds: int,
) -> int:
    best_index = 0
    best_gap = 0
    for index in range(min_size, hard_limit + 1):
        if index >= len(buffer):
            break
        gap = _message_gap_seconds(buffer[index - 1], buffer[index])
        if gap >= gap_threshold_seconds and gap >= best_gap:
            best_index = index
            best_gap = gap
    return best_index


def _natural_gap_threshold_seconds(max_interval_seconds: int) -> int:
    interval = max(1, int(max_interval_seconds))
    return max(60, min(180, interval // 4))


def _message_gap_seconds(previous: MessageContext, current: MessageContext) -> int:
    if previous.timestamp <= 0 or current.timestamp <= 0:
        return 0
    return max(0, current.timestamp - previous.timestamp)
