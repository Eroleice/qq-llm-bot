from __future__ import annotations

from dataclasses import replace

from qq_llm_bot.models import MessageAttachment, MessageContext
from qq_llm_bot.repetition_guard import GroupTextRepeatGuard, normalize_repeat_text


def _context(**overrides: object) -> MessageContext:
    base = MessageContext(
        group_id="100",
        user_id="42",
        message_id="m1",
        plain_text="coco look",
        raw_message="coco look",
        timestamp=1000,
    )
    return replace(base, **overrides)


def test_group_text_repeat_guard_skips_second_same_text_in_group() -> None:
    guard = GroupTextRepeatGuard(window_seconds=60)
    first = _context()
    second = _context(user_id="43", message_id="m2", timestamp=1005)

    assert guard.is_repeat(first) is False
    assert guard.is_repeat(second) is True


def test_group_text_repeat_guard_tracks_groups_separately() -> None:
    guard = GroupTextRepeatGuard(window_seconds=60)

    assert guard.is_repeat(_context(group_id="100")) is False
    assert guard.is_repeat(_context(group_id="200", message_id="m2")) is False


def test_group_text_repeat_guard_resets_after_window() -> None:
    guard = GroupTextRepeatGuard(window_seconds=10)

    assert guard.is_repeat(_context(timestamp=1000)) is False
    assert guard.is_repeat(_context(message_id="m2", timestamp=1015)) is False
    assert guard.is_repeat(_context(message_id="m3", timestamp=1016)) is True


def test_group_text_repeat_guard_ignores_attachment_messages() -> None:
    guard = GroupTextRepeatGuard(window_seconds=60)
    attachment = MessageAttachment(attachment_type="image", url="https://example.test/a.png")
    with_image = _context(attachments=[attachment])

    assert guard.is_repeat(with_image) is False
    assert guard.is_repeat(_context(message_id="m2", timestamp=1001)) is False


def test_repeat_text_normalization_collapses_spaces_and_width() -> None:
    assert normalize_repeat_text(_context(plain_text="COCO   ping")) == "coco ping"
    assert normalize_repeat_text(_context(plain_text="ＣＯＣＯ ping")) == "coco ping"
