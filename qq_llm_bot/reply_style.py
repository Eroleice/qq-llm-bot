from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable


EMOJI_RE = re.compile(r"[\U0001F300-\U0001FAFF]")
NONBREAKABLE_RE = re.compile(
    r"https?://\S+|www\.\S+|\[CQ:[^\]]+\]|@[^\s@()]{1,32}\(QQ:\d{5,20}\)|@?QQ:\d{5,20}",
    re.I,
)
EXPLANATION_QUERY_RE = re.compile(
    r"怎么做|为什么|为啥|方案|解释|步骤|怎么实现|如何实现|怎么弄|怎么改|怎么写|原因|思路|讲讲|展开|细说"
)
HIGH_VALUE_TYPES = {"answer", "useful_context", "synthesis"}
TERMINAL_PERIODS = "。."
TERMINAL_KEEP = "？?!！~～…"


@dataclass(frozen=True)
class ReplyStyleSettings:
    enabled: bool = True
    bubbles_enabled: bool = True
    bubble_trigger_chars: int = 45
    bubble_target_chars: int = 28
    bubble_max_parts: int = 3
    emoji_cooldown_messages: int = 10


def settings_from_bot_config(bot_config: object) -> ReplyStyleSettings:
    return ReplyStyleSettings(
        enabled=bool(getattr(bot_config, "reply_style_enabled", True)),
        bubbles_enabled=bool(getattr(bot_config, "reply_bubbles_enabled", True)),
        bubble_trigger_chars=max(1, int(getattr(bot_config, "reply_bubble_trigger_chars", 45))),
        bubble_target_chars=max(1, int(getattr(bot_config, "reply_bubble_target_chars", 28))),
        bubble_max_parts=max(1, min(3, int(getattr(bot_config, "reply_bubble_max_parts", 3)))),
        emoji_cooldown_messages=max(
            0,
            int(getattr(bot_config, "reply_emoji_cooldown_messages", 10)),
        ),
    )


def style_reply_text(
    text: str | None,
    settings: ReplyStyleSettings,
    *,
    action: str = "",
    value_type: str = "",
    trigger_text: str = "",
    has_sticker: bool = False,
    recent_bot_replies: Iterable[str] = (),
) -> str:
    reply = _normalize_layout(text or "")
    if not reply or not settings.enabled:
        return reply

    reply = _apply_emoji_cooldown(
        reply,
        recent_bot_replies,
        settings.emoji_cooldown_messages,
    )
    reply = _normalize_layout(reply)

    allow_long = _allows_long_reply(trigger_text, value_type)
    target = _target_chars(action, value_type, has_sticker)

    if len(reply) <= 60:
        reply = _collapse_lines(reply)
    elif not allow_long:
        reply = _compact_to_target(reply, target)

    if has_sticker:
        reply = _compact_to_target(reply, 20)
        if _sticker_can_replace_text(reply):
            return ""
    elif len(reply) > target and not allow_long:
        reply = _compact_to_target(reply, target)

    return _strip_short_period(_normalize_layout(reply))


def split_reply_bubbles(text: str | None, settings: ReplyStyleSettings) -> tuple[str, ...]:
    reply = _normalize_layout(text or "")
    if not reply:
        return ()
    if not settings.enabled:
        return (reply,)
    if not settings.bubbles_enabled:
        return (_strip_short_period(reply),)

    if "\n" not in reply and len(reply) <= settings.bubble_trigger_chars:
        return (_strip_short_period(reply),)

    protected, tokens = _protect_nonbreakable(reply)
    seed_segments = [item.strip() for item in protected.splitlines() if item.strip()]
    if not seed_segments:
        seed_segments = [protected]

    parts: list[str] = []
    for segment in seed_segments:
        if len(_restore_nonbreakable(segment, tokens)) <= settings.bubble_target_chars:
            parts.append(segment)
            continue
        parts.extend(_split_segment(segment, settings.bubble_target_chars))

    restored = [
        _strip_short_period(_restore_nonbreakable(part, tokens))
        for part in parts
        if _restore_nonbreakable(part, tokens).strip()
    ]
    restored = [part for part in restored if part]
    if not restored:
        return ()
    if len(restored) <= settings.bubble_max_parts:
        return tuple(restored)

    head = restored[: settings.bubble_max_parts - 1]
    tail = _strip_short_period(_collapse_lines("\n".join(restored[settings.bubble_max_parts - 1 :])))
    return tuple([*head, tail] if tail else head)


def _normalize_layout(text: str) -> str:
    text = str(text or "").strip()
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    lines = [" ".join(line.split()) for line in text.splitlines()]
    text = "\n".join(line for line in lines if line)
    return re.sub(r"\n{2,}", "\n", text).strip()


def _collapse_lines(text: str) -> str:
    return " ".join(line.strip() for line in text.splitlines() if line.strip()).strip()


def _target_chars(action: str, value_type: str, has_sticker: bool) -> int:
    if has_sticker:
        return 20
    if action == "proactive_reply":
        return 30
    if value_type == "direct_reply":
        return 25
    return 40


def _allows_long_reply(trigger_text: str, value_type: str) -> bool:
    return value_type in HIGH_VALUE_TYPES or bool(EXPLANATION_QUERY_RE.search(trigger_text or ""))


def _compact_to_target(text: str, target: int) -> str:
    text = _collapse_lines(text)
    if len(text) <= target:
        return text
    clauses = _split_clauses(text)
    if not clauses:
        return _hard_cut(text, target)

    selected: list[str] = []
    for clause in clauses:
        candidate = "".join([*selected, clause]).strip()
        if len(candidate) <= target:
            selected.append(clause)
            continue
        if not selected:
            return _hard_cut(clause, target)
        break
    return "".join(selected).strip() or _hard_cut(text, target)


def _split_clauses(text: str) -> list[str]:
    clauses = re.findall(r"[^。！？!?；;\n]+[。！？!?；;]?", text)
    result: list[str] = []
    for clause in clauses:
        clause = clause.strip()
        if not clause:
            continue
        if len(clause) <= 34:
            result.append(clause)
            continue
        result.extend(item for item in re.split(r"(?<=[，,、])", clause) if item.strip())
    return result


def _hard_cut(text: str, target: int) -> str:
    text = text.strip()
    if len(text) <= target:
        return text
    for index in range(min(len(text), target), max(0, target - 10), -1):
        if text[index - 1] in "，,、；; ":
            return text[:index].rstrip("，,、；; ")
    return text[:target].rstrip()


def _strip_short_period(text: str) -> str:
    text = text.strip()
    if len(text) <= 60 and text.endswith(tuple(TERMINAL_PERIODS)):
        return text.rstrip(TERMINAL_PERIODS).rstrip()
    return text


def _sticker_can_replace_text(text: str) -> bool:
    compact = re.sub(r"[\s，。,.!！?？~～…、\ufe0f\u200d]+", "", text)
    compact = EMOJI_RE.sub("", compact)
    return compact in {
        "",
        "哈哈",
        "哈哈哈",
        "笑死",
        "草",
        "乐",
        "绷",
        "好耶",
        "好好好",
        "懂了",
        "收到",
        "确实",
        "可以",
        "行",
    }


def _apply_emoji_cooldown(
    text: str,
    recent_bot_replies: Iterable[str],
    cooldown_messages: int,
) -> str:
    recent_text = "\n".join(list(recent_bot_replies)[: max(0, cooldown_messages)])
    blocked = set(EMOJI_RE.findall(recent_text))
    kept_current = False

    def replace(match: re.Match[str]) -> str:
        nonlocal kept_current
        emoji = match.group(0)
        if emoji in blocked or kept_current:
            return ""
        kept_current = True
        return emoji

    text = EMOJI_RE.sub(replace, text)
    text = re.sub(r"[\u200d\ufe0f]+(?=\s|$|[。！？!?，,、])", "", text)
    text = re.sub(r"(?<=\s)[\u200d\ufe0f]+", "", text)
    return text.strip()


def _protect_nonbreakable(text: str) -> tuple[str, dict[str, str]]:
    tokens: dict[str, str] = {}

    def replace(match: re.Match[str]) -> str:
        token = f"QQLLMBOTREPLYTOKEN{len(tokens)}X"
        tokens[token] = match.group(0)
        return token

    return NONBREAKABLE_RE.sub(replace, text), tokens


def _restore_nonbreakable(text: str, tokens: dict[str, str]) -> str:
    for token, value in tokens.items():
        text = text.replace(token, value)
    return text.strip()


def _split_segment(segment: str, target: int) -> list[str]:
    clauses = _split_clauses(segment)
    if not clauses:
        return [segment.strip()]
    parts: list[str] = []
    current = ""
    for clause in clauses:
        candidate = f"{current}{clause}".strip()
        if current and len(candidate) > target:
            parts.append(current.strip())
            current = clause.strip()
        else:
            current = candidate
    if current:
        parts.append(current.strip())
    return parts
