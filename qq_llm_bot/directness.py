from __future__ import annotations

import re
from typing import Iterable


_DIRECT_NAME_PUNCTUATION = tuple(",，:：、.!！?？~～ \t\r\n")
_DIRECT_AFTER_NAME_PREFIXES = (
    "你",
    "妳",
    "您",
    "我",
    "在吗",
    "在不在",
    "来",
    "帮",
    "帮忙",
    "麻烦",
    "请",
    "说",
    "讲",
    "解释",
    "看看",
    "看下",
    "看一下",
    "评价",
    "分析",
    "总结",
    "推荐",
    "查",
    "搜",
    "画",
    "记",
    "算",
    "给我",
    "告诉",
    "回答",
    "回复",
    "觉得",
    "认为",
    "能",
    "可以",
    "要不要",
    "该不该",
    "是不是",
    "怎么",
    "咋",
    "为什么",
    "为啥",
    "谁",
    "什么",
    "哪个",
    "哪",
    "吗",
    "嘛",
    "呢",
)


def text_mentions_bot_name(text: str, nicknames: Iterable[str]) -> bool:
    folded = str(text or "").casefold()
    return any(name.casefold() in folded for name in _clean_nicknames(nicknames))


def looks_like_bot_address(text: str, nicknames: Iterable[str]) -> bool:
    message = str(text or "").strip()
    if not message:
        return False
    for nickname in _clean_nicknames(nicknames):
        if _starts_with_nickname_call(message, nickname):
            return True
        if _ends_with_nickname_call(message, nickname):
            return True
    return False


def _clean_nicknames(nicknames: Iterable[str]) -> tuple[str, ...]:
    return tuple(name for name in (" ".join(str(item or "").split()) for item in nicknames) if name)


def _starts_with_nickname_call(message: str, nickname: str) -> bool:
    for prefix in (nickname, f"@{nickname}", f"＠{nickname}"):
        if not message.casefold().startswith(prefix.casefold()):
            continue
        rest = message[len(prefix) :].lstrip()
        if not rest:
            return True
        if rest.startswith(_DIRECT_NAME_PUNCTUATION):
            return True
        return rest.startswith(_DIRECT_AFTER_NAME_PREFIXES)
    return False


def _ends_with_nickname_call(message: str, nickname: str) -> bool:
    escaped = re.escape(nickname)
    pattern = rf"(?:[?？!！。.,，:：~～\s]|[吗嘛呢吧呀呗喂])[@＠]?{escaped}[。！？?!~～]*$"
    return bool(re.search(pattern, message, re.I))
