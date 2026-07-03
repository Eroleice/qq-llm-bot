from __future__ import annotations

import re
from collections.abc import Iterable

MAX_RELATIONSHIP_SUMMARY_ITEMS = 6
MAX_RELATIONSHIP_SUMMARY_CHARS = 360

_LEDGER_PREFIXES = (
    "分享",
    "发送",
    "上传",
    "转发",
    "发表",
    "表达",
    "讨论",
    "提示",
    "解释",
    "描述",
    "展示",
    "回复",
    "转述",
    "发布",
    "贴",
    "晒",
    "用",
)
_LOW_VALUE_EVENT_HINTS = (
    "图片",
    "截图",
    "梗图",
    "表情包",
    "插画",
    "人像",
    "穿搭",
    "空消息",
    "事件开始",
    "纯乐子",
    "60帧",
)
_RELATIONSHIP_HINTS = (
    "可可",
    "机器人",
    "bot",
    "互动",
    "对话",
    "聊过",
    "聊天",
    "主动",
    "经常",
    "多次",
    "反复",
    "持续",
    "习惯",
    "求助",
    "请教",
    "帮忙",
    "感谢",
    "道谢",
    "信任",
    "依赖",
    "质疑机器人",
    "质疑可可",
    "怀疑机器人",
    "怀疑可可",
    "调侃可可",
    "调侃机器人",
    "冒犯",
    "冲突",
    "紧张",
    "不满可可",
    "不满机器人",
    "反馈",
    "纠正",
    "边界",
    "亲近",
    "熟悉",
    "记得",
    "记住",
    "友好",
    "礼貌",
    "配合",
    "耐心",
)
_NO_SIGNAL_HINTS = (
    "空字符串",
    "无明显关系",
    "没有明显关系",
    "无稳定关系",
    "没有稳定关系",
    "无关系洞察",
    "无特别关系",
)


def clean_relationship_summary_patch(value: str) -> str:
    return merge_relationship_summary("", value, max_items=1, max_chars=120)


def merge_relationship_summary(
    current: str,
    patch: str,
    *,
    max_items: int = MAX_RELATIONSHIP_SUMMARY_ITEMS,
    max_chars: int = MAX_RELATIONSHIP_SUMMARY_CHARS,
) -> str:
    items = _relationship_summary_items((current, patch))
    deduped: list[str] = []
    seen: set[str] = set()
    for item in items:
        key = item.casefold()
        if key in seen:
            continue
        deduped.append(item)
        seen.add(key)

    kept = deduped[-max_items:]
    while kept and len("；".join(kept)) > max_chars:
        kept = kept[1:]
    return "；".join(kept)


def _relationship_summary_items(values: Iterable[str]) -> list[str]:
    items: list[str] = []
    for value in values:
        for raw_item in re.split(r"[；;\n]+", str(value or "")):
            item = _normalize_summary_item(raw_item)
            if item and _is_relationship_summary_item(item):
                items.append(item)
    return items


def _normalize_summary_item(value: str) -> str:
    item = " ".join(str(value or "").strip().split())
    item = re.sub(r"^[\-*•\d.、)\s]+", "", item)
    item = re.sub(r"^(关系摘要|摘要|summary_patch)[:：]\s*", "", item, flags=re.I)
    return item[:120].strip()


def _is_relationship_summary_item(item: str) -> bool:
    if not item or item in {"(empty)", "none", "null", "无", "暂无"}:
        return False
    if any(hint in item for hint in _NO_SIGNAL_HINTS):
        return False
    if _has_relationship_hint(item):
        return True
    return not _looks_like_event_log(item)


def _has_relationship_hint(item: str) -> bool:
    lowered = item.casefold()
    return any(hint.casefold() in lowered for hint in _RELATIONSHIP_HINTS)


def _looks_like_event_log(item: str) -> bool:
    if any(item.startswith(prefix) for prefix in _LEDGER_PREFIXES):
        return True
    return any(hint in item for hint in _LOW_VALUE_EVENT_HINTS)
