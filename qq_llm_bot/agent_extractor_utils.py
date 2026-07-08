from __future__ import annotations

import re


def _extract_topics(text: str) -> list[str]:
    topics = []
    for keyword in ("游戏", "电影", "工作", "学校", "代码", "AI", "LLM", "吃", "旅行", "音乐"):
        if keyword.lower() in text.lower():
            topics.append(keyword)
    return topics[:5]

def _strip_bot_call(text: str, nicknames: list[str]) -> str:
    cleaned = text.strip()
    for nickname in nicknames:
        if not nickname:
            continue
        cleaned = re.sub(
            rf"^\s*@?{re.escape(nickname)}[\s,锛?锛歖*",
            "",
            cleaned,
            count=1,
        ).strip()
    return cleaned

def _emotion_hint(text: str) -> str:
    if any(token in text for token in ("哈哈", "笑死", "开心", "舒服")):
        return "positive"
    if any(token in text for token in ("难受", "烦", "崩溃", "气死")):
        return "negative"
    return "neutral"
