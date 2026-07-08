from __future__ import annotations

import json
from typing import Any

from qq_llm_bot.storage_profile_queries import get_user_profile, should_update_user_profile
from qq_llm_bot.storage_profile_writes import clear_user_profile, maybe_update_user_profile
from qq_llm_bot.storage_records import (
    format_fact_record,
)

__all__ = [
    "clear_user_profile",
    "format_user_profile",
    "get_user_profile",
    "maybe_update_user_profile",
    "should_update_user_profile",
]


def format_user_profile(storage: Any, user_id: str) -> str:
    profile = storage.get_user_profile(user_id)
    facts = storage.list_user_facts(user_id, limit=8)
    if profile is None and not facts:
        return "暂无该用户画像。"
    lines = []
    if profile is not None:
        lines.append(
            f"QQ {profile.user_id} profile v{profile.version} "
            f"(facts={profile.fact_count})\n{profile.summary}"
        )
        if profile.traits:
            lines.append(json.dumps(profile.traits, ensure_ascii=False))
    if facts:
        lines.append("近期 FACT：")
        lines.extend(format_fact_record(record) for record in facts)
    return "\n".join(lines)
