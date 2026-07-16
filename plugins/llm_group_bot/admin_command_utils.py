from __future__ import annotations

from qq_llm_bot.config import ParticipationMode
from qq_llm_bot.models import FactRecord
from plugins.llm_group_bot import admin_command_state as _state


def format_user_relation(group_id: str, user_id: str) -> str:
    relation = _state.storage.get_relationship(group_id, user_id)
    return (
        f"group={relation.group_id}\n"
        f"QQ={relation.user_id}\n"
        f"closeness={relation.closeness}\n"
        f"trust={relation.trust}\n"
        f"familiarity={relation.familiarity}\n"
        f"tension={relation.tension}"
    )

def format_user_pending_fact(fact: FactRecord) -> str:
    return (
        f"#{fact.id} [{fact.fact_type}/{fact.claim_scope}] {fact.claim_text}\n"
        f"topic={fact.topic or '-'}, source={fact.source_user_id or '-'}, "
        f"conf={fact.confidence:.2f}\n"
        f"#approval {fact.id} | #reject {fact.id}"
    )

def parse_memory_id(value: str) -> int | None:
    try:
        return int(value.lstrip("#"))
    except ValueError:
        return None

def normalize_mode(value: str) -> ParticipationMode | None:
    mapping = {
        "silent": "silent",
        "静默": "silent",
        "passive": "passive",
        "被动": "passive",
        "被动回复": "passive",
        "active": "active",
        "主动": "active",
        "主动参与": "active",
    }
    mode = mapping.get(value.lower(), mapping.get(value))
    return mode if mode in {"silent", "passive", "active"} else None  # type: ignore[return-value]

def help_text() -> str:
    return (
        "可用指令：\n"
        "#bot status\n"
        "#bot mode silent|passive|active\n"
        "#bot whitelist list|add <group_id>|remove <group_id>\n"
        "#bot admin list|add <qq_id>|remove <qq_id>\n"
        "#bot ignore list|add <qq_id>|remove <qq_id>\n"
        "#bot memory lexicon [term]|pending|conflicts|approve <id>|reject <id>\n"
        "#bot facts user <qq_id>|pending|approve <id>|reject <id>|forget <id>\n"
        "#guess who|hint|@成员|answer|rank [wrong]（群猜人游戏，所有成员可用）\n"
        "#bot profile <qq_id>\n"
        "#bot stickers list [数量]|enable <id>|disable <id>|delete <id>\n"
        "#bot persona show|self [pending|conflicts|approve <id>|reject <id>|forget <id>]\n"
        "#bot _state.llm status|test [prompt]\n"
        "#bot token\n"
        "#bot why\n"
        "#bot relation <qq_id>|top [数量]|rank [数量]\n"
        "#bot forget <memory_id>"
    )

def memory_help_text() -> str:
    return (
        "用法：\n"
        "#bot memory lexicon [term]\n"
        "#bot memory pending\n"
        "#bot memory conflicts\n"
        "#bot memory approve <memory_id>\n"
        "#bot memory reject <memory_id>"
    )

def facts_help_text() -> str:
    return (
        "用法：\n"
        "#bot facts user <qq_id>\n"
        "#bot facts pending\n"
        "#bot facts approve <fact_id>\n"
        "#bot facts reject <fact_id>\n"
        "#bot facts forget <fact_id>\n"
        "#bot profile <qq_id>"
    )

def persona_help_text() -> str:
    return (
        "用法：\n"
        "#bot persona show\n"
        "#bot persona self\n"
        "#bot persona self pending\n"
        "#bot persona self conflicts\n"
        "#bot persona self approve <memory_id>\n"
        "#bot persona self reject <memory_id>\n"
        "#bot persona self forget <memory_id>"
    )
