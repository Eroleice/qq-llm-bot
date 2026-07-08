from __future__ import annotations

from typing import Any

from qq_llm_bot.config_models import BotConfig
from qq_llm_bot.config_values import (
    bool_value as _bool_value,
    float_in_range as _float_in_range,
    int_in_range as _int_in_range,
    positive_int as _positive_int,
    string_list as _string_list,
)


def bot_config(raw: dict[str, Any], *, nicknames: list[str], mode: str) -> BotConfig:
    return BotConfig(
        nicknames=nicknames,
        command_start=_string_list(raw.get("command_start", ["#", "/"])),
        admin_ids=_string_list(raw.get("admin_ids", [])),
        ignored_user_ids=_string_list(raw.get("ignored_user_ids", [])),
        enabled_groups=_string_list(raw.get("enabled_groups", [])),
        default_group_mode=mode,  # type: ignore[arg-type]
        proactive_cooldown_seconds=_positive_int(
            raw.get("proactive_cooldown_seconds", 90),
            "bot.proactive_cooldown_seconds",
        ),
        proactive_value_threshold=_float_in_range(
            raw.get("proactive_value_threshold", 0.65),
            "bot.proactive_value_threshold",
            0,
            1,
        ),
        proactive_busy_value_threshold=_float_in_range(
            raw.get("proactive_busy_value_threshold", 0.78),
            "bot.proactive_busy_value_threshold",
            0,
            1,
        ),
        proactive_busy_human_messages=_positive_int(
            raw.get("proactive_busy_human_messages", 6),
            "bot.proactive_busy_human_messages",
        ),
        interaction_followup_seconds=_positive_int(
            raw.get("interaction_followup_seconds", 180),
            "bot.interaction_followup_seconds",
        ),
        max_reply_chars=_positive_int(raw.get("max_reply_chars", 180), "bot.max_reply_chars"),
        reply_style_enabled=_bool_value(raw.get("reply_style_enabled", True)),
        reply_bubbles_enabled=_bool_value(raw.get("reply_bubbles_enabled", True)),
        reply_bubble_trigger_chars=_positive_int(
            raw.get("reply_bubble_trigger_chars", 45),
            "bot.reply_bubble_trigger_chars",
        ),
        reply_bubble_target_chars=_positive_int(
            raw.get("reply_bubble_target_chars", 28),
            "bot.reply_bubble_target_chars",
        ),
        reply_bubble_max_parts=_int_in_range(
            raw.get("reply_bubble_max_parts", 3),
            "bot.reply_bubble_max_parts",
            1,
            3,
        ),
        reply_bubble_delay_seconds=_float_in_range(
            raw.get("reply_bubble_delay_seconds", 0.9),
            "bot.reply_bubble_delay_seconds",
            0,
            10,
        ),
        reply_emoji_cooldown_messages=_int_in_range(
            raw.get("reply_emoji_cooldown_messages", 10),
            "bot.reply_emoji_cooldown_messages",
            0,
            1000,
        ),
        realtime_merge_enabled=_bool_value(raw.get("realtime_merge_enabled", True)),
        realtime_merge_grace_seconds=_float_in_range(
            raw.get("realtime_merge_grace_seconds", 3.0),
            "bot.realtime_merge_grace_seconds",
            0,
            30,
        ),
        realtime_merge_max_messages=_int_in_range(
            raw.get("realtime_merge_max_messages", 5),
            "bot.realtime_merge_max_messages",
            1,
            20,
        ),
        realtime_merge_max_window_seconds=_float_in_range(
            raw.get("realtime_merge_max_window_seconds", 12.0),
            "bot.realtime_merge_max_window_seconds",
            1,
            120,
        ),
        send_retry_enabled=_bool_value(raw.get("send_retry_enabled", True)),
        send_retry_max_attempts=_int_in_range(
            raw.get("send_retry_max_attempts", 6),
            "bot.send_retry_max_attempts",
            1,
            20,
        ),
        send_retry_max_age_seconds=_int_in_range(
            raw.get("send_retry_max_age_seconds", 180),
            "bot.send_retry_max_age_seconds",
            1,
            3600,
        ),
        send_retry_queue_limit=_int_in_range(
            raw.get("send_retry_queue_limit", 100),
            "bot.send_retry_queue_limit",
            1,
            1000,
        ),
        send_retry_base_delay_seconds=_float_in_range(
            raw.get("send_retry_base_delay_seconds", 2.0),
            "bot.send_retry_base_delay_seconds",
            0.1,
            60,
        ),
        send_retry_max_delay_seconds=_float_in_range(
            raw.get("send_retry_max_delay_seconds", 30.0),
            "bot.send_retry_max_delay_seconds",
            0.1,
            300,
        ),
        final_qa_enabled=_bool_value(raw.get("final_qa_enabled", True)),
        context_understanding_enabled=_bool_value(raw.get("context_understanding_enabled", True)),
    )
