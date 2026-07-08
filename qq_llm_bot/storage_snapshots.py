from __future__ import annotations

from typing import Any

from qq_llm_bot.models import ConversationSnapshot, MessageContext
from qq_llm_bot.storage_target_resolution import (
    latest_profile_name_candidates as latest_profile_name_candidates,
    list_active_aliases as list_active_aliases,
    lookup_alias_users as lookup_alias_users,
    lookup_display_name_users as lookup_display_name_users,
    rank_name_candidates as rank_name_candidates,
    recent_group_display_name_candidates as recent_group_display_name_candidates,
    resolve_target_user_contexts as resolve_target_user_contexts,
)


__all__ = [
    "build_snapshot",
    "latest_profile_name_candidates",
    "list_active_aliases",
    "lookup_alias_users",
    "lookup_display_name_users",
    "rank_name_candidates",
    "recent_group_display_name_candidates",
    "resolve_target_user_contexts",
]


def build_snapshot(storage: Any, context: MessageContext) -> ConversationSnapshot:
    human_count, bot_count = storage.get_recent_activity_counts(context.group_id)
    target_users, unknown_refs, ambiguous_refs = storage._resolve_target_user_contexts(context)
    speaker_messages, other_messages = storage.get_focused_recent_messages(
        context.group_id,
        context.user_id,
    )
    recent_bot_reply, recent_bot_reply_seconds = storage.get_recent_bot_reply_to_user(
        context.group_id,
        context.user_id,
        storage.interaction_followup_seconds,
    )
    return ConversationSnapshot(
        recent_messages=storage.get_recent_messages(context.group_id, limit=12),
        speaker_recent_messages=speaker_messages,
        other_recent_messages=other_messages,
        recent_bot_reply_to_user=recent_bot_reply,
        recent_bot_reply_to_user_seconds=recent_bot_reply_seconds,
        recent_human_messages_60s=human_count,
        recent_bot_messages_120s=bot_count,
        recent_image_descriptions=storage.get_recent_image_descriptions(context.group_id, limit=8),
        sticker_assets=storage.list_sticker_assets(
            context.group_id,
            limit=storage.sticker_context_limit,
        ),
        user_memories=[],
        user_facts=storage.list_user_facts(context.user_id, limit=20),
        user_profile=storage.get_user_profile(context.user_id),
        self_memories=storage.list_memories("self", "bot", limit=12),
        group_reflections=storage.list_memories("group", context.group_id, limit=3),
        group_lexicon=storage.list_group_lexicon_records(context.group_id, limit=10),
        relationship=storage.get_relationship(context.group_id, context.user_id),
        persona_lines=storage.get_persona_lines(),
        target_users=target_users,
        unknown_name_refs=unknown_refs,
        ambiguous_name_refs=ambiguous_refs,
    )
