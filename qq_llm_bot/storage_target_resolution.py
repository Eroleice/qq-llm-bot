from __future__ import annotations

from dataclasses import replace
from typing import Any

from qq_llm_bot.models import MessageContext, TargetUserContext
from qq_llm_bot.onebot_messages import strip_quoted_messages
from qq_llm_bot.storage_helpers import (
    NameResolutionMatch,
    dedupe_short_strings,
    extract_explicit_target_user_ids,
    extract_identity_name_refs,
)
from qq_llm_bot.storage_target_lookup import (
    latest_profile_name_candidates,
    list_active_aliases,
    lookup_alias_users,
    lookup_display_name_users,
    rank_name_candidates,
    recent_group_display_name_candidates,
)

__all__ = [
    "latest_profile_name_candidates",
    "list_active_aliases",
    "lookup_alias_users",
    "lookup_display_name_users",
    "rank_name_candidates",
    "recent_group_display_name_candidates",
    "resolve_target_user_contexts",
]


def resolve_target_user_contexts(
    storage: Any,
    context: MessageContext,
) -> tuple[list[TargetUserContext], list[str], dict[str, list[str]]]:
    context = replace(context, plain_text=strip_quoted_messages(context.plain_text))
    target_reasons: dict[str, str] = {}
    for user_id in extract_explicit_target_user_ids(context):
        target_reasons.setdefault(user_id, "explicit_qq")

    name_refs = extract_identity_name_refs(context.plain_text)
    unknown_refs: list[str] = []
    ambiguous_refs: dict[str, list[str]] = {}
    with storage._connect() as conn:
        for ref in name_refs:
            matches = [
                NameResolutionMatch(user_id, f"alias:{ref}", 1.0)
                for user_id in storage._lookup_alias_users(conn, ref)
            ]
            if not matches:
                matches = storage._lookup_display_name_users(conn, context.group_id, ref)
            if not matches:
                unknown_refs.append(ref)
                continue
            if len(matches) > 1:
                ambiguous_refs[ref] = [
                    match.user_id for match in matches[: storage.target_user_limit]
                ]
                continue
            target_reasons.setdefault(matches[0].user_id, matches[0].reason)

        contexts: list[TargetUserContext] = []
        for user_id, reason in list(target_reasons.items())[: storage.target_user_limit]:
            aliases = storage._list_active_aliases(conn, user_id, limit=12)
            facts = storage.list_user_facts(
                user_id,
                limit=storage.context_fact_limit,
                status="accepted",
            )
            contexts.append(
                TargetUserContext(
                    user_id=user_id,
                    resolution_status=(
                        "guessed"
                        if str(reason).startswith(
                            ("display_name_guess:", "nickname_guess:", "profile_name_guess:")
                        )
                        else "resolved"
                    ),
                    match_reason=reason,
                    aliases=aliases,
                    facts=facts,
                    profile=storage.get_user_profile(user_id),
                )
            )

    return contexts, dedupe_short_strings(unknown_refs), ambiguous_refs
