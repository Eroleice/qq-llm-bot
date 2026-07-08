from __future__ import annotations

import sqlite3
from typing import Any

from qq_llm_bot.storage_helpers import (
    NameResolutionMatch,
    clean_alias,
    dedupe_short_strings,
    display_name_match_score,
    is_reasonable_member_alias,
    select_name_resolution_matches,
)
from qq_llm_bot.storage_records import _dashboard_user_id


def lookup_alias_users(storage: Any, conn: sqlite3.Connection, name: str) -> list[str]:
    alias = clean_alias(name)
    if not alias or not is_reasonable_member_alias(alias):
        return []
    rows = conn.execute(
        """
        SELECT user_id, MAX(confidence) AS confidence, MAX(updated_at) AS updated_at
        FROM member_aliases
        WHERE alias = ?
          AND status = 'active'
        GROUP BY user_id
        ORDER BY confidence DESC, updated_at DESC
        LIMIT ?
        """,
        (alias, storage.target_user_limit + 1),
    ).fetchall()
    return [str(row["user_id"]) for row in rows if str(row["user_id"] or "").strip()]


def lookup_display_name_users(
    storage: Any,
    conn: sqlite3.Connection,
    group_id: str,
    name: str,
) -> list[NameResolutionMatch]:
    ref = clean_alias(name)
    if not ref or not is_reasonable_member_alias(ref):
        return []

    group_matches = storage._rank_name_candidates(
        ref,
        storage._recent_group_display_name_candidates(conn, group_id),
        reason_prefix="display_name_guess",
        source_bonus=0.03,
    )
    selected = select_name_resolution_matches(group_matches, storage.target_user_limit)
    if selected:
        return selected

    profile_matches = storage._rank_name_candidates(
        ref,
        storage._latest_profile_name_candidates(conn),
        reason_prefix="profile_name_guess",
        source_bonus=0.0,
    )
    return select_name_resolution_matches(profile_matches, storage.target_user_limit)


def recent_group_display_name_candidates(
    conn: sqlite3.Connection,
    group_id: str,
) -> list[tuple[str, str, str]]:
    rows = conn.execute(
        """
        SELECT user_id, sender_name, MAX(time) AS last_seen
        FROM messages
        WHERE group_id = ?
          AND sender_name != ''
          AND sender_role != 'bot'
        GROUP BY user_id, sender_name
        ORDER BY last_seen DESC
        LIMIT 80
        """,
        (str(group_id),),
    ).fetchall()
    candidates: list[tuple[str, str, str]] = []
    for row in rows:
        user_id = _dashboard_user_id(str(row["user_id"] or ""))
        display_name = str(row["sender_name"] or "")
        if user_id and display_name:
            candidates.append((user_id, display_name, "display_name"))
    return candidates


def latest_profile_name_candidates(conn: sqlite3.Connection) -> list[tuple[str, str, str]]:
    rows = conn.execute(
        """
        SELECT user_id, nickname, display_name
        FROM user_profiles
        WHERE nickname != '' OR display_name != ''
        ORDER BY last_seen_at DESC
        LIMIT 120
        """
    ).fetchall()
    candidates: list[tuple[str, str, str]] = []
    for row in rows:
        user_id = _dashboard_user_id(str(row["user_id"] or ""))
        if not user_id:
            continue
        display_name = str(row["display_name"] or "")
        nickname = str(row["nickname"] or "")
        if display_name:
            candidates.append((user_id, display_name, "display_name"))
        if nickname and nickname != display_name:
            candidates.append((user_id, nickname, "nickname"))
    return candidates


def rank_name_candidates(
    ref: str,
    candidates: list[tuple[str, str, str]],
    *,
    reason_prefix: str,
    source_bonus: float,
) -> list[NameResolutionMatch]:
    ranked: dict[str, NameResolutionMatch] = {}
    for user_id, candidate_name, name_type in candidates:
        score = display_name_match_score(ref, candidate_name)
        if score < 0.78:
            continue
        score = min(1.0, score + source_bonus)
        reason = f"{reason_prefix}:{ref}->{candidate_name}"
        if name_type == "nickname":
            reason = f"nickname_guess:{ref}->{candidate_name}"
        existing = ranked.get(user_id)
        if existing is None or score > existing.score:
            ranked[user_id] = NameResolutionMatch(
                user_id=user_id,
                reason=reason,
                score=score,
                status="guessed",
            )
    return sorted(ranked.values(), key=lambda item: item.score, reverse=True)


def list_active_aliases(
    storage: Any,
    conn: sqlite3.Connection,
    user_id: str,
    limit: int = 12,
) -> list[str]:
    subject = _dashboard_user_id(user_id)
    if not subject:
        return []
    rows = conn.execute(
        """
        SELECT alias
        FROM member_aliases
        WHERE user_id = ?
          AND status = 'active'
        ORDER BY confidence DESC, updated_at DESC, id DESC
        LIMIT ?
        """,
        (subject, int(limit)),
    ).fetchall()
    return dedupe_short_strings(
        str(row["alias"] or "")
        for row in rows
        if is_reasonable_member_alias(str(row["alias"] or ""))
    )
