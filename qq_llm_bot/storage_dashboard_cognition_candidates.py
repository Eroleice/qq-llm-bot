from __future__ import annotations

import sqlite3

from qq_llm_bot.storage_records import _dashboard_user_id, _dashboard_user_id_variants


def collect_dashboard_cognition_candidates(
    conn: sqlite3.Connection,
    requested_group_id: str,
    requested_user_id: str,
    query_limit: int,
) -> dict[str, dict[str, object]]:
    candidates: dict[str, dict[str, object]] = {}

    user_variants = _dashboard_user_id_variants(requested_user_id) if requested_user_id else []
    user_filter_sql = ""
    user_filter_params: list[object] = []
    if user_variants:
        placeholders = ", ".join("?" for _ in user_variants)
        user_filter_sql = f" AND user_id IN ({placeholders})"
        user_filter_params.extend(user_variants)

    fact_user_filter_sql = ""
    fact_user_filter_params: list[object] = []
    if user_variants:
        placeholders = ", ".join("?" for _ in user_variants)
        fact_user_filter_sql = f" AND subject_user_id IN ({placeholders})"
        fact_user_filter_params.extend(user_variants)

    relation_rows = _fetch_relationship_candidate_rows(
        conn,
        requested_group_id,
        user_filter_sql,
        user_filter_params,
        query_limit,
    )
    fact_rows = _fetch_fact_candidate_rows(
        conn,
        requested_group_id,
        fact_user_filter_sql,
        fact_user_filter_params,
        query_limit,
    )
    profile_rows = _fetch_profile_candidate_rows(conn, requested_group_id, user_variants, query_limit)

    for row in relation_rows:
        entry = _candidate_for(candidates, str(row["user_id"]))
        _add_group_id(entry, str(row["group_id"]))
        _bump_sort_at(entry, int(row["updated_at"]))

    for row in fact_rows:
        entry = _candidate_for(candidates, str(row["user_id"]))
        _add_group_id(entry, str(row["source_group_id"] or ""))
        _bump_sort_at(entry, int(row["updated_at"]))

    for row in profile_rows:
        entry = _candidate_for(candidates, str(row["user_id"]))
        _bump_sort_at(entry, int(row["updated_at"]))

    return candidates


def _candidate_for(
    candidates: dict[str, dict[str, object]],
    raw_user_id: str,
) -> dict[str, object]:
    key = _dashboard_user_id(raw_user_id)
    if key not in candidates:
        candidates[key] = {
            "user_id": key,
            "group_ids": set(),
            "sort_at": 0,
        }
    return candidates[key]


def _add_group_id(entry: dict[str, object], group_id: str) -> None:
    if not group_id:
        return
    group_ids = entry["group_ids"]
    assert isinstance(group_ids, set)
    group_ids.add(group_id)


def _bump_sort_at(entry: dict[str, object], updated_at: int) -> None:
    entry["sort_at"] = max(int(entry["sort_at"]), updated_at)


def _fetch_relationship_candidate_rows(
    conn: sqlite3.Connection,
    requested_group_id: str,
    user_filter_sql: str,
    user_filter_params: list[object],
    query_limit: int,
) -> list[sqlite3.Row]:
    relationship_where = "WHERE 1 = 1"
    relationship_params: list[object] = []
    if requested_group_id:
        relationship_where += " AND group_id = ?"
        relationship_params.append(requested_group_id)
    relationship_where += user_filter_sql
    relationship_params.extend(user_filter_params)
    return conn.execute(
        f"""
        SELECT group_id, user_id, closeness, trust, familiarity, tension, summary, updated_at
        FROM relationships
        {relationship_where}
        ORDER BY updated_at DESC
        LIMIT ?
        """,
        [*relationship_params, query_limit],
    ).fetchall()


def _fetch_fact_candidate_rows(
    conn: sqlite3.Connection,
    requested_group_id: str,
    fact_user_filter_sql: str,
    fact_user_filter_params: list[object],
    query_limit: int,
) -> list[sqlite3.Row]:
    fact_where = ["status = 'accepted'"]
    fact_params: list[object] = []
    if requested_group_id:
        fact_where.append("source_group_id = ?")
        fact_params.append(requested_group_id)
    return conn.execute(
        f"""
        SELECT subject_user_id AS user_id, source_group_id, MAX(updated_at) AS updated_at
        FROM member_facts
        WHERE {' AND '.join(fact_where)}
        {fact_user_filter_sql}
        GROUP BY subject_user_id, source_group_id
        ORDER BY updated_at DESC
        LIMIT ?
        """,
        [*fact_params, *fact_user_filter_params, query_limit],
    ).fetchall()


def _fetch_profile_candidate_rows(
    conn: sqlite3.Connection,
    requested_group_id: str,
    user_variants: list[str],
    query_limit: int,
) -> list[sqlite3.Row]:
    if requested_group_id:
        return []
    profile_where = "WHERE 1 = 1"
    profile_params: list[object] = []
    if user_variants:
        placeholders = ", ".join("?" for _ in user_variants)
        profile_where += f" AND user_id IN ({placeholders})"
        profile_params.extend(user_variants)
    return conn.execute(
        f"""
        SELECT user_id, updated_at
        FROM member_profiles
        {profile_where}
        ORDER BY updated_at DESC
        LIMIT ?
        """,
        [*profile_params, query_limit],
    ).fetchall()
