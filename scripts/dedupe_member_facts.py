from __future__ import annotations

import argparse
import json
import re
import sqlite3
import time
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Iterable


DEFAULT_DB = Path("data/bot.sqlite3")
DEFAULT_BACKUP_DIR = Path(".tmp/backups")
DEFAULT_REPORT_DIR = Path(".tmp")
ACTIVE_STATUSES = ("accepted", "pending_confirmation")
SENSITIVE_FACT_TYPES = {"identity", "alias", "boundary"}

USER = "\u7528\u6237"
THIS_USER = "\u8be5\u7528\u6237"
HE = "\u4ed6"
SHE = "\u5979"
IMAGE = "\u56fe\u7247"
IN = "\u4e2d"
DE = "\u7684"
CONTENT = "\u5185\u5bb9"
INTERESTED = "\u611f\u5174\u8da3"
RELATED = "\u76f8\u5173"

GENERIC_IMAGE_TOKENS = (
    IMAGE,
    IN,
    DE,
    CONTENT,
    INTERESTED,
    RELATED,
    "\u622a\u56fe",
    "\u9875\u9762",
    "\u7f51\u9875",
    "\u4e3b\u9875",
    "\u754c\u9762",
    "\u56fe\u50cf",
    "\u56fe\u7247\u4e2d",
    "\u56fe\u4e2d",
)
NEGATIVE_TOKENS = (
    "\u4e0d",
    "\u6ca1",
    "\u65e0",
    "\u975e",
    "\u8ba8\u538c",
    "\u53cd\u611f",
    "\u62d2\u7edd",
    "\u4e0d\u559c\u6b22",
    "\u4e0d\u652f\u6301",
    "\u4e0d\u8ba4\u53ef",
)


@dataclass(frozen=True)
class Fact:
    id: int
    subject_user_id: str
    fact_type: str
    claim_text: str
    topic: str
    stance: str
    confidence: float
    status: str
    claim_scope: str
    source_user_id: str
    source_group_id: str
    evidence_message_id: str
    created_at: int
    updated_at: int
    importance: float
    last_seen_at: int


@dataclass(frozen=True)
class DuplicateDecision:
    user_id: str
    fact_type: str
    keep_id: int
    forget_id: int
    score: float
    reason: str
    keep_claim: str
    forget_claim: str
    keep_topic: str
    forget_topic: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find and optionally forget near-duplicate rows in member_facts by user id."
    )
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write changes. Without this flag the script only produces a dry-run report.",
    )
    parser.add_argument(
        "--status",
        action="append",
        choices=ACTIVE_STATUSES,
        help="Statuses to scan. Defaults to accepted and pending_confirmation.",
    )
    parser.add_argument(
        "--user",
        action="append",
        help="Only process the given subject_user_id. Can be passed multiple times.",
    )
    parser.add_argument("--report", type=Path, help="Where to write the JSON report.")
    parser.add_argument("--backup-dir", type=Path, default=DEFAULT_BACKUP_DIR)
    parser.add_argument(
        "--reason",
        default="near_duplicate_cleanup",
        help="Prefix written to forget_reason when --apply is used.",
    )
    return parser.parse_args()


def normalize_basic(value: str, user_id: str = "") -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).lower()
    user_id = unicodedata.normalize("NFKC", str(user_id or "")).lower().strip()
    if user_id:
        for variant in (f"qq:{user_id}", f"qq\uff1a{user_id}", user_id):
            text = text.replace(variant, "")
    for token in (THIS_USER, USER, HE, SHE, "ta"):
        text = text.replace(token, "")
    return re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", text)


def normalize_topic(value: str) -> str:
    return normalize_basic(value)


def bigrams(value: str) -> set[str]:
    if not value:
        return set()
    if len(value) == 1:
        return {value}
    return {value[index : index + 2] for index in range(len(value) - 1)}


def text_score(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    sequence_score = SequenceMatcher(None, left, right).ratio()
    left_bigrams = bigrams(left)
    right_bigrams = bigrams(right)
    dice_score = (
        2 * len(left_bigrams & right_bigrams) / (len(left_bigrams) + len(right_bigrams))
        if left_bigrams and right_bigrams
        else 0.0
    )
    containment_score = 0.0
    if left in right or right in left:
        containment_score = min(len(left), len(right)) / max(len(left), len(right))
    return max(sequence_score, dice_score, containment_score)


def negation_mismatch(left: str, right: str) -> bool:
    left_has = any(token in left for token in NEGATIVE_TOKENS)
    right_has = any(token in right for token in NEGATIVE_TOKENS)
    return left_has != right_has


def topics_match(left: Fact, right: Fact) -> bool:
    left_topic = normalize_topic(left.topic)
    right_topic = normalize_topic(right.topic)
    if not left_topic or not right_topic:
        return False
    if left_topic == right_topic:
        return True
    if left_topic in right_topic or right_topic in left_topic:
        return min(len(left_topic), len(right_topic)) / max(len(left_topic), len(right_topic)) >= 0.75
    return text_score(left_topic, right_topic) >= 0.92


def raw_lower(value: str) -> str:
    return unicodedata.normalize("NFKC", str(value or "")).lower()


def compact_image_topic(value: str) -> str:
    topic = normalize_basic(value)
    for token in GENERIC_IMAGE_TOKENS:
        topic = topic.replace(normalize_basic(token), "")
    return topic


def image_interest_key(fact: Fact) -> str:
    if fact.fact_type != "preference":
        return ""
    claim = raw_lower(fact.claim_text)
    if IMAGE not in claim or INTERESTED not in claim:
        return ""

    candidates: list[str] = []
    if fact.topic:
        candidates.append(fact.topic)

    image_middle_pattern = rf"{IMAGE}\s*{IN}\s*{DE}?(.{{1,60}}?)(?:{CONTENT}|{INTERESTED})"
    related_image_pattern = rf"(.{{1,60}}?){RELATED}\s*{IMAGE}"
    for pattern in (image_middle_pattern, related_image_pattern):
        match = re.search(pattern, claim)
        if match:
            candidates.append(match.group(1))

    for candidate in candidates:
        key = compact_image_topic(candidate)
        if key:
            return key
    return ""


def image_keys_match(left: str, right: str) -> bool:
    if not left or not right:
        return False
    if left == right:
        return True
    if left in right or right in left:
        return min(len(left), len(right)) / max(len(left), len(right)) >= 0.72
    return text_score(left, right) >= 0.90


def duplicate_reason(left: Fact, right: Fact) -> tuple[str, float] | None:
    if left.subject_user_id != right.subject_user_id or left.fact_type != right.fact_type:
        return None
    left_norm = normalize_basic(left.claim_text, left.subject_user_id)
    right_norm = normalize_basic(right.claim_text, right.subject_user_id)
    if not left_norm or not right_norm:
        return None
    if left_norm == right_norm:
        return ("exact_normalized_claim", 1.0)

    score = text_score(left_norm, right_norm)
    if min(len(left_norm), len(right_norm)) < 8 and score < 0.99:
        return None
    if negation_mismatch(left_norm, right_norm) and score < 0.99:
        return None

    left_stance = normalize_basic(left.stance)
    right_stance = normalize_basic(right.stance)
    if left_stance and right_stance and left_stance != right_stance and score < 0.98:
        return None

    left_image_key = image_interest_key(left)
    right_image_key = image_interest_key(right)
    if image_keys_match(left_image_key, right_image_key) and score >= 0.88:
        return ("image_interest_topic", score)

    if left.fact_type in SENSITIVE_FACT_TYPES:
        if score >= 0.965:
            return ("sensitive_high_similarity", score)
        return None

    if score >= 0.975:
        return ("very_high_similarity", score)
    if topics_match(left, right) and score >= 0.93:
        return ("same_topic_similarity", score)
    if left_norm in right_norm or right_norm in left_norm:
        length_ratio = min(len(left_norm), len(right_norm)) / max(len(left_norm), len(right_norm))
        if score >= 0.92 and length_ratio >= 0.75:
            return ("claim_containment", score)
    return None


def keeper_rank(fact: Fact) -> tuple[float, float, int, int, int, int, int]:
    normalized_length = len(normalize_basic(fact.claim_text, fact.subject_user_id))
    return (
        fact.importance,
        fact.confidence,
        normalized_length,
        fact.last_seen_at,
        fact.updated_at,
        fact.created_at,
        fact.id,
    )


def find_duplicates_for_user(facts: list[Fact]) -> list[DuplicateDecision]:
    decisions: list[DuplicateDecision] = []
    by_type: dict[str, list[Fact]] = {}
    for fact in facts:
        by_type.setdefault(fact.fact_type, []).append(fact)

    for same_type_facts in by_type.values():
        ordered = sorted(same_type_facts, key=keeper_rank, reverse=True)
        forgotten_ids: set[int] = set()
        for index, keeper in enumerate(ordered):
            if keeper.id in forgotten_ids:
                continue
            for candidate in ordered[index + 1 :]:
                if candidate.id in forgotten_ids:
                    continue
                reason = duplicate_reason(keeper, candidate)
                if reason is None:
                    continue
                reason_name, score = reason
                forgotten_ids.add(candidate.id)
                decisions.append(
                    DuplicateDecision(
                        user_id=keeper.subject_user_id,
                        fact_type=keeper.fact_type,
                        keep_id=keeper.id,
                        forget_id=candidate.id,
                        score=round(score, 4),
                        reason=reason_name,
                        keep_claim=keeper.claim_text,
                        forget_claim=candidate.claim_text,
                        keep_topic=keeper.topic,
                        forget_topic=candidate.topic,
                    )
                )
    return decisions


def fetch_facts(
    conn: sqlite3.Connection,
    statuses: Iterable[str],
    users: Iterable[str] | None,
) -> dict[str, list[Fact]]:
    status_values = tuple(statuses)
    status_placeholders = ", ".join("?" for _ in status_values)
    where = [f"status IN ({status_placeholders})"]
    params: list[object] = list(status_values)
    user_values = tuple(user for user in (users or ()) if user)
    if user_values:
        user_placeholders = ", ".join("?" for _ in user_values)
        where.append(f"subject_user_id IN ({user_placeholders})")
        params.extend(user_values)
    rows = conn.execute(
        f"""
        SELECT id, subject_user_id, fact_type, claim_text, topic, stance,
               confidence, status, claim_scope, source_user_id, source_group_id,
               evidence_message_id, created_at, updated_at, importance, last_seen_at
        FROM member_facts
        WHERE {' AND '.join(where)}
        ORDER BY subject_user_id ASC, fact_type ASC, id ASC
        """,
        params,
    ).fetchall()

    by_user: dict[str, list[Fact]] = {}
    for row in rows:
        fact = Fact(
            id=int(row["id"]),
            subject_user_id=str(row["subject_user_id"] or ""),
            fact_type=str(row["fact_type"] or ""),
            claim_text=str(row["claim_text"] or ""),
            topic=str(row["topic"] or ""),
            stance=str(row["stance"] or ""),
            confidence=float(row["confidence"] or 0.0),
            status=str(row["status"] or ""),
            claim_scope=str(row["claim_scope"] or ""),
            source_user_id=str(row["source_user_id"] or ""),
            source_group_id=str(row["source_group_id"] or ""),
            evidence_message_id=str(row["evidence_message_id"] or ""),
            created_at=int(row["created_at"] or 0),
            updated_at=int(row["updated_at"] or 0),
            importance=float(row["importance"] or 0.0),
            last_seen_at=int(row["last_seen_at"] or 0),
        )
        if fact.subject_user_id:
            by_user.setdefault(fact.subject_user_id, []).append(fact)
    return by_user


def build_report(
    db_path: Path,
    statuses: tuple[str, ...],
    by_user: dict[str, list[Fact]],
    decisions_by_user: dict[str, list[DuplicateDecision]],
    apply: bool,
    backup_path: Path | None,
) -> dict[str, object]:
    affected_users = {
        user_id: decisions
        for user_id, decisions in decisions_by_user.items()
        if decisions
    }
    return {
        "db": str(db_path.resolve()),
        "apply": apply,
        "statuses": list(statuses),
        "created_at": int(time.time()),
        "backup": str(backup_path.resolve()) if backup_path else "",
        "totals": {
            "users_scanned": len(by_user),
            "facts_scanned": sum(len(facts) for facts in by_user.values()),
            "users_affected": len(affected_users),
            "facts_to_forget": sum(len(decisions) for decisions in affected_users.values()),
        },
        "users": [
            {
                "user_id": user_id,
                "facts_scanned": len(by_user.get(user_id, [])),
                "facts_to_forget": len(decisions),
                "items": [decision.__dict__ for decision in decisions],
            }
            for user_id, decisions in sorted(affected_users.items())
        ],
    }


def backup_db(db_path: Path, backup_dir: Path) -> Path:
    backup_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    backup_path = backup_dir / f"{db_path.stem}.fact-dedupe-{stamp}{db_path.suffix}.bak"
    source = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        destination = sqlite3.connect(backup_path)
        try:
            source.backup(destination)
        finally:
            destination.close()
    finally:
        source.close()
    return backup_path


def apply_decisions(
    conn: sqlite3.Connection,
    decisions: list[DuplicateDecision],
    statuses: tuple[str, ...],
    reason_prefix: str,
) -> int:
    if not decisions:
        return 0
    now = int(time.time())
    status_placeholders = ", ".join("?" for _ in statuses)
    changed = 0
    for decision in decisions:
        reason = f"{reason_prefix}:kept_fact:{decision.keep_id}:score:{decision.score:.4f}"
        cursor = conn.execute(
            f"""
            UPDATE member_facts
            SET status = 'forgotten',
                forget_reason = ?,
                superseded_by_fact_id = COALESCE(superseded_by_fact_id, ?),
                updated_at = ?
            WHERE id = ?
              AND status IN ({status_placeholders})
            """,
            [reason, decision.keep_id, now, decision.forget_id, *statuses],
        )
        if cursor.rowcount <= 0:
            continue
        changed += int(cursor.rowcount)
        conn.execute(
            """
            UPDATE member_aliases
            SET status = 'forgotten',
                updated_at = ?
            WHERE source_fact_id = ?
              AND status = 'active'
            """,
            (now, decision.forget_id),
        )
    refresh_profile_fact_counts(conn, decisions)
    return changed


def refresh_profile_fact_counts(conn: sqlite3.Connection, decisions: list[DuplicateDecision]) -> None:
    by_user: dict[str, set[int]] = {}
    for decision in decisions:
        by_user.setdefault(decision.user_id, set()).add(decision.forget_id)
    for user_id, forgotten_ids in by_user.items():
        count = conn.execute(
            "SELECT COUNT(*) FROM member_facts WHERE subject_user_id = ? AND status = 'accepted'",
            (user_id,),
        ).fetchone()[0]
        row = conn.execute(
            "SELECT supporting_fact_ids FROM member_profiles WHERE user_id = ?",
            (user_id,),
        ).fetchone()
        if row is None:
            continue
        try:
            supporting_ids = json.loads(str(row["supporting_fact_ids"] or "[]"))
        except json.JSONDecodeError:
            supporting_ids = []
        filtered_ids = [
            int(item)
            for item in supporting_ids
            if str(item).strip().isdigit() and int(item) not in forgotten_ids
        ]
        conn.execute(
            """
            UPDATE member_profiles
            SET fact_count = ?,
                supporting_fact_ids = ?
            WHERE user_id = ?
            """,
            (int(count), json.dumps(filtered_ids, ensure_ascii=False), user_id),
        )


def default_report_path(apply: bool) -> Path:
    DEFAULT_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    suffix = "apply" if apply else "dry-run"
    return DEFAULT_REPORT_DIR / f"fact_dedupe_report_{suffix}_{stamp}.json"


def main() -> int:
    args = parse_args()
    db_path = args.db
    if not db_path.exists():
        raise SystemExit(f"Database not found: {db_path}")

    statuses = tuple(args.status or ACTIVE_STATUSES)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        by_user = fetch_facts(conn, statuses, args.user)
        decisions_by_user = {
            user_id: find_duplicates_for_user(facts)
            for user_id, facts in sorted(by_user.items())
        }
        decisions = [
            decision
            for user_id in sorted(decisions_by_user)
            for decision in decisions_by_user[user_id]
        ]
        backup_path = backup_db(db_path, args.backup_dir) if args.apply and decisions else None
        changed = 0
        if args.apply and decisions:
            with conn:
                changed = apply_decisions(conn, decisions, statuses, args.reason)

        report = build_report(
            db_path=db_path,
            statuses=statuses,
            by_user=by_user,
            decisions_by_user=decisions_by_user,
            apply=args.apply,
            backup_path=backup_path,
        )
        report["totals"]["facts_forgotten"] = changed
        report_path = args.report or default_report_path(args.apply)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    finally:
        conn.close()

    totals = report["totals"]
    mode = "APPLY" if args.apply else "DRY-RUN"
    print(
        f"{mode}: scanned {totals['facts_scanned']} facts for {totals['users_scanned']} users; "
        f"affected {totals['users_affected']} users; "
        f"candidates {totals['facts_to_forget']}; forgotten {totals['facts_forgotten']}."
    )
    if backup_path:
        print(f"Backup: {backup_path}")
    print(f"Report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
