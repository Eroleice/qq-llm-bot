from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from qq_llm_bot.model_types import ClaimScope, FactStatus, MemoryStatus, VerificationStatus


@dataclass(frozen=True)
class PerceptionResult:
    is_question: bool
    is_self_disclosure: bool
    mentions_bot: bool
    topics: list[str] = field(default_factory=list)
    emotion_hint: str = "neutral"
    confidence: float = 0.5


@dataclass(frozen=True)
class MemoryCandidate:
    owner_type: Literal["user", "self", "group"]
    owner_id: str
    kind: str
    content: str
    confidence: float
    evidence_message_id: str
    importance: float = 0.5
    source_text: str = ""
    status: MemoryStatus = "candidate"
    conflict_of: int | None = None
    source_user_id: str = ""
    source_group_id: str = ""
    subject_user_id: str = ""
    claim_scope: ClaimScope = "self_report"
    verification_status: VerificationStatus = "pending_confirmation"


@dataclass(frozen=True)
class MemoryRecord:
    id: int
    owner_type: str
    owner_id: str
    kind: str
    content: str
    confidence: float
    importance: float
    status: str
    updated_at: int
    source_user_id: str = ""
    source_group_id: str = ""
    subject_user_id: str = ""
    claim_scope: str = "self_report"
    verification_status: str = "accepted"


@dataclass(frozen=True)
class MemoryWriteSet:
    accepted: list[MemoryCandidate] = field(default_factory=list)
    pending: list[MemoryCandidate] = field(default_factory=list)
    conflicts: list[MemoryCandidate] = field(default_factory=list)
    rejected: list[MemoryCandidate] = field(default_factory=list)


@dataclass(frozen=True)
class FactCandidate:
    subject_user_id: str
    fact_type: str
    claim_text: str
    topic: str
    stance: str
    confidence: float
    evidence_message_id: str
    evidence_text: str
    source_user_id: str
    source_group_id: str
    claim_scope: ClaimScope = "self_report"
    status: FactStatus = "candidate"
    importance: float = 0.5


@dataclass(frozen=True)
class FactRecord:
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
    evidence_text: str
    created_at: int
    updated_at: int
    importance: float = 0.5
    last_seen_at: int = 0
    superseded_by_fact_id: int | None = None
    forget_reason: str = ""


@dataclass(frozen=True)
class FactWriteSet:
    accepted: list[FactRecord] = field(default_factory=list)
    pending: list[FactRecord] = field(default_factory=list)
    rejected: list[FactCandidate] = field(default_factory=list)


@dataclass(frozen=True)
class UserProfileRecord:
    user_id: str
    summary: str
    traits: dict[str, object] = field(default_factory=dict)
    supporting_fact_ids: tuple[int, ...] = ()
    fact_count: int = 0
    version: int = 0
    updated_at: int = 0


@dataclass(frozen=True)
class GuessWhoScoreRecord:
    group_id: str
    user_id: str
    correct_count: int
    wrong_count: int
    nickname: str = ""
    updated_at: int = 0


@dataclass(frozen=True)
class TargetUserContext:
    user_id: str
    resolution_status: str
    match_reason: str
    aliases: list[str] = field(default_factory=list)
    facts: list[FactRecord] = field(default_factory=list)
    profile: UserProfileRecord | None = None


@dataclass(frozen=True)
class SemanticContext:
    current_intent: str = ""
    relevant_messages: list[str] = field(default_factory=list)
    resolved_references: list[str] = field(default_factory=list)
    member_context: list[str] = field(default_factory=list)
    uncertain_references: list[str] = field(default_factory=list)
    ignored_noise: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class UserProfileDraft:
    summary: str
    traits: dict[str, object] = field(default_factory=dict)
    supporting_fact_ids: tuple[int, ...] = ()
