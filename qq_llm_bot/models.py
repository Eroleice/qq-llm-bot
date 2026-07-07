from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from qq_llm_bot.config import ParticipationMode

DecisionAction = Literal["observe", "reply", "proactive_reply"]
ParticipationValueType = Literal[
    "none",
    "direct_reply",
    "answer",
    "synthesis",
    "missing_angle",
    "useful_context",
    "clarifying_question",
    "humor",
    "agreement",
    "empathy",
    "rephrase",
]
MemoryStatus = Literal["active", "candidate", "pending_confirmation", "conflict", "rejected", "forgotten"]
ClaimScope = Literal["self_report", "third_party", "bot_directed", "group_fact"]
VerificationStatus = Literal["accepted", "pending_confirmation", "conflict", "rejected"]
FactStatus = Literal["candidate", "accepted", "pending_confirmation", "rejected", "superseded", "forgotten"]


@dataclass(frozen=True)
class MessageAttachment:
    attachment_type: Literal["image"]
    file: str = ""
    url: str = ""
    summary: str = ""
    raw_data: str = ""


@dataclass(frozen=True)
class MessageMention:
    user_id: str
    display_name: str = ""
    is_bot: bool = False
    raw_data: str = ""


@dataclass(frozen=True)
class ImageVisionCacheRecord:
    url: str
    description: str
    ocr_text: str = ""
    topics: tuple[str, ...] = ()
    memory: str = ""
    confidence: float = 0.0
    importance: float = 0.5
    model: str = ""
    created_at: int = 0
    updated_at: int = 0
    last_seen_at: int = 0
    hit_count: int = 0


@dataclass(frozen=True)
class StickerCandidate:
    url: str
    file: str = ""
    description: str = ""
    ocr_text: str = ""
    mood: str = ""
    usage: str = ""
    tags: tuple[str, ...] = ()
    confidence: float = 0.0


@dataclass(frozen=True)
class StickerAssetRecord:
    id: int
    group_id: str
    source_user_id: str
    source_message_id: str
    url: str
    file: str
    local_path: str
    sha256: str
    description: str
    ocr_text: str
    mood: str
    usage: str
    tags: tuple[str, ...]
    confidence: float
    enabled: bool
    created_at: int
    updated_at: int
    last_seen_at: int
    hit_count: int = 0
    send_count: int = 0
    last_sent_at: int = 0


@dataclass(frozen=True)
class MessageContext:
    group_id: str
    user_id: str
    message_id: str
    plain_text: str
    raw_message: str
    sender_name: str = ""
    sender_nickname: str = ""
    sender_role: str = ""
    is_direct: bool = False
    bot_mentioned: bool = False
    timestamp: int = 0
    attachments: list[MessageAttachment] = field(default_factory=list)
    mentions: list[MessageMention] = field(default_factory=list)


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
class TargetUserContext:
    user_id: str
    resolution_status: str
    match_reason: str
    aliases: list[str] = field(default_factory=list)
    facts: list[FactRecord] = field(default_factory=list)
    profile: UserProfileRecord | None = None


@dataclass(frozen=True)
class UserProfileDraft:
    summary: str
    traits: dict[str, object] = field(default_factory=dict)
    supporting_fact_ids: tuple[int, ...] = ()


@dataclass(frozen=True)
class RelationshipState:
    group_id: str
    user_id: str
    closeness: int = 0
    trust: int = 0
    familiarity: int = 0
    tension: int = 0
    summary: str = ""


@dataclass(frozen=True)
class RelationDelta:
    closeness: int = 0
    trust: int = 0
    familiarity: int = 0
    tension: int = 0
    summary_patch: str = ""
    reason: str = ""


@dataclass(frozen=True)
class ConversationSnapshot:
    recent_messages: list[str] = field(default_factory=list)
    recent_human_messages_60s: int = 0
    recent_bot_messages_120s: int = 0
    recent_image_descriptions: list[str] = field(default_factory=list)
    sticker_assets: list[StickerAssetRecord] = field(default_factory=list)
    user_memories: list[MemoryRecord] = field(default_factory=list)
    user_facts: list[FactRecord] = field(default_factory=list)
    user_profile: UserProfileRecord | None = None
    self_memories: list[MemoryRecord] = field(default_factory=list)
    group_reflections: list[MemoryRecord] = field(default_factory=list)
    group_lexicon: list[MemoryRecord] = field(default_factory=list)
    relationship: RelationshipState | None = None
    persona_lines: list[str] = field(default_factory=list)
    target_users: list[TargetUserContext] = field(default_factory=list)
    unknown_name_refs: list[str] = field(default_factory=list)
    ambiguous_name_refs: dict[str, list[str]] = field(default_factory=dict)
    speaker_recent_messages: list[str] = field(default_factory=list)
    other_recent_messages: list[str] = field(default_factory=list)
    recent_bot_reply_to_user: str = ""
    recent_bot_reply_to_user_seconds: int = 0


@dataclass(frozen=True)
class ParticipationDecision:
    action: DecisionAction
    reason: str
    mode: ParticipationMode
    score: float = 0.0
    value_type: ParticipationValueType = "none"
    value_score: float = 0.0
    traffic_level: str = "normal"


@dataclass(frozen=True)
class ReplyDraft:
    text: str | None = None
    self_memory_candidates: list[MemoryCandidate] = field(default_factory=list)


@dataclass(frozen=True)
class PipelineResult:
    perception: PerceptionResult
    memories: list[MemoryCandidate]
    facts: list[FactCandidate]
    relationship_delta: RelationDelta
    decision: ParticipationDecision
    reply: str | None = None
    reply_self_memories: list[MemoryCandidate] = field(default_factory=list)
    image_descriptions: list[str] = field(default_factory=list)
    sticker_candidates: list[StickerCandidate] = field(default_factory=list)
    selected_sticker: StickerAssetRecord | None = None
