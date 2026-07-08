from __future__ import annotations

from dataclasses import dataclass, field

from qq_llm_bot.config import ParticipationMode
from qq_llm_bot.knowledge_models import (
    FactCandidate,
    FactRecord,
    MemoryCandidate,
    MemoryRecord,
    PerceptionResult,
    SemanticContext,
    TargetUserContext,
    UserProfileRecord,
)
from qq_llm_bot.message_models import StickerAssetRecord, StickerCandidate
from qq_llm_bot.model_types import DecisionAction, ParticipationValueType


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
    semantic_context: SemanticContext | None = None


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
    final_qa_blocked_reply: str | None = None
    final_qa_reason: str = ""
    final_qa_categories: tuple[str, ...] = ()
    final_qa_confidence: float = 0.0
