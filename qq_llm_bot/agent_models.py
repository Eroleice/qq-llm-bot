from __future__ import annotations

from dataclasses import dataclass, field

from qq_llm_bot.models import FactCandidate, MemoryCandidate


@dataclass(frozen=True)
class LexiconTermCandidate:
    term: str
    reason: str = ""
    search_query: str = ""
    confidence: float = 0.5


@dataclass(frozen=True)
class SelfNarrativePlan:
    needs_self_narrative: bool
    purpose: str = "answer_question"
    allowed_kinds: tuple[str, ...] = ("self_preference", "self_habit", "self_hobby")
    should_invent: bool = False
    reason: str = ""
    requires_background: bool = False
    fallback_caution: str = ""


@dataclass(frozen=True)
class SelfNarrativePreparation:
    memories: list[MemoryCandidate] = field(default_factory=list)
    requires_background: bool = False
    background_available: bool = True
    blocked: bool = False
    block_reason: str = ""
    fallback_caution: str = ""


@dataclass(frozen=True)
class BatchObservationResult:
    memories: list[MemoryCandidate] = field(default_factory=list)
    facts: list[FactCandidate] = field(default_factory=list)
    reflection: MemoryCandidate | None = None
