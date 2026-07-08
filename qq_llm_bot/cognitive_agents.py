from __future__ import annotations

from dataclasses import replace

from qq_llm_bot.agent_core import (
    ContextUnderstandingAgent,
    ParticipationPolicyAgent,
    RelationshipAgent,
    ResponseAgent,
)
from qq_llm_bot.agent_extractors import (
    FactExtractorAgent,
    LexiconAgent,
    MemoryCuratorAgent as MemoryCuratorAgent,
    PerceptionAgent,
)
from qq_llm_bot.agent_formatters import (
    semantic_context_has_content as _semantic_context_has_content,
)
from qq_llm_bot.agent_final_qa import FinalQAAgent
from qq_llm_bot.agent_models import (
    BatchObservationResult,
)
from qq_llm_bot.agent_maintenance import (
    BatchObservationAgent,
    ProfileAggregatorAgent,
    ReflectionAgent,
)
from qq_llm_bot.agent_policy_helpers import (
    FOLLOWUP_CUE_PATTERN as FOLLOWUP_CUE_PATTERN,
    _alias_from_fact_text as _alias_from_fact_text,
    _best_target_alias as _best_target_alias,
    _clean_string_items as _clean_string_items,
    _clamp_delta as _clamp_delta,
    _fallback_semantic_context as _fallback_semantic_context,
    _first_identity_fact_text as _first_identity_fact_text,
    _has_unresolved_identity_target as _has_unresolved_identity_target,
    _looks_like_identity_query as _looks_like_identity_query,
    _looks_like_low_value_proactive_reply as _looks_like_low_value_proactive_reply,
    _looks_like_recent_interaction_followup as _looks_like_recent_interaction_followup,
    _looks_like_uncertain_reply as _looks_like_uncertain_reply,
    _needs_context_understanding as _needs_context_understanding,
    _proactive_value_type_allowed as _proactive_value_type_allowed,
    _reply_has_incremental_value as _reply_has_incremental_value,
    _safe_participation_value_type as _safe_participation_value_type,
    _semantic_context_from_json as _semantic_context_from_json,
    _target_confirmation_reply as _target_confirmation_reply,
    _target_fact_fallback_reply as _target_fact_fallback_reply,
)
from qq_llm_bot.agent_self_narrative import (
    SelfNarrativeAgent,
)
from qq_llm_bot.agent_stickers import StickerSelectorAgent
from qq_llm_bot.agent_vision import VisionAgent
from qq_llm_bot.config import AppConfig, ParticipationMode
from qq_llm_bot.final_qa import (
    FinalQAResult,
    sanitize_reply as _sanitize_reply,  # noqa: F401 - compatibility re-export
)
from qq_llm_bot.llm import LLMClient
from qq_llm_bot.llm_json_helpers import (
    complete_json as _complete_json,  # noqa: F401 - compatibility re-export
    complete_vision_json as _complete_vision_json,  # noqa: F401 - compatibility re-export
)
from qq_llm_bot.models import (
    ConversationSnapshot,
    FactRecord,
    MemoryCandidate,
    MemoryRecord,
    MessageContext,
    ParticipationDecision,
    PipelineResult,
    ReplyDraft,
    UserProfileDraft,
    UserProfileRecord,
)
from qq_llm_bot.vision_analysis import (
    VisionAnalysis,
    VisionCacheStore,
    context_with_vision as _context_with_vision,
    recordable_image_descriptions as _recordable_image_descriptions,
    unresolved_context_image_urls as _unresolved_context_image_urls,
)
from qq_llm_bot.web_search import WebSearchClient


class AgentPipeline:
    def __init__(
        self,
        config: AppConfig,
        llm: LLMClient,
        web_search: WebSearchClient | None = None,
        vision_cache: VisionCacheStore | None = None,
    ) -> None:
        self.config = config
        self.perception = PerceptionAgent(llm)
        self.vision = VisionAgent(config, llm, vision_cache)
        self.fact_extractor = FactExtractorAgent(llm)
        self.lexicon = LexiconAgent(config, llm, web_search)
        self.relationship = RelationshipAgent(llm)
        self.policy = ParticipationPolicyAgent(config, llm)
        self.self_narrative = SelfNarrativeAgent(llm)
        self.context_understanding = ContextUnderstandingAgent(config, llm)
        self.response = ResponseAgent(config, llm)
        self.final_qa = FinalQAAgent(config, llm)
        self.stickers = StickerSelectorAgent(config, llm)
        self.reflection = ReflectionAgent(llm)
        self.profile_aggregator = ProfileAggregatorAgent(llm)
        self.batch_observation = BatchObservationAgent(config, llm)

    async def run(
        self,
        context: MessageContext,
        mode: ParticipationMode,
        snapshot: ConversationSnapshot,
        *,
        analyze_images: bool = True,
    ) -> PipelineResult:
        vision = await self.vision.analyze(context, allow_remote=analyze_images)
        enriched_context = _context_with_vision(context, vision)
        perception = await self.perception.analyze(enriched_context, snapshot)
        facts = await self.fact_extractor.extract(enriched_context, perception, snapshot)
        lexicon_memories = await self.lexicon.learn(enriched_context, snapshot)
        relationship_delta = await self.relationship.calculate_delta(
            enriched_context,
            perception,
            snapshot,
        )
        decision = await self.policy.decide(enriched_context, perception, mode, snapshot)
        self_preparation = await self.self_narrative.prepare(
            enriched_context,
            perception,
            decision,
            snapshot,
        )
        final_decision = decision
        if self_preparation.blocked:
            final_decision = replace(
                decision,
                action="observe",
                reason=(
                    f"{decision.reason}; self background gate blocked: "
                    f"{self_preparation.block_reason}"
                ),
                score=min(decision.score, 0.49),
            )
        semantic_context = await self.context_understanding.analyze(
            enriched_context,
            perception,
            final_decision,
            snapshot,
        )
        response_snapshot = (
            replace(snapshot, semantic_context=semantic_context)
            if _semantic_context_has_content(semantic_context)
            else snapshot
        )
        reply_draft = await self.response.generate(
            enriched_context,
            perception,
            final_decision,
            response_snapshot,
            self_preparation.memories,
            self_preparation.fallback_caution
            if self_preparation.requires_background and not self_preparation.background_available
            else "",
            image_urls=_unresolved_context_image_urls(
                context,
                vision,
                self.config.vision.max_images_per_message,
            ),
        )
        final_qa_blocked_reply: str | None = None
        final_qa_reason = ""
        final_qa_categories: tuple[str, ...] = ()
        final_qa_confidence = 0.0
        if reply_draft.text:
            qa_result = await self.final_qa.review(
                enriched_context,
                final_decision,
                response_snapshot,
                reply_draft.text,
            )
            if not qa_result.allowed:
                blocked_reply = reply_draft.text
                repaired_reply = await self.final_qa.repair(
                    enriched_context,
                    final_decision,
                    response_snapshot,
                    blocked_reply,
                    qa_result,
                )
                if repaired_reply:
                    repair_qa_result = await self.final_qa.review(
                        enriched_context,
                        final_decision,
                        response_snapshot,
                        repaired_reply,
                    )
                    if repair_qa_result.allowed:
                        reply_draft = replace(reply_draft, text=repaired_reply)
                    else:
                        final_qa_blocked_reply = repaired_reply
                        final_qa_reason = repair_qa_result.reason
                        final_qa_categories = repair_qa_result.categories
                        final_qa_confidence = repair_qa_result.confidence
                        final_decision = replace(
                            final_decision,
                            action="observe",
                            reason=(
                                f"{final_decision.reason}; final QA blocked reply: "
                                f"{qa_result.reason}; repair blocked: {repair_qa_result.reason}"
                            ),
                            score=min(final_decision.score, 0.49),
                        )
                        reply_draft = ReplyDraft()
                else:
                    final_qa_blocked_reply = blocked_reply
                    final_qa_reason = qa_result.reason
                    final_qa_categories = qa_result.categories
                    final_qa_confidence = qa_result.confidence
                    final_decision = replace(
                        final_decision,
                        action="observe",
                        reason=f"{final_decision.reason}; final QA blocked reply: {qa_result.reason}",
                        score=min(final_decision.score, 0.49),
                    )
                    reply_draft = ReplyDraft()
        if decision.action == "proactive_reply" and not reply_draft.text:
            if final_decision.action != "observe":
                final_decision = replace(
                    decision,
                    action="observe",
                    reason=f"{decision.reason}; proactive reply suppressed by value guard",
                    score=min(decision.score, 0.49),
                )
        selected_sticker = await self.stickers.select(
            enriched_context,
            final_decision,
            response_snapshot,
            reply_draft.text,
        )
        return PipelineResult(
            perception=perception,
            memories=[*lexicon_memories, *vision.memory_candidates],
            facts=[*facts, *vision.fact_candidates],
            relationship_delta=relationship_delta,
            decision=final_decision,
            reply=reply_draft.text,
            reply_self_memories=reply_draft.self_memory_candidates,
            image_descriptions=_recordable_image_descriptions(context, vision),
            sticker_candidates=list(vision.sticker_candidates),
            selected_sticker=selected_sticker,
            final_qa_blocked_reply=final_qa_blocked_reply,
            final_qa_reason=final_qa_reason,
            final_qa_categories=final_qa_categories,
            final_qa_confidence=final_qa_confidence,
        )

    async def reflect(
        self,
        group_id: str,
        recent_messages: list[str],
        prior_reflections: list[MemoryRecord],
    ) -> MemoryCandidate | None:
        return await self.reflection.reflect(group_id, recent_messages, prior_reflections)

    async def profile(
        self,
        user_id: str,
        facts: list[FactRecord],
        current_profile: UserProfileRecord | None = None,
    ) -> UserProfileDraft | None:
        return await self.profile_aggregator.aggregate(user_id, facts, current_profile)

    async def review_reply(
        self,
        context: MessageContext,
        decision: ParticipationDecision,
        snapshot: ConversationSnapshot,
        reply: str | None,
    ) -> FinalQAResult:
        return await self.final_qa.review(context, decision, snapshot, reply)

    async def observe_vision(self, context: MessageContext) -> VisionAnalysis:
        return await self.vision.analyze(context)

    async def observe_batch(
        self,
        group_id: str,
        contexts: list[MessageContext],
        prior_reflections: list[MemoryRecord],
        group_lexicon: list[MemoryRecord],
    ) -> BatchObservationResult:
        return await self.batch_observation.summarize(
            group_id,
            contexts,
            prior_reflections,
            group_lexicon,
        )
