from __future__ import annotations

from qq_llm_bot.config import VisionConfig

CHAT_PREPROCESS_PURPOSES = {
    "addressing_gate",
    "context_understanding",
    "draw_intent",
    "draw_prompt",
    "followup_gate",
    "lexicon_detect",
    "lexicon_summarize",
    "llm_test",
    "participation_policy",
    "perception",
    "sticker_select",
}
CHAT_GENERATION_PURPOSES = {
    "answer_question",
    "banter",
    "empathy",
    "response",
    "self_claim_rewrite",
    "topic_join",
}
QA_PURPOSES = {
    "final_qa",
    "final_qa_repair",
}
FACT_EXTRACTION_PURPOSES = {
    "fact_extract",
    "memory_curator",
}
COGNITION_PURPOSES = {
    "batch_observation",
    "guesswho",
    "profile_aggregate",
    "reflection",
    "relationship",
    "self_narrative_check",
    "self_narrative_draft",
    "self_narrative_plan",
    "whoami",
}


class LLMRoutingMixin:
    def should_retry_with_flagship(self, purpose: str) -> bool:
        if not self._routing_enabled():
            return False
        normal_model = self._text_model_for_purpose(purpose, "")
        escalated_model = self._text_model_for_purpose(purpose, "flagship")
        return bool(normal_model and escalated_model and normal_model != escalated_model)

    def should_retry_vision_with_flagship(self, vision_config: VisionConfig) -> bool:
        if not self._routing_enabled():
            return False
        normal_model = self._vision_model_for_tier(vision_config, "")
        escalated_model = self._vision_model_for_tier(vision_config, "flagship")
        return bool(normal_model and escalated_model and normal_model != escalated_model)

    def _routing_enabled(self) -> bool:
        return bool(self.config.routing.enabled)

    def _default_text_model(self) -> str:
        routing = self.config.routing
        return (
            routing.chat_generation_model
            or routing.chat_preprocess_model
            or routing.qa_model
            or routing.cognition_model
            or routing.fact_extraction_model
            or self.config.model
        )

    def _text_model_for_purpose(self, purpose: str, model_tier: str = "") -> str:
        fallback_model = self._default_text_model()
        if not self._routing_enabled():
            return fallback_model

        normalized = (purpose or "text").strip().lower()
        if model_tier == "flagship":
            return self._escalated_text_model_for_purpose(normalized) or fallback_model
        if normalized in CHAT_PREPROCESS_PURPOSES:
            return self.config.routing.chat_preprocess_model or fallback_model
        if normalized in CHAT_GENERATION_PURPOSES:
            return self.config.routing.chat_generation_model or fallback_model
        if normalized in QA_PURPOSES:
            return self.config.routing.qa_model or fallback_model
        if normalized in FACT_EXTRACTION_PURPOSES:
            return self.config.routing.fact_extraction_model or fallback_model
        if normalized in COGNITION_PURPOSES:
            return self.config.routing.cognition_model or fallback_model
        return fallback_model

    def _escalated_text_model_for_purpose(self, normalized_purpose: str) -> str:
        routing = self.config.routing
        fallback_model = self._default_text_model()
        if normalized_purpose in QA_PURPOSES:
            return routing.qa_model or fallback_model
        if normalized_purpose in FACT_EXTRACTION_PURPOSES:
            return routing.cognition_model or routing.qa_model or routing.chat_generation_model or fallback_model
        if normalized_purpose in COGNITION_PURPOSES:
            return routing.chat_generation_model or routing.qa_model or routing.cognition_model or fallback_model
        if normalized_purpose in CHAT_PREPROCESS_PURPOSES:
            return routing.chat_generation_model or routing.qa_model or fallback_model
        return routing.chat_generation_model or fallback_model

    def _max_tokens_for_text_purpose(self, purpose: str) -> int:
        normalized = (purpose or "text").strip().lower()
        if normalized == "draw_prompt":
            return max(self.config.max_tokens, 1024)
        if normalized == "draw_intent":
            return max(self.config.max_tokens, 512)
        return self.config.max_tokens

    def _vision_model_for_tier(
        self,
        vision_config: VisionConfig,
        model_tier: str = "",
    ) -> str:
        fallback_model = self._default_text_model()
        detailed_model = (
            self.config.routing.detailed_vision_model
            or self.config.routing.simple_vision_model
            or fallback_model
        )
        if not self._routing_enabled():
            return detailed_model
        if model_tier == "flagship":
            return detailed_model
        return self.config.routing.simple_vision_model or detailed_model

    def _should_retry_vision_failure_with_flagship(
        self,
        attempted_model: str,
        vision_config: VisionConfig,
    ) -> bool:
        if not self.should_retry_vision_with_flagship(vision_config):
            return False
        if attempted_model == self._vision_model_for_tier(vision_config, "flagship"):
            return False
        if self._last_chat_failure_kind != "http_error":
            return False
        return self._last_chat_failure_status in {400, 404, 415, 422}
