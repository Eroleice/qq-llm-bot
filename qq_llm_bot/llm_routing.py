from __future__ import annotations

from qq_llm_bot.config import VisionConfig

BASE_TEXT_PURPOSES = {
    "batch_observation",
    "draw_intent",
    "fact_extract",
    "final_qa_repair",
    "followup_gate",
    "lexicon_detect",
    "lexicon_summarize",
    "perception",
    "profile_aggregate",
    "reflection",
    "relationship",
    "sticker_select",
    "vision",
}
BASE_TEXT_PURPOSES_WITH_ESCALATION = {
    "draw_prompt",
    "final_qa",
    "participation_policy",
}
FLAGSHIP_TEXT_PURPOSES = {
    "response",
    "self_claim_rewrite",
    "self_narrative_check",
    "self_narrative_draft",
}


class LLMRoutingMixin:
    def should_retry_with_flagship(self, purpose: str) -> bool:
        if not self._routing_enabled():
            return False
        base_model = self._text_model_for_purpose(purpose, "")
        flagship_model = self._text_model_for_purpose(purpose, "flagship")
        return bool(base_model and flagship_model and base_model != flagship_model)

    def should_retry_vision_with_flagship(self, vision_config: VisionConfig) -> bool:
        if not self._routing_enabled():
            return False
        base_model = self._vision_model_for_tier(vision_config, "")
        flagship_model = self._vision_model_for_tier(vision_config, "flagship")
        return bool(base_model and flagship_model and base_model != flagship_model)

    def _routing_enabled(self) -> bool:
        return bool(self.config.routing.enabled)

    def _text_model_for_purpose(self, purpose: str, model_tier: str = "") -> str:
        if not self._routing_enabled():
            return self.config.model
        if model_tier == "flagship":
            return self.config.routing.flagship_model or self.config.model
        if model_tier == "base":
            return self.config.routing.base_model or self.config.model

        normalized = (purpose or "text").strip().lower()
        if normalized in FLAGSHIP_TEXT_PURPOSES:
            return self.config.routing.flagship_model or self.config.model
        if normalized in BASE_TEXT_PURPOSES or normalized in BASE_TEXT_PURPOSES_WITH_ESCALATION:
            return self.config.routing.base_model or self.config.model
        return self.config.model

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
        if not self._routing_enabled():
            return vision_config.model or self.config.model
        if model_tier == "flagship":
            return vision_config.model or self.config.routing.flagship_model or self.config.model
        return (
            self.config.routing.vision_base_model
            or self.config.routing.base_model
            or vision_config.model
            or self.config.model
        )

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
