from __future__ import annotations

from typing import Any

from loguru import logger

from qq_llm_bot.config import VisionConfig
from qq_llm_bot.llm import LLMClient


STRUCTURED_JSON_REQUIRED_KEYS = {
    "batch_observation": {"memories", "facts", "reflection"},
    "context_understanding": {
        "current_intent",
        "relevant_messages",
        "resolved_references",
        "member_context",
        "uncertain_references",
        "ignored_noise",
    },
    "draw_prompt": {"prompt"},
    "fact_extract": {"facts"},
    "final_qa": {"verdict", "reason", "categories", "confidence"},
    "followup_gate": {"action", "confidence", "value_type", "reason"},
    "lexicon_detect": {"terms"},
    "lexicon_summarize": {"should_remember", "definition", "confidence"},
    "memory_curator": {"memories"},
    "participation_policy": {"action", "score", "value_type", "value_score", "reason"},
    "perception": {"is_question", "is_self_disclosure", "topics", "emotion_hint", "confidence"},
    "profile_aggregate": {"summary", "traits", "supporting_fact_ids"},
    "reflection": {"summary", "topics", "importance"},
    "relationship": {"closeness", "trust", "familiarity", "tension", "summary_patch", "reason"},
    "self_narrative_check": {"status", "reason", "safe_rewrite"},
    "self_narrative_draft": {"kind", "content", "fictionality", "confidence", "importance"},
    "self_narrative_plan": {
        "needs_self_narrative",
        "purpose",
        "allowed_kinds",
        "should_invent",
        "requires_background",
        "fallback_caution",
        "reason",
    },
    "sticker_select": {"asset_id", "confidence", "reason"},
}


def can_retry_text_with_flagship(
    llm: LLMClient,
    purpose: str,
    model_tier: str,
    allow_flagship_retry: bool,
) -> bool:
    if not allow_flagship_retry or model_tier == "flagship":
        return False
    retry_checker = getattr(llm, "should_retry_with_flagship", None)
    if not callable(retry_checker):
        return False
    try:
        return bool(retry_checker(purpose))
    except Exception as exc:  # pragma: no cover - retry checks must not break fallback
        logger.warning("LLM flagship retry check failed: {}", exc)
        return False


def can_retry_vision_with_flagship(
    llm: LLMClient,
    vision_config: VisionConfig,
    model_tier: str,
    allow_flagship_retry: bool,
) -> bool:
    if not allow_flagship_retry or model_tier == "flagship":
        return False
    retry_checker = getattr(llm, "should_retry_vision_with_flagship", None)
    if not callable(retry_checker):
        return False
    try:
        return bool(retry_checker(vision_config))
    except Exception as exc:  # pragma: no cover - retry checks must not break fallback
        logger.warning("LLM vision flagship retry check failed: {}", exc)
        return False


def structured_json_flagship_retry_reason(purpose: str, data: dict[str, Any]) -> str:
    normalized = (purpose or "").strip().lower()
    required = STRUCTURED_JSON_REQUIRED_KEYS.get(normalized, set())
    missing = sorted(key for key in required if key not in data)
    if missing:
        return "missing_keys:" + ",".join(missing[:4])
    if normalized == "final_qa":
        confidence = _clamp_float(data.get("confidence", 0.0))
        if 0.0 < confidence < 0.72:
            return f"low_final_qa_confidence:{confidence:.2f}"
    if normalized == "participation_policy":
        action = str(data.get("action", "")).strip().lower()
        score = _clamp_float(data.get("score", 0.0))
        value_score = _clamp_float(data.get("value_score", 0.0))
        if action == "proactive_reply":
            return f"proactive_review:{score:.2f}/{value_score:.2f}"
    if normalized in {"followup_gate", "lexicon_summarize", "sticker_select"}:
        confidence = _clamp_float(data.get("confidence", 0.0))
        if 0.0 < confidence < 0.68:
            return f"low_confidence:{confidence:.2f}"
    return ""


def vision_json_flagship_retry_reason(
    data: dict[str, Any],
    *,
    expected_images: int,
    direct_image_hint: bool,
) -> str:
    images = data.get("images")
    if not isinstance(images, list):
        return "missing_images"
    if expected_images > 1 and len(images) < expected_images:
        return f"incomplete_multi_image:{len(images)}/{expected_images}"
    for item in images:
        if not isinstance(item, dict):
            return "invalid_image_item"
        description = str(item.get("description", "")).strip()
        ocr_text = str(item.get("ocr_text", "")).strip()
        confidence = _clamp_float(item.get("confidence", 0.0))
        if direct_image_hint and not description and not ocr_text:
            return "direct_image_empty_description"
        if 0.0 < confidence < 0.68:
            return f"low_vision_confidence:{confidence:.2f}"
    return ""


def _clamp_float(value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = 0.0
    return max(0.0, min(1.0, parsed))
