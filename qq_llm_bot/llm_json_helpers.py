from __future__ import annotations

from typing import Any

from loguru import logger

from qq_llm_bot.config import AppConfig
from qq_llm_bot.json_utils import parse_json_object
from qq_llm_bot.llm import LLMClient
from qq_llm_bot.llm_json_policy import (
    STRUCTURED_JSON_REQUIRED_KEYS as STRUCTURED_JSON_REQUIRED_KEYS,
    can_retry_text_with_flagship,
    can_retry_vision_with_flagship,
    structured_json_flagship_retry_reason,
    vision_json_flagship_retry_reason,
)
from qq_llm_bot.models import MessageContext


async def complete_json(
    llm: LLMClient,
    system_prompt: str,
    user_prompt: str,
    purpose: str = "structured_json",
    model_tier: str = "",
    allow_flagship_retry: bool = True,
) -> dict[str, Any] | None:
    text = await llm.complete_text(
        system_prompt,
        user_prompt,
        purpose=purpose,
        model_tier=model_tier,
    )
    if not text:
        if can_retry_text_with_flagship(llm, purpose, model_tier, allow_flagship_retry):
            logger.info("Retrying structured LLM JSON with flagship model: purpose={} reason=empty", purpose)
            return await complete_json(
                llm,
                system_prompt,
                user_prompt,
                purpose=purpose,
                model_tier="flagship",
                allow_flagship_retry=False,
            )
        return None
    try:
        data = parse_json_object(text)
    except ValueError as exc:
        logger.warning("Structured LLM JSON parse failed: {}", exc)
        if can_retry_text_with_flagship(llm, purpose, model_tier, allow_flagship_retry):
            logger.info(
                "Retrying structured LLM JSON with flagship model: purpose={} reason=parse_error",
                purpose,
            )
            return await complete_json(
                llm,
                system_prompt,
                user_prompt,
                purpose=purpose,
                model_tier="flagship",
                allow_flagship_retry=False,
            )
        return None
    retry_reason = structured_json_flagship_retry_reason(purpose, data)
    if retry_reason and can_retry_text_with_flagship(llm, purpose, model_tier, allow_flagship_retry):
        logger.info(
            "Retrying structured LLM JSON with flagship model: purpose={} reason={}",
            purpose,
            retry_reason,
        )
        retry_data = await complete_json(
            llm,
            system_prompt,
            user_prompt,
            purpose=purpose,
            model_tier="flagship",
            allow_flagship_retry=False,
        )
        return retry_data or data
    return data


async def complete_vision_json(
    llm: LLMClient,
    config: AppConfig,
    system_prompt: str,
    user_prompt: str,
    image_urls: list[str],
    purpose: str = "vision",
    model_tier: str = "",
    direct_image_hint: bool = False,
    allow_flagship_retry: bool = True,
) -> dict[str, Any] | None:
    text = await llm.complete_vision(
        system_prompt,
        user_prompt,
        image_urls,
        config.vision,
        purpose=purpose,
        model_tier=model_tier,
    )
    if not text:
        if can_retry_vision_with_flagship(
            llm,
            config.vision,
            model_tier,
            allow_flagship_retry,
        ):
            logger.info("Retrying LLM vision JSON with flagship model: reason=empty")
            return await complete_vision_json(
                llm,
                config,
                system_prompt,
                user_prompt,
                image_urls,
                purpose=purpose,
                model_tier="flagship",
                direct_image_hint=direct_image_hint,
                allow_flagship_retry=False,
            )
        return None
    try:
        data = parse_json_object(text)
    except ValueError as exc:
        logger.warning("Structured vision JSON parse failed: {}", exc)
        if can_retry_vision_with_flagship(
            llm,
            config.vision,
            model_tier,
            allow_flagship_retry,
        ):
            logger.info("Retrying LLM vision JSON with flagship model: reason=parse_error")
            return await complete_vision_json(
                llm,
                config,
                system_prompt,
                user_prompt,
                image_urls,
                purpose=purpose,
                model_tier="flagship",
                direct_image_hint=direct_image_hint,
                allow_flagship_retry=False,
            )
        return None
    retry_reason = vision_json_flagship_retry_reason(
        data,
        expected_images=len(image_urls),
        direct_image_hint=direct_image_hint,
    )
    if retry_reason and can_retry_vision_with_flagship(
        llm,
        config.vision,
        model_tier,
        allow_flagship_retry,
    ):
        logger.info("Retrying LLM vision JSON with flagship model: reason={}", retry_reason)
        retry_data = await complete_vision_json(
            llm,
            config,
            system_prompt,
            user_prompt,
            image_urls,
            purpose=purpose,
            model_tier="flagship",
            direct_image_hint=direct_image_hint,
            allow_flagship_retry=False,
        )
        return retry_data or data
    return data


def vision_direct_image_hint(context: MessageContext, image_urls: list[str]) -> bool:
    if not image_urls:
        return False
    text = context.plain_text.strip()
    if context.is_direct and text:
        return True
    lowered = text.lower()
    image_cues = (
        "ocr",
        "image",
        "图",
        "图片",
        "截图",
        "照片",
        "这张",
        "这些图",
        "看一下",
        "帮我看",
        "识别",
        "读图",
        "文字",
        "什么意思",
        "是什么",
    )
    return any(cue in lowered for cue in image_cues)
