from __future__ import annotations

from typing import Any

from qq_llm_bot.config_models import ImageGenerationConfig, StickerConfig, VisionConfig
from qq_llm_bot.config_values import (
    bool_value as _bool_value,
    float_in_range as _float_in_range,
    image_generation_size as _image_generation_size,
    int_in_range as _int_in_range,
    positive_int as _positive_int,
)
from qq_llm_bot.text_utils import safe_choice as _safe_choice


def vision_config(raw: dict[str, Any]) -> VisionConfig:
    return VisionConfig(
        enabled=_bool_value(raw.get("enabled", False)),
        max_images_per_message=_positive_int(
            raw.get("max_images_per_message", 3),
            "vision.max_images_per_message",
        ),
        detail=_safe_choice(
            str(raw.get("detail", "low")).strip().lower(),
            {"low", "high", "auto"},
            "low",
        ),
        timeout_seconds=_float_in_range(
            raw.get("timeout_seconds", 45.0),
            "vision.timeout_seconds",
            1,
            300,
        ),
        remember_threshold=_float_in_range(
            raw.get("remember_threshold", 0.78),
            "vision.remember_threshold",
            0,
            1,
        ),
    )


def image_generation_config(raw: dict[str, Any]) -> ImageGenerationConfig:
    return ImageGenerationConfig(
        enabled=_bool_value(raw.get("enabled", False)),
        storage_dir=str(raw.get("storage_dir", "data/generated_images")).strip()
        or "data/generated_images",
        size=_image_generation_size(
            raw.get("size", "512x512"),
            max_dimension=832,
        ),
        quality=_safe_choice(
            str(raw.get("quality", "low")).strip().lower(),
            {"low", "medium", "high", "auto"},
            "low",
        ),
        output_format=_safe_choice(
            str(raw.get("output_format", "jpeg")).strip().lower(),
            {"jpeg", "png", "webp"},
            "jpeg",
        ),
        output_compression=_int_in_range(
            raw.get("output_compression", 65),
            "image_generation.output_compression",
            1,
            100,
        ),
        timeout_seconds=_float_in_range(
            raw.get("timeout_seconds", 240.0),
            "image_generation.timeout_seconds",
            1,
            900,
        ),
        max_send_dimension=_int_in_range(
            raw.get("max_send_dimension", 832),
            "image_generation.max_send_dimension",
            256,
            2048,
        ),
        min_trust=_int_in_range(
            raw.get("min_trust", 5),
            "image_generation.min_trust",
            0,
            100,
        ),
        daily_limit=_positive_int(
            raw.get("daily_limit", 5),
            "image_generation.daily_limit",
        ),
        max_prompt_chars=_positive_int(
            raw.get("max_prompt_chars", 800),
            "image_generation.max_prompt_chars",
        ),
        max_reference_images=_int_in_range(
            raw.get("max_reference_images", 3),
            "image_generation.max_reference_images",
            0,
            3,
        ),
        reference_image_max_bytes=_int_in_range(
            raw.get("reference_image_max_bytes", 4 * 1024 * 1024),
            "image_generation.reference_image_max_bytes",
            64 * 1024,
            20 * 1024 * 1024,
        ),
        reference_image_max_dimension=_int_in_range(
            raw.get("reference_image_max_dimension", 1536),
            "image_generation.reference_image_max_dimension",
            256,
            4096,
        ),
        reference_image_quality=_int_in_range(
            raw.get("reference_image_quality", 85),
            "image_generation.reference_image_quality",
            30,
            95,
        ),
    )


def sticker_config(raw: dict[str, Any]) -> StickerConfig:
    return StickerConfig(
        enabled=_bool_value(raw.get("enabled", False)),
        storage_dir=str(raw.get("storage_dir", "data/stickers")).strip() or "data/stickers",
        min_confidence=_float_in_range(
            raw.get("min_confidence", 0.72),
            "stickers.min_confidence",
            0,
            1,
        ),
        selection_threshold=_float_in_range(
            raw.get("selection_threshold", 0.68),
            "stickers.selection_threshold",
            0,
            1,
        ),
        max_context_stickers=_positive_int(
            raw.get("max_context_stickers", 24),
            "stickers.max_context_stickers",
        ),
        download_timeout_seconds=_float_in_range(
            raw.get("download_timeout_seconds", 20.0),
            "stickers.download_timeout_seconds",
            1,
            300,
        ),
        max_download_bytes=_positive_int(
            raw.get("max_download_bytes", 8 * 1024 * 1024),
            "stickers.max_download_bytes",
        ),
        send_cooldown_seconds=_positive_int(
            raw.get("send_cooldown_seconds", 120),
            "stickers.send_cooldown_seconds",
        ),
        unused_ttl_hours=_positive_int(
            raw.get("unused_ttl_hours", 72),
            "stickers.unused_ttl_hours",
        ),
        cleanup_interval_hours=_positive_int(
            raw.get("cleanup_interval_hours", 24),
            "stickers.cleanup_interval_hours",
        ),
    )
