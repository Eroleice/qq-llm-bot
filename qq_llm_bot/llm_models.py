from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

from qq_llm_bot.config import ImageGenerationConfig, VisionConfig


@dataclass(frozen=True)
class GeneratedImage:
    data: bytes | None = None
    url: str = ""
    mime_type: str = "image/png"


@dataclass(frozen=True)
class LLMUsageRecord:
    purpose: str
    model: str
    prompt_chars: int
    completion_chars: int
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    created_at: int = 0


LLMUsageRecorder = Callable[[LLMUsageRecord], None]


class LLMClient(Protocol):
    last_image_generation_error: str

    async def complete_text(
        self,
        system_prompt: str,
        user_prompt: str,
        purpose: str = "",
        model_tier: str = "",
    ) -> str | None:
        ...

    async def complete_vision(
        self,
        system_prompt: str,
        user_prompt: str,
        image_urls: list[str],
        vision_config: VisionConfig,
        purpose: str = "vision",
        model_tier: str = "",
    ) -> str | None:
        ...

    async def complete_multimodal(
        self,
        system_prompt: str,
        user_prompt: str,
        image_urls: list[str],
        vision_config: VisionConfig,
        purpose: str = "response",
        model_tier: str = "",
    ) -> str | None:
        ...

    async def generate_image(
        self,
        prompt: str,
        image_config: ImageGenerationConfig,
        image_urls: list[str] | None = None,
    ) -> GeneratedImage | None:
        ...


class DisabledLLMClient:
    last_image_generation_error = "llm.provider=disabled"

    async def complete_text(
        self,
        system_prompt: str,
        user_prompt: str,
        purpose: str = "",
        model_tier: str = "",
    ) -> str | None:
        return None

    async def complete_vision(
        self,
        system_prompt: str,
        user_prompt: str,
        image_urls: list[str],
        vision_config: VisionConfig,
        purpose: str = "vision",
        model_tier: str = "",
    ) -> str | None:
        return None

    async def complete_multimodal(
        self,
        system_prompt: str,
        user_prompt: str,
        image_urls: list[str],
        vision_config: VisionConfig,
        purpose: str = "response",
        model_tier: str = "",
    ) -> str | None:
        return None

    async def generate_image(
        self,
        prompt: str,
        image_config: ImageGenerationConfig,
        image_urls: list[str] | None = None,
    ) -> GeneratedImage | None:
        return None
