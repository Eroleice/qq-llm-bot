from __future__ import annotations

from loguru import logger

from qq_llm_bot.config import ImageGenerationConfig
from qq_llm_bot.llm_models import GeneratedImage
from qq_llm_bot.llm_response_helpers import _image_generation_input


class LLMImageGenerationMixin:
    async def generate_image(
        self,
        prompt: str,
        image_config: ImageGenerationConfig,
        image_urls: list[str] | None = None,
    ) -> GeneratedImage | None:
        self.last_image_generation_error = ""
        self._last_image_generation_failure_kind = ""
        missing = self._missing_config_items()
        if missing:
            self.last_image_generation_error = "missing: " + ", ".join(missing)
            logger.warning(
                "LLM image generation is not configured; missing: {}",
                ", ".join(missing),
            )
            return None
        clean_prompt = prompt.strip()
        if not clean_prompt:
            self.last_image_generation_error = "empty prompt"
            return None

        tool: dict[str, object] = {"type": "image_generation"}
        if image_config.size:
            tool["size"] = image_config.size
        if image_config.quality:
            tool["quality"] = image_config.quality
        if image_config.output_format:
            tool["output_format"] = image_config.output_format
        if image_config.output_compression:
            tool["output_compression"] = image_config.output_compression
        image_model = image_config.model.strip()
        if not image_model:
            self.last_image_generation_error = "missing: image_generation.model"
            logger.warning("LLM image generation requires explicit image_generation.model")
            return None

        payload = {
            "model": image_model,
            "input": _image_generation_input(clean_prompt, image_urls or []),
            "tools": [tool],
            "tool_choice": {"type": "image_generation"},
        }
        for attempt in range(1, 3):
            generated = await self._post_image_generation_response(
                payload,
                image_config.timeout_seconds,
            )
            if generated is not None:
                return generated
            if attempt == 1 and self._last_image_generation_failure_kind == "no_image":
                logger.warning(
                    "Retrying image generation once after response without image result: {}",
                    self.last_image_generation_error,
                )
                continue
            break
        return None
