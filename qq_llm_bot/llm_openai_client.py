from __future__ import annotations

from loguru import logger

from qq_llm_bot.config import LLMConfig, VisionConfig
from qq_llm_bot.llm_config_helpers import resolve_api_key, resolve_model_config
from qq_llm_bot.llm_image_generation import LLMImageGenerationMixin
from qq_llm_bot.llm_models import LLMUsageRecorder
from qq_llm_bot.llm_response_helpers import normalize_chat_completions_url, normalize_responses_url
from qq_llm_bot.llm_routing import LLMRoutingMixin
from qq_llm_bot.llm_transport import LLMTransportMixin


class OpenAICompatibleLLMClient(LLMImageGenerationMixin, LLMRoutingMixin, LLMTransportMixin):
    def __init__(
        self,
        config: LLMConfig,
        usage_recorder: LLMUsageRecorder | None = None,
    ) -> None:
        self.config = config
        self.usage_recorder = usage_recorder
        self.api_key = resolve_api_key(config)
        self.chat_completions_url = (
            normalize_chat_completions_url(config.base_url) if config.base_url else ""
        )
        self.responses_url = normalize_responses_url(config.base_url) if config.base_url else ""
        self.last_chat_error = ""
        self._last_chat_failure_kind = ""
        self._last_chat_failure_status = 0
        self._last_chat_failure_body = ""
        self.last_image_generation_error = ""
        self._last_image_generation_failure_kind = ""

    async def complete_text(
        self,
        system_prompt: str,
        user_prompt: str,
        purpose: str = "",
        model_tier: str = "",
    ) -> str | None:
        model = self._text_model_for_purpose(purpose or "text", model_tier)
        missing = self._missing_config_items(model)
        if missing:
            logger.warning("LLM is not configured; missing: {}", ", ".join(missing))
            return None

        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.config.temperature,
            "max_tokens": self._max_tokens_for_text_purpose(purpose or "text"),
        }
        return await self._post_chat_completion(
            payload,
            self.config.timeout_seconds,
            purpose or "text",
            len(system_prompt) + len(user_prompt),
        )

    async def complete_vision(
        self,
        system_prompt: str,
        user_prompt: str,
        image_urls: list[str],
        vision_config: VisionConfig,
        purpose: str = "vision",
        model_tier: str = "",
    ) -> str | None:
        model = self._vision_model_for_tier(vision_config, model_tier)
        missing = self._missing_config_items(model)
        if missing:
            logger.warning("LLM vision is not configured; missing: {}", ", ".join(missing))
            return None
        clean_urls = [url.strip() for url in image_urls if url.strip()]
        if not clean_urls:
            return None

        content = [{"type": "text", "text": user_prompt}]
        content.extend(
            {
                "type": "image_url",
                "image_url": {
                    "url": url,
                    "detail": vision_config.detail,
                },
            }
            for url in clean_urls[: vision_config.max_images_per_message]
        )
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content},
            ],
            "temperature": 0.2,
            "max_tokens": max(self.config.max_tokens, 512),
        }
        text = await self._post_chat_completion(
            payload,
            vision_config.timeout_seconds,
            purpose or "vision",
            len(system_prompt) + len(user_prompt),
        )
        if (
            text is None
            and model_tier != "flagship"
            and self._should_retry_vision_failure_with_flagship(model, vision_config)
        ):
            payload["model"] = self._vision_model_for_tier(vision_config, "flagship")
            missing = self._missing_config_items(str(payload["model"]))
            if missing:
                logger.warning(
                    "LLM flagship vision retry is not configured; missing: {}",
                    ", ".join(missing),
                )
                return None
            logger.info(
                "Retrying LLM vision with detailed model after simple-model failure: "
                "purpose={} simple_model={} detailed_model={}",
                purpose or "vision",
                model,
                payload["model"],
            )
            return await self._post_chat_completion(
                payload,
                vision_config.timeout_seconds,
                purpose or "vision",
                len(system_prompt) + len(user_prompt),
            )
        return text

    async def complete_multimodal(
        self,
        system_prompt: str,
        user_prompt: str,
        image_urls: list[str],
        vision_config: VisionConfig,
        purpose: str = "response",
        model_tier: str = "",
    ) -> str | None:
        clean_urls = [url.strip() for url in image_urls if url.strip()]
        if not clean_urls:
            return await self.complete_text(system_prompt, user_prompt, purpose, model_tier)

        model = self._vision_model_for_tier(
            vision_config,
            model_tier or ("flagship" if (purpose or "").strip().lower() == "response" else ""),
        )
        missing = self._missing_config_items(model)
        if missing:
            logger.warning("LLM multimodal response is not configured; missing: {}", ", ".join(missing))
            return None

        content = [{"type": "text", "text": user_prompt}]
        content.extend(
            {
                "type": "image_url",
                "image_url": {
                    "url": url,
                    "detail": vision_config.detail,
                },
            }
            for url in clean_urls[: vision_config.max_images_per_message]
        )
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content},
            ],
            "temperature": self.config.temperature,
            "max_tokens": self._max_tokens_for_text_purpose(purpose or "response"),
        }
        return await self._post_chat_completion(
            payload,
            max(self.config.timeout_seconds, vision_config.timeout_seconds),
            purpose or "response",
            len(system_prompt) + len(user_prompt),
        )

    def _missing_config_items(self, model_ref: str = "") -> list[str]:
        missing = []
        clean_model = (model_ref or self.config.model).strip()
        if not clean_model:
            missing.append("llm.router.chat_generation_model")
            return missing
        try:
            resolved = resolve_model_config(
                self.config,
                clean_model,
                require_supported_transport=True,
            )
        except ValueError as exc:
            return [str(exc)]
        if not resolved.base_url:
            missing.append(f"provider.{resolved.provider_id}.url" if resolved.provider_id else "llm.base_url")
        if not resolved.api_key:
            if resolved.provider_id and resolved.api_key_env:
                missing.append(f"{resolved.api_key_env}/provider.{resolved.provider_id}.key_env")
            elif resolved.api_key_env:
                missing.append(f"{resolved.api_key_env}/llm.api_key")
            else:
                missing.append("llm.api_key")
        return missing
