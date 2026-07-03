from __future__ import annotations

import os
from typing import Protocol

import httpx
from loguru import logger

from qq_llm_bot.config import LLMConfig, VisionConfig


class LLMClient(Protocol):
    async def complete_text(self, system_prompt: str, user_prompt: str) -> str | None:
        ...

    async def complete_vision(
        self,
        system_prompt: str,
        user_prompt: str,
        image_urls: list[str],
        vision_config: VisionConfig,
    ) -> str | None:
        ...


class DisabledLLMClient:
    async def complete_text(self, system_prompt: str, user_prompt: str) -> str | None:
        return None

    async def complete_vision(
        self,
        system_prompt: str,
        user_prompt: str,
        image_urls: list[str],
        vision_config: VisionConfig,
    ) -> str | None:
        return None


class OpenAICompatibleLLMClient:
    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        self.api_key = resolve_api_key(config)
        self.chat_completions_url = (
            normalize_chat_completions_url(config.base_url) if config.base_url else ""
        )

    async def complete_text(self, system_prompt: str, user_prompt: str) -> str | None:
        missing = self._missing_config_items()
        if missing:
            logger.warning("LLM is not configured; missing: {}", ", ".join(missing))
            return None

        payload = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }
        return await self._post_chat_completion(payload, self.config.timeout_seconds)

    async def complete_vision(
        self,
        system_prompt: str,
        user_prompt: str,
        image_urls: list[str],
        vision_config: VisionConfig,
    ) -> str | None:
        missing = self._missing_config_items()
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
            "model": vision_config.model or self.config.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content},
            ],
            "temperature": 0.2,
            "max_tokens": max(self.config.max_tokens, 512),
        }
        return await self._post_chat_completion(payload, vision_config.timeout_seconds)

    async def _post_chat_completion(self, payload: dict, timeout_seconds: float) -> str | None:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        try:
            async with httpx.AsyncClient(timeout=timeout_seconds) as client:
                response = await client.post(self.chat_completions_url, headers=headers, json=payload)
                response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            body = exc.response.text[:500]
            logger.warning("LLM HTTP error: status={} body={}", exc.response.status_code, body)
            return None
        except httpx.HTTPError as exc:
            logger.warning("LLM request failed: {}", exc)
            return None

        try:
            data = response.json()
            return extract_chat_completion_text(data)
        except (KeyError, TypeError, ValueError) as exc:
            logger.warning("LLM response parse failed: {}", exc)
            return None

    def _missing_config_items(self) -> list[str]:
        missing = []
        if not self.config.model:
            missing.append("llm.model")
        if not self.config.base_url:
            missing.append("llm.base_url")
        if not self.api_key:
            missing.append(f"{self.config.api_key_env}/llm.api_key")
        return missing


def build_llm_client(config: LLMConfig) -> LLMClient:
    provider = config.provider.lower().replace("_", "-")
    if provider == "disabled":
        return DisabledLLMClient()
    if provider in {"openai-compatible", "openai"}:
        return OpenAICompatibleLLMClient(config)
    raise ValueError(
        f"Unsupported llm.provider={config.provider!r}. "
        "Supported providers: disabled, openai-compatible."
    )


def resolve_api_key(config: LLMConfig) -> str:
    if config.api_key:
        return config.api_key
    return os.getenv(config.api_key_env, "").strip()


def is_llm_configured(config: LLMConfig) -> bool:
    provider = config.provider.lower().replace("_", "-")
    if provider == "disabled":
        return False
    return bool(config.base_url and config.model and resolve_api_key(config))


def normalize_chat_completions_url(base_url: str) -> str:
    base = base_url.strip().rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    if base.endswith("/v1"):
        return f"{base}/chat/completions"
    return f"{base}/v1/chat/completions"


def extract_chat_completion_text(data: dict) -> str | None:
    message = data["choices"][0]["message"]
    content = message.get("content")
    if isinstance(content, str):
        return content.strip() or None
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
        text = "".join(parts).strip()
        return text or None
    return None
