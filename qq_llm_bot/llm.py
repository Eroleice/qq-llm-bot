from __future__ import annotations

import base64
import json
import os
import time
from dataclasses import dataclass
from typing import Callable, Protocol

import httpx
from loguru import logger

from qq_llm_bot.config import ImageGenerationConfig, LLMConfig, VisionConfig


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
    async def complete_text(
        self,
        system_prompt: str,
        user_prompt: str,
        purpose: str = "",
    ) -> str | None:
        ...

    async def complete_vision(
        self,
        system_prompt: str,
        user_prompt: str,
        image_urls: list[str],
        vision_config: VisionConfig,
        purpose: str = "vision",
    ) -> str | None:
        ...

    async def generate_image(
        self,
        prompt: str,
        image_config: ImageGenerationConfig,
    ) -> GeneratedImage | None:
        ...


class DisabledLLMClient:
    async def complete_text(
        self,
        system_prompt: str,
        user_prompt: str,
        purpose: str = "",
    ) -> str | None:
        return None

    async def complete_vision(
        self,
        system_prompt: str,
        user_prompt: str,
        image_urls: list[str],
        vision_config: VisionConfig,
        purpose: str = "vision",
    ) -> str | None:
        return None

    async def generate_image(
        self,
        prompt: str,
        image_config: ImageGenerationConfig,
    ) -> GeneratedImage | None:
        return None


class OpenAICompatibleLLMClient:
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

    async def complete_text(
        self,
        system_prompt: str,
        user_prompt: str,
        purpose: str = "",
    ) -> str | None:
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
        return await self._post_chat_completion(
            payload,
            vision_config.timeout_seconds,
            purpose or "vision",
            len(system_prompt) + len(user_prompt),
        )

    async def generate_image(
        self,
        prompt: str,
        image_config: ImageGenerationConfig,
    ) -> GeneratedImage | None:
        missing = self._missing_config_items()
        if missing:
            logger.warning(
                "LLM image generation is not configured; missing: {}",
                ", ".join(missing),
            )
            return None
        clean_prompt = prompt.strip()
        if not clean_prompt:
            return None

        tool: dict[str, str] = {"type": "image_generation"}
        if image_config.size:
            tool["size"] = image_config.size
        if image_config.quality:
            tool["quality"] = image_config.quality
        payload = {
            "model": image_config.model or self.config.model,
            "input": clean_prompt,
            "tools": [tool],
            "tool_choice": {"type": "image_generation"},
        }
        return await self._post_image_generation_response(payload, image_config.timeout_seconds)

    async def _post_chat_completion(
        self,
        payload: dict,
        timeout_seconds: float,
        purpose: str,
        prompt_chars: int,
    ) -> str | None:
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
            text = extract_chat_completion_text(data)
            record = _chat_usage_record(
                purpose=purpose,
                model=str(payload.get("model", "")),
                prompt_chars=prompt_chars,
                completion_chars=len(text or ""),
                usage=data.get("usage") if isinstance(data.get("usage"), dict) else {},
            )
            _log_chat_usage(record)
            self._record_chat_usage(record)
            return text
        except (KeyError, TypeError, ValueError) as exc:
            logger.warning("LLM response parse failed: {}", exc)
            return None

    async def _post_image_generation_response(
        self,
        payload: dict,
        timeout_seconds: float,
    ) -> GeneratedImage | None:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        try:
            async with httpx.AsyncClient(timeout=timeout_seconds) as client:
                response = await client.post(self.responses_url, headers=headers, json=payload)
                response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            body = exc.response.text[:500]
            request_id = _response_request_id(exc.response)
            logger.warning(
                "LLM image generation HTTP error: status={} request_id={} body={}",
                exc.response.status_code,
                request_id or "(none)",
                body,
            )
            return None
        except httpx.HTTPError as exc:
            logger.warning("LLM image generation request failed: {}", exc)
            return None

        try:
            data = response.json()
        except (KeyError, TypeError, ValueError) as exc:
            logger.warning("LLM image generation response parse failed: {}", exc)
            return None
        try:
            generated = extract_generated_image(data)
        except (KeyError, TypeError, ValueError) as exc:
            logger.warning("LLM image generation response parse failed: {}", exc)
            return None
        if generated is None:
            logger.warning(
                "LLM image generation response had no image result: request_id={} "
                "status={} error={} incomplete_details={} output_types={} body={}",
                _response_request_id(response) or "(none)",
                data.get("status"),
                data.get("error"),
                data.get("incomplete_details"),
                _response_output_types(data),
                _json_preview(data),
            )
        return generated

    def _missing_config_items(self) -> list[str]:
        missing = []
        if not self.config.model:
            missing.append("llm.model")
        if not self.config.base_url:
            missing.append("llm.base_url")
        if not self.api_key:
            missing.append(f"{self.config.api_key_env}/llm.api_key")
        return missing

    def _record_chat_usage(self, record: LLMUsageRecord) -> None:
        if self.usage_recorder is None:
            return
        try:
            self.usage_recorder(record)
        except Exception as exc:  # pragma: no cover - usage telemetry must not break replies
            logger.warning("LLM usage record failed: {}", exc)


def build_llm_client(
    config: LLMConfig,
    usage_recorder: LLMUsageRecorder | None = None,
) -> LLMClient:
    provider = config.provider.lower().replace("_", "-")
    if provider == "disabled":
        return DisabledLLMClient()
    if provider in {"openai-compatible", "openai"}:
        return OpenAICompatibleLLMClient(config, usage_recorder=usage_recorder)
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


def normalize_responses_url(base_url: str) -> str:
    base = base_url.strip().rstrip("/")
    if base.endswith("/responses"):
        return base
    if base.endswith("/v1"):
        return f"{base}/responses"
    return f"{base}/v1/responses"


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


def extract_generated_image(data: dict) -> GeneratedImage | None:
    for item in data.get("output", []):
        if not isinstance(item, dict):
            continue
        if item.get("type") != "image_generation_call":
            continue
        result = str(item.get("result", "")).strip()
        if result:
            return _generated_image_from_result(result)
    return None


def _generated_image_from_result(result: str) -> GeneratedImage:
    if result.startswith(("http://", "https://")):
        return GeneratedImage(url=result)
    if result.startswith("data:"):
        header, encoded = result.split(",", 1)
        mime_type = header[5:].split(";", 1)[0] or "image/png"
        return GeneratedImage(data=base64.b64decode(encoded), mime_type=mime_type)
    return GeneratedImage(data=base64.b64decode(result), mime_type="image/png")


def _response_request_id(response: httpx.Response) -> str:
    return response.headers.get("x-request-id", "") or response.headers.get("openai-request-id", "")


def _chat_usage_record(
    *,
    purpose: str,
    model: str,
    prompt_chars: int,
    completion_chars: int,
    usage: dict,
) -> LLMUsageRecord:
    prompt_tokens = _usage_int(usage.get("prompt_tokens"))
    completion_tokens = _usage_int(usage.get("completion_tokens"))
    total_tokens = _usage_int(usage.get("total_tokens")) or prompt_tokens + completion_tokens
    return LLMUsageRecord(
        purpose=purpose or "(unspecified)",
        model=model,
        prompt_chars=max(0, int(prompt_chars)),
        completion_chars=max(0, int(completion_chars)),
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        created_at=int(time.time()),
    )


def _usage_int(value: object) -> int:
    try:
        parsed = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0
    return max(0, parsed)


def _log_chat_usage(record: LLMUsageRecord) -> None:
    logger.info(
        "LLM chat usage purpose={} model={} prompt_chars={} completion_chars={} "
        "prompt_tokens={} completion_tokens={} total_tokens={}",
        record.purpose,
        record.model or "(empty)",
        record.prompt_chars,
        record.completion_chars,
        record.prompt_tokens,
        record.completion_tokens,
        record.total_tokens,
    )


def _response_output_types(data: dict) -> list[str]:
    types = []
    for item in data.get("output", []):
        if isinstance(item, dict):
            types.append(str(item.get("type", "")) or "(missing)")
    return types


def _json_preview(data: dict, limit: int = 800) -> str:
    return json.dumps(data, ensure_ascii=False)[:limit]
