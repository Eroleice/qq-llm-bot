from __future__ import annotations

import httpx
from loguru import logger

from qq_llm_bot.llm_config_helpers import (
    GEMINI_PROVIDER_TYPES,
    OPENAI_COMPATIBLE_PROVIDER_TYPES,
    ResolvedLLMModel,
    resolve_model_config,
)
from qq_llm_bot.llm_gemini import (
    extract_gemini_generated_image,
    extract_gemini_text,
    gemini_payload_from_chat_payload,
    gemini_payload_from_image_generation_payload,
    gemini_usage,
    normalize_gemini_generate_content_url,
)
from qq_llm_bot.llm_models import GeneratedImage, LLMUsageRecord
from qq_llm_bot.llm_response_helpers import (
    _chat_usage_record,
    _image_generation_failure_summary,
    _json_preview,
    _log_chat_usage,
    _response_output_types,
    _response_request_id,
    extract_chat_completion_text,
    extract_generated_image,
    normalize_chat_completions_url,
    normalize_responses_url,
)


class LLMTransportMixin:
    async def _post_chat_completion(
        self,
        payload: dict,
        timeout_seconds: float,
        purpose: str,
        prompt_chars: int,
    ) -> str | None:
        self.last_chat_error = ""
        self._last_chat_failure_kind = ""
        self._last_chat_failure_status = 0
        self._last_chat_failure_body = ""
        try:
            request_config = resolve_model_config(
                self.config,
                str(payload.get("model", "")),
                require_supported_transport=True,
            )
        except ValueError as exc:
            self._last_chat_failure_kind = "config_error"
            self.last_chat_error = f"config_error={exc}"
            logger.warning("LLM request is not configured: {}", exc)
            return None
        if request_config.provider_type in GEMINI_PROVIDER_TYPES:
            return await self._post_gemini_chat_completion(
                request_config,
                payload,
                timeout_seconds,
                purpose,
                prompt_chars,
            )
        if request_config.provider_type not in OPENAI_COMPATIBLE_PROVIDER_TYPES:
            self._last_chat_failure_kind = "config_error"
            self.last_chat_error = f"config_error=unsupported provider type {request_config.provider_type}"
            logger.warning("LLM provider type is not supported: {}", request_config.provider_type)
            return None
        request_payload = dict(payload)
        request_payload["model"] = request_config.model
        headers = {
            "Authorization": f"Bearer {request_config.api_key}",
            "Content-Type": "application/json",
        }

        try:
            async with httpx.AsyncClient(timeout=timeout_seconds) as client:
                response = await client.post(
                    normalize_chat_completions_url(request_config.base_url),
                    headers=headers,
                    json=request_payload,
                )
                response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            body = exc.response.text[:500]
            self._last_chat_failure_kind = "http_error"
            self._last_chat_failure_status = exc.response.status_code
            self._last_chat_failure_body = body
            self.last_chat_error = f"http_status={exc.response.status_code} body={body}"
            logger.warning("LLM HTTP error: status={} body={}", exc.response.status_code, body)
            return None
        except httpx.HTTPError as exc:
            self._last_chat_failure_kind = "request_error"
            self.last_chat_error = f"request_error={type(exc).__name__}: {exc}"
            logger.warning("LLM request failed: {}", exc)
            return None

        try:
            data = response.json()
            text = extract_chat_completion_text(data)
            record = _chat_usage_record(
                purpose=purpose,
                model=request_config.model_ref,
                prompt_chars=prompt_chars,
                completion_chars=len(text or ""),
                usage=data.get("usage") if isinstance(data.get("usage"), dict) else {},
            )
            _log_chat_usage(record)
            self._record_chat_usage(record)
            return text
        except (KeyError, TypeError, ValueError) as exc:
            self._last_chat_failure_kind = "parse_error"
            self.last_chat_error = f"parse_error={type(exc).__name__}: {exc}"
            logger.warning("LLM response parse failed: {}", exc)
            return None

    async def _post_gemini_chat_completion(
        self,
        request_config: ResolvedLLMModel,
        payload: dict,
        timeout_seconds: float,
        purpose: str,
        prompt_chars: int,
    ) -> str | None:
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": request_config.api_key,
        }

        try:
            async with httpx.AsyncClient(timeout=timeout_seconds) as client:
                request_payload = await gemini_payload_from_chat_payload(payload, client)
                response = await client.post(
                    normalize_gemini_generate_content_url(
                        request_config.base_url,
                        request_config.model,
                    ),
                    headers=headers,
                    json=request_payload,
                )
                response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            body = exc.response.text[:500]
            self._last_chat_failure_kind = "http_error"
            self._last_chat_failure_status = exc.response.status_code
            self._last_chat_failure_body = body
            self.last_chat_error = f"http_status={exc.response.status_code} body={body}"
            logger.warning("Gemini LLM HTTP error: status={} body={}", exc.response.status_code, body)
            return None
        except (httpx.HTTPError, ValueError) as exc:
            self._last_chat_failure_kind = "request_error"
            self.last_chat_error = f"request_error={type(exc).__name__}: {exc}"
            logger.warning("Gemini LLM request failed: {}", exc)
            return None

        try:
            data = response.json()
            text = extract_gemini_text(data)
            record = _chat_usage_record(
                purpose=purpose,
                model=request_config.model_ref,
                prompt_chars=prompt_chars,
                completion_chars=len(text or ""),
                usage=gemini_usage(data),
            )
            _log_chat_usage(record)
            self._record_chat_usage(record)
            return text
        except (KeyError, TypeError, ValueError) as exc:
            self._last_chat_failure_kind = "parse_error"
            self.last_chat_error = f"parse_error={type(exc).__name__}: {exc}"
            logger.warning("Gemini LLM response parse failed: {}", exc)
            return None

    async def _post_image_generation_response(
        self,
        payload: dict,
        timeout_seconds: float,
    ) -> GeneratedImage | None:
        try:
            request_config = resolve_model_config(
                self.config,
                str(payload.get("model", "")),
                require_supported_transport=True,
            )
        except ValueError as exc:
            self._last_image_generation_failure_kind = "config_error"
            self.last_image_generation_error = f"config_error={exc}"
            logger.warning("LLM image generation is not configured: {}", exc)
            return None
        if request_config.provider_type in GEMINI_PROVIDER_TYPES:
            return await self._post_gemini_image_generation_response(
                request_config,
                payload,
                timeout_seconds,
            )
        if request_config.provider_type not in OPENAI_COMPATIBLE_PROVIDER_TYPES:
            self._last_image_generation_failure_kind = "config_error"
            self.last_image_generation_error = (
                f"config_error=unsupported provider type {request_config.provider_type}"
            )
            logger.warning(
                "LLM image generation provider type is not supported: {}",
                request_config.provider_type,
            )
            return None
        request_payload = dict(payload)
        request_payload["model"] = request_config.model
        headers = {
            "Authorization": f"Bearer {request_config.api_key}",
            "Content-Type": "application/json",
        }

        try:
            async with httpx.AsyncClient(timeout=timeout_seconds) as client:
                response = await client.post(
                    normalize_responses_url(request_config.base_url),
                    headers=headers,
                    json=request_payload,
                )
                response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            body = exc.response.text[:500]
            request_id = _response_request_id(exc.response)
            self._last_image_generation_failure_kind = "http_error"
            self.last_image_generation_error = (
                f"http_status={exc.response.status_code} request_id={request_id or '(none)'} "
                f"body={body}"
            )
            logger.warning(
                "LLM image generation HTTP error: status={} request_id={} body={}",
                exc.response.status_code,
                request_id or "(none)",
                body,
            )
            return None
        except httpx.HTTPError as exc:
            self._last_image_generation_failure_kind = "request_error"
            self.last_image_generation_error = f"request_error={type(exc).__name__}: {exc}"
            logger.warning("LLM image generation request failed: {}", exc)
            return None

        try:
            data = response.json()
        except (KeyError, TypeError, ValueError) as exc:
            self._last_image_generation_failure_kind = "parse_error"
            self.last_image_generation_error = f"json_parse_error={type(exc).__name__}: {exc}"
            logger.warning("LLM image generation response parse failed: {}", exc)
            return None
        try:
            generated = extract_generated_image(data)
        except (KeyError, TypeError, ValueError) as exc:
            self._last_image_generation_failure_kind = "parse_error"
            self.last_image_generation_error = (
                f"image_parse_error={type(exc).__name__}: {exc}; "
                f"{_image_generation_failure_summary(data, response)}"
            )
            logger.warning("LLM image generation response parse failed: {}", exc)
            return None
        if generated is None:
            self._last_image_generation_failure_kind = "no_image"
            self.last_image_generation_error = _image_generation_failure_summary(
                data,
                response,
            )
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

    async def _post_gemini_image_generation_response(
        self,
        request_config: ResolvedLLMModel,
        payload: dict,
        timeout_seconds: float,
    ) -> GeneratedImage | None:
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": request_config.api_key,
        }

        try:
            async with httpx.AsyncClient(timeout=timeout_seconds) as client:
                request_payload = await gemini_payload_from_image_generation_payload(
                    payload,
                    client,
                )
                response = await client.post(
                    normalize_gemini_generate_content_url(
                        request_config.base_url,
                        request_config.model,
                    ),
                    headers=headers,
                    json=request_payload,
                )
                response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            body = exc.response.text[:500]
            request_id = _response_request_id(exc.response)
            self._last_image_generation_failure_kind = "http_error"
            self.last_image_generation_error = (
                f"http_status={exc.response.status_code} request_id={request_id or '(none)'} "
                f"body={body}"
            )
            logger.warning(
                "Gemini image generation HTTP error: status={} request_id={} body={}",
                exc.response.status_code,
                request_id or "(none)",
                body,
            )
            return None
        except (httpx.HTTPError, ValueError) as exc:
            self._last_image_generation_failure_kind = "request_error"
            self.last_image_generation_error = f"request_error={type(exc).__name__}: {exc}"
            logger.warning("Gemini image generation request failed: {}", exc)
            return None

        try:
            data = response.json()
            generated = extract_gemini_generated_image(data)
        except (KeyError, TypeError, ValueError) as exc:
            self._last_image_generation_failure_kind = "parse_error"
            self.last_image_generation_error = f"image_parse_error={type(exc).__name__}: {exc}"
            logger.warning("Gemini image generation response parse failed: {}", exc)
            return None
        if generated is None:
            self._last_image_generation_failure_kind = "no_image"
            self.last_image_generation_error = _json_preview(data)
            logger.warning(
                "Gemini image generation response had no image result: body={}",
                _json_preview(data),
            )
        return generated

    def _record_chat_usage(self, record: LLMUsageRecord) -> None:
        if self.usage_recorder is None:
            return
        try:
            self.usage_recorder(record)
        except Exception as exc:  # pragma: no cover - usage telemetry must not break replies
            logger.warning("LLM usage record failed: {}", exc)
