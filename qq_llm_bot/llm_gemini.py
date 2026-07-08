from __future__ import annotations

import base64
import math
import mimetypes
from typing import Any
from urllib.parse import quote

import httpx

from qq_llm_bot.llm_models import GeneratedImage

GEMINI_PROVIDER_TYPES = {"gemini"}
MAX_GEMINI_INLINE_IMAGE_BYTES = 8 * 1024 * 1024


def normalize_gemini_generate_content_url(base_url: str, model: str) -> str:
    base = base_url.strip().rstrip("/")
    if not base:
        return ""
    if base.endswith(":generateContent"):
        return base
    if "{model}" in base:
        return base.replace("{model}", quote(model, safe="/:._-"))
    if base.endswith("/generateContent"):
        return base[:-len("/generateContent")] + ":generateContent"
    if base.endswith(f"/models/{model}"):
        return f"{base}:generateContent"
    model_path = model if model.startswith(("models/", "tunedModels/")) else f"models/{model}"
    quoted_model_path = quote(model_path, safe="/:._-")
    if base.endswith(("/v1", "/v1beta")):
        return f"{base}/{quoted_model_path}:generateContent"
    return f"{base}/v1beta/{quoted_model_path}:generateContent"


async def gemini_payload_from_chat_payload(
    payload: dict[str, Any],
    client: httpx.AsyncClient,
) -> dict[str, Any]:
    system_parts: list[dict[str, Any]] = []
    contents: list[dict[str, Any]] = []
    for message in payload.get("messages", []):
        if not isinstance(message, dict):
            continue
        parts = await _gemini_parts_from_openai_content(message.get("content", ""), client)
        if not parts:
            continue
        role = str(message.get("role", "user")).strip().lower()
        if role == "system":
            system_parts.extend(_text_only_parts(parts))
            continue
        contents.append({"role": "model" if role == "assistant" else "user", "parts": parts})

    if not contents:
        contents.append({"role": "user", "parts": [{"text": ""}]})

    gemini_payload: dict[str, Any] = {"contents": contents}
    if system_parts:
        gemini_payload["systemInstruction"] = {"parts": system_parts}
    generation_config = _gemini_generation_config(payload)
    if generation_config:
        gemini_payload["generationConfig"] = generation_config
    return gemini_payload


async def gemini_payload_from_image_generation_payload(
    payload: dict[str, Any],
    client: httpx.AsyncClient,
) -> dict[str, Any]:
    parts = await _gemini_parts_from_image_generation_input(payload.get("input", ""), client)
    if not parts:
        parts = [{"text": ""}]
    generation_config: dict[str, Any] = {"responseModalities": ["TEXT", "IMAGE"]}
    aspect_ratio = _image_generation_aspect_ratio(payload)
    if aspect_ratio:
        generation_config["imageConfig"] = {"aspectRatio": aspect_ratio}
    return {
        "contents": [{"role": "user", "parts": parts}],
        "generationConfig": generation_config,
    }


def extract_gemini_text(data: dict[str, Any]) -> str | None:
    text_parts = []
    for candidate in data.get("candidates", []):
        if not isinstance(candidate, dict):
            continue
        content = candidate.get("content") or {}
        if not isinstance(content, dict):
            continue
        for part in content.get("parts", []):
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                text_parts.append(part["text"])
    text = "".join(text_parts).strip()
    return text or None


def extract_gemini_generated_image(data: dict[str, Any]) -> GeneratedImage | None:
    for candidate in data.get("candidates", []):
        if not isinstance(candidate, dict):
            continue
        content = candidate.get("content") or {}
        if not isinstance(content, dict):
            continue
        for part in content.get("parts", []):
            if not isinstance(part, dict):
                continue
            inline_data = part.get("inlineData") or part.get("inline_data") or {}
            if not isinstance(inline_data, dict):
                continue
            encoded = str(inline_data.get("data", "")).strip()
            if not encoded:
                continue
            mime_type = (
                str(inline_data.get("mimeType", inline_data.get("mime_type", ""))).strip()
                or "image/png"
            )
            return GeneratedImage(data=base64.b64decode(encoded), mime_type=mime_type)
    return None


def gemini_usage(data: dict[str, Any]) -> dict[str, int]:
    usage = data.get("usageMetadata") or data.get("usage_metadata") or {}
    if not isinstance(usage, dict):
        return {}
    prompt_tokens = _usage_int(usage.get("promptTokenCount", usage.get("prompt_token_count")))
    completion_tokens = _usage_int(
        usage.get("candidatesTokenCount", usage.get("candidates_token_count"))
    )
    total_tokens = _usage_int(usage.get("totalTokenCount", usage.get("total_token_count")))
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens or prompt_tokens + completion_tokens,
    }


def _gemini_generation_config(payload: dict[str, Any]) -> dict[str, Any]:
    config: dict[str, Any] = {}
    if "temperature" in payload:
        config["temperature"] = payload["temperature"]
    max_tokens = payload.get("max_tokens")
    if max_tokens:
        config["maxOutputTokens"] = max_tokens
    return config


async def _gemini_parts_from_openai_content(
    content: Any,
    client: httpx.AsyncClient,
) -> list[dict[str, Any]]:
    if isinstance(content, str):
        return [{"text": content}]
    if not isinstance(content, list):
        return [{"text": str(content or "")}]

    parts: list[dict[str, Any]] = []
    for item in content:
        if not isinstance(item, dict):
            continue
        item_type = str(item.get("type", "")).strip()
        if item_type in {"text", "input_text"}:
            text = str(item.get("text", "")).strip()
            if text:
                parts.append({"text": text})
            continue
        if item_type in {"image_url", "input_image"}:
            image_url = item.get("image_url", "")
            if isinstance(image_url, dict):
                image_url = image_url.get("url", "")
            image_part = await _gemini_image_part(str(image_url or ""), client)
            if image_part:
                parts.append(image_part)
    return parts


async def _gemini_parts_from_image_generation_input(
    value: Any,
    client: httpx.AsyncClient,
) -> list[dict[str, Any]]:
    if isinstance(value, str):
        return [{"text": value.strip()}] if value.strip() else []
    if not isinstance(value, list):
        return [{"text": str(value or "").strip()}] if str(value or "").strip() else []

    parts: list[dict[str, Any]] = []
    for message in value:
        if not isinstance(message, dict):
            continue
        content = message.get("content", "")
        parts.extend(await _gemini_parts_from_openai_content(content, client))
    return parts


async def _gemini_image_part(url: str, client: httpx.AsyncClient) -> dict[str, Any] | None:
    clean_url = url.strip()
    if not clean_url:
        return None
    if clean_url.startswith("data:"):
        return _gemini_data_url_part(clean_url)
    response = await client.get(clean_url, follow_redirects=True)
    response.raise_for_status()
    data = response.content
    if len(data) > MAX_GEMINI_INLINE_IMAGE_BYTES:
        raise ValueError(
            "image is too large for Gemini inlineData "
            f"({len(data)} > {MAX_GEMINI_INLINE_IMAGE_BYTES} bytes)"
        )
    mime_type = _image_mime_type(response.headers.get("content-type", ""), clean_url)
    return {
        "inlineData": {
            "mimeType": mime_type,
            "data": base64.b64encode(data).decode("ascii"),
        }
    }


def _gemini_data_url_part(url: str) -> dict[str, Any]:
    header, separator, encoded = url.partition(",")
    if not separator:
        raise ValueError("invalid data URL for Gemini image input")
    mime_type = header[5:].split(";", 1)[0] or "image/png"
    return {"inlineData": {"mimeType": mime_type, "data": encoded.strip()}}


def _text_only_parts(parts: list[dict[str, Any]]) -> list[dict[str, str]]:
    text_parts = []
    for part in parts:
        text = str(part.get("text", "")).strip()
        if text:
            text_parts.append({"text": text})
    return text_parts


def _image_generation_aspect_ratio(payload: dict[str, Any]) -> str:
    tools = payload.get("tools")
    if not isinstance(tools, list) or not tools or not isinstance(tools[0], dict):
        return ""
    size = str(tools[0].get("size", "")).strip().lower()
    width, separator, height = size.partition("x")
    if not separator:
        return ""
    try:
        width_value = int(width)
        height_value = int(height)
    except ValueError:
        return ""
    if width_value <= 0 or height_value <= 0:
        return ""
    divisor = math.gcd(width_value, height_value)
    return f"{width_value // divisor}:{height_value // divisor}"


def _image_mime_type(content_type: str, url: str) -> str:
    header_mime = content_type.split(";", 1)[0].strip().lower()
    if header_mime.startswith("image/"):
        return header_mime
    guessed, _ = mimetypes.guess_type(url)
    if guessed and guessed.startswith("image/"):
        return guessed
    return "image/jpeg"


def _usage_int(value: object) -> int:
    try:
        parsed = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0
    return max(0, parsed)
