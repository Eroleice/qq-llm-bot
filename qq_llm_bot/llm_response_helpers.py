from __future__ import annotations

import base64
import json
import time

import httpx
from loguru import logger

from qq_llm_bot.llm_models import GeneratedImage, LLMUsageRecord


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


def _image_generation_input(prompt: str, image_urls: list[str]) -> object:
    clean_urls = [url.strip() for url in image_urls if str(url or "").strip()]
    if not clean_urls:
        return prompt
    content: list[dict[str, str]] = [{"type": "input_text", "text": prompt}]
    content.extend({"type": "input_image", "image_url": url} for url in clean_urls)
    return [{"role": "user", "content": content}]


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


def _image_generation_failure_summary(data: dict, response: httpx.Response) -> str:
    return (
        f"request_id={_response_request_id(response) or '(none)'} "
        f"status={data.get('status') or '(missing)'} "
        f"error={_compact_json(data.get('error'))} "
        f"incomplete={_compact_json(data.get('incomplete_details'))} "
        f"output={_response_output_summary(data)}"
    )


def _response_output_summary(data: dict) -> str:
    parts = []
    for item in data.get("output", []):
        if not isinstance(item, dict):
            continue
        keys = ",".join(str(key) for key in item.keys())
        result = item.get("result")
        result_state = "present" if result else "empty"
        parts.append(
            f"{item.get('type') or '(missing)'}"
            f"/status={item.get('status') or '(none)'}"
            f"/result={result_state}"
            f"/keys={keys}"
        )
    return "[" + "; ".join(parts)[:700] + "]"


def _compact_json(value: object) -> str:
    if value in (None, "", [], {}):
        return "(none)"
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))[:300]


def _json_preview(data: dict, limit: int = 800) -> str:
    return json.dumps(data, ensure_ascii=False)[:limit]
