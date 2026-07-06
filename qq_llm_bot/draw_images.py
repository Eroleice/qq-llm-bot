from __future__ import annotations

import base64
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import httpx
from loguru import logger

from qq_llm_bot.models import MessageAttachment


@dataclass(frozen=True)
class PreparedDrawImages:
    image_urls: list[str]
    error: str = ""


async def prepare_draw_reference_images(
    attachments: Iterable[MessageAttachment],
    *,
    max_images: int,
    max_bytes: int,
    max_dimension: int,
    quality: int,
    timeout_seconds: float,
) -> PreparedDrawImages:
    images = _dedupe_image_attachments(attachments)
    if len(images) > max_images:
        return PreparedDrawImages(
            [],
            f"参考图最多支持 {max_images} 张，请减少图片数量后再试。",
        )

    prepared: list[str] = []
    for image in images:
        ref = _attachment_ref(image)
        if not ref:
            return PreparedDrawImages([], "有参考图缺少可读取的 URL。")
        result = await _prepare_single_image(
            ref,
            max_bytes=max_bytes,
            max_dimension=max_dimension,
            quality=quality,
            timeout_seconds=timeout_seconds,
        )
        if not result:
            return PreparedDrawImages([], "参考图读取或压缩失败，请换一张图片试试。")
        prepared.append(result)
    return PreparedDrawImages(prepared)


def _dedupe_image_attachments(attachments: Iterable[MessageAttachment]) -> list[MessageAttachment]:
    seen: set[str] = set()
    deduped: list[MessageAttachment] = []
    for attachment in attachments:
        if attachment.attachment_type != "image":
            continue
        ref = _attachment_ref(attachment)
        key = ref or attachment.raw_data or attachment.file
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(attachment)
    return deduped


async def _prepare_single_image(
    ref: str,
    *,
    max_bytes: int,
    max_dimension: int,
    quality: int,
    timeout_seconds: float,
) -> str | None:
    try:
        raw, mime_type = await _read_image_bytes(ref, timeout_seconds)
    except Exception as exc:  # pragma: no cover - third-party URL/filesystem boundary
        logger.warning("Draw reference image read failed for {}: {}", ref, exc)
        return None

    if not raw:
        return None
    if _image_fits(raw, max_bytes=max_bytes, max_dimension=max_dimension):
        return ref if ref.startswith(("http://", "https://", "data:")) else _data_url(raw, mime_type)

    compressed = _compress_image(raw, max_bytes=max_bytes, max_dimension=max_dimension, quality=quality)
    if not compressed:
        return None
    return _data_url(compressed, "image/jpeg")


async def _read_image_bytes(ref: str, timeout_seconds: float) -> tuple[bytes, str]:
    if ref.startswith("data:"):
        header, encoded = ref.split(",", 1)
        mime_type = header[5:].split(";", 1)[0] or "image/png"
        return base64.b64decode(encoded), mime_type
    if ref.startswith(("http://", "https://")):
        async with httpx.AsyncClient(timeout=timeout_seconds, follow_redirects=True) as client:
            response = await client.get(ref)
            response.raise_for_status()
        return response.content, _content_type(response.headers.get("content-type", ""))
    path = Path(ref.removeprefix("file://"))
    return path.read_bytes(), _mime_from_suffix(path.suffix)


def _image_fits(raw: bytes, *, max_bytes: int, max_dimension: int) -> bool:
    if len(raw) > max_bytes:
        return False
    try:
        from PIL import Image

        with Image.open(io.BytesIO(raw)) as image:
            return max(image.size) <= max_dimension
    except Exception:
        return False


def _compress_image(
    raw: bytes,
    *,
    max_bytes: int,
    max_dimension: int,
    quality: int,
) -> bytes | None:
    try:
        from PIL import Image

        with Image.open(io.BytesIO(raw)) as source:
            image = source.convert("RGBA")
            for dimension in _dimension_steps(max_dimension):
                resized = image.copy()
                resized.thumbnail((dimension, dimension))
                background = Image.new("RGB", resized.size, "white")
                background.paste(resized, mask=resized.getchannel("A"))
                for current_quality in _quality_steps(quality):
                    output = io.BytesIO()
                    background.save(output, format="JPEG", quality=current_quality, optimize=True)
                    data = output.getvalue()
                    if len(data) <= max_bytes:
                        return data
            return None
    except Exception as exc:
        logger.warning("Draw reference image compression failed: {}", exc)
        return None


def _dimension_steps(max_dimension: int) -> list[int]:
    start = max(64, int(max_dimension))
    steps = [start]
    current = start
    while current > 256:
        current = max(256, int(current * 0.75))
        steps.append(current)
    if 128 not in steps:
        steps.append(128)
    return list(dict.fromkeys(step for step in steps if step > 0))


def _quality_steps(quality: int) -> list[int]:
    start = max(30, min(95, int(quality)))
    steps = [start, 75, 65, 55, 45, 35]
    return list(dict.fromkeys(step for step in steps if 30 <= step <= start))


def _attachment_ref(attachment: MessageAttachment) -> str:
    url = str(attachment.url or "").strip()
    if url:
        return url
    file = str(attachment.file or "").strip()
    if file.startswith(("http://", "https://", "data:", "file://")) or Path(file).exists():
        return file
    return ""


def _data_url(raw: bytes, mime_type: str) -> str:
    encoded = base64.b64encode(raw).decode("ascii")
    return f"data:{mime_type or 'image/jpeg'};base64,{encoded}"


def _content_type(value: str) -> str:
    content_type = str(value or "").split(";", 1)[0].strip().lower()
    return content_type if content_type.startswith("image/") else "image/png"


def _mime_from_suffix(suffix: str) -> str:
    normalized = suffix.lower()
    if normalized in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if normalized == ".webp":
        return "image/webp"
    if normalized == ".gif":
        return "image/gif"
    return "image/png"
