from __future__ import annotations

import hashlib
import io
import time
from dataclasses import dataclass

from loguru import logger

from qq_llm_bot.config import AppConfig
from qq_llm_bot.llm import GeneratedImage
from qq_llm_bot.models import MessageContext
from qq_llm_bot.text_utils import safe_path_part as _safe_path_part


@dataclass(frozen=True)
class SavedGeneratedImage:
    file_ref: str
    local_path: str = ""
    url: str = ""
    sha256: str = ""
    mime_type: str = ""


class GeneratedImageStore:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.root = config.resolve_path(config.image_generation.storage_dir)

    def save(self, context: MessageContext, image: GeneratedImage) -> SavedGeneratedImage | None:
        if image.url:
            return SavedGeneratedImage(file_ref=image.url, url=image.url, mime_type=image.mime_type)
        if not image.data:
            return None

        image_data, mime_type = _prepare_chat_image(
            image.data,
            image.mime_type,
            self.config.image_generation.output_format,
            self.config.image_generation.output_compression,
            self.config.image_generation.max_send_dimension,
        )
        digest = hashlib.sha256(image_data).hexdigest()
        suffix = _image_suffix(mime_type)
        target_dir = self.root / _safe_path_part(context.group_id, limit=64)
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / f"{int(time.time())}-{digest[:16]}{suffix}"
        if not target.exists():
            target.write_bytes(image_data)
        local_path = str(target.resolve())
        return SavedGeneratedImage(
            file_ref=local_path,
            local_path=local_path,
            sha256=digest,
            mime_type=mime_type,
        )


def _image_suffix(mime_type: str) -> str:
    suffixes = {
        "image/jpeg": ".jpg",
        "image/jpg": ".jpg",
        "image/png": ".png",
        "image/webp": ".webp",
    }
    return suffixes.get(mime_type.lower().strip(), ".png")


def _prepare_chat_image(
    data: bytes,
    mime_type: str,
    output_format: str,
    output_compression: int,
    max_dimension: int,
) -> tuple[bytes, str]:
    if not data or max_dimension <= 0:
        return data, mime_type
    try:
        from PIL import Image, ImageOps
    except ImportError:
        logger.warning("Pillow is not installed; generated image will be saved without resizing")
        return data, mime_type

    try:
        with Image.open(io.BytesIO(data)) as image:
            image = ImageOps.exif_transpose(image)
            if max(image.size) > max_dimension:
                image.thumbnail((max_dimension, max_dimension), Image.Resampling.LANCZOS)

            target_format = _pillow_output_format(output_format)
            target_mime = _mime_type_for_output_format(target_format)
            if target_format == "JPEG" and image.mode not in {"RGB", "L"}:
                image = image.convert("RGB")

            output = io.BytesIO()
            save_kwargs: dict[str, object] = {}
            if target_format == "JPEG":
                save_kwargs.update(
                    {
                        "quality": output_compression,
                        "optimize": True,
                        "progressive": True,
                    }
                )
            elif target_format == "WEBP":
                save_kwargs.update({"quality": output_compression, "method": 4})
            image.save(output, format=target_format, **save_kwargs)
            return output.getvalue(), target_mime
    except Exception as exc:  # pragma: no cover - corrupt image fallback
        logger.warning("Generated image resize/compress failed: {}", exc)
        return data, mime_type


def _pillow_output_format(output_format: str) -> str:
    normalized = output_format.lower().strip()
    if normalized in {"jpg", "jpeg"}:
        return "JPEG"
    if normalized == "webp":
        return "WEBP"
    return "PNG"


def _mime_type_for_output_format(output_format: str) -> str:
    if output_format == "JPEG":
        return "image/jpeg"
    if output_format == "WEBP":
        return "image/webp"
    return "image/png"

