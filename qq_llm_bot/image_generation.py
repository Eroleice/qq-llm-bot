from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from pathlib import Path

from qq_llm_bot.config import AppConfig
from qq_llm_bot.llm import GeneratedImage
from qq_llm_bot.models import MessageContext


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

        digest = hashlib.sha256(image.data).hexdigest()
        suffix = _image_suffix(image.mime_type)
        target_dir = self.root / _safe_path_part(context.group_id)
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / f"{int(time.time())}-{digest[:16]}{suffix}"
        if not target.exists():
            target.write_bytes(image.data)
        local_path = str(target.resolve())
        return SavedGeneratedImage(
            file_ref=local_path,
            local_path=local_path,
            sha256=digest,
            mime_type=image.mime_type,
        )


def _image_suffix(mime_type: str) -> str:
    suffixes = {
        "image/jpeg": ".jpg",
        "image/jpg": ".jpg",
        "image/png": ".png",
        "image/webp": ".webp",
    }
    return suffixes.get(mime_type.lower().strip(), ".png")


def _safe_path_part(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(value))
    return cleaned[:64] or "unknown"
