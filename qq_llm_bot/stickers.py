from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import httpx
from loguru import logger

from qq_llm_bot.config import AppConfig
from qq_llm_bot.models import MessageContext, StickerCandidate


@dataclass(frozen=True)
class SavedStickerFile:
    local_path: str
    sha256: str
    content_type: str = ""


class StickerLocalStore:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.root = config.resolve_path(config.stickers.storage_dir)

    async def save_candidate(
        self,
        context: MessageContext,
        candidate: StickerCandidate,
    ) -> SavedStickerFile | None:
        if not self.config.stickers.enabled:
            return None
        if candidate.confidence < self.config.stickers.min_confidence:
            return None

        url = candidate.url.strip()
        if not url.lower().startswith(("http://", "https://")):
            return None

        try:
            data, content_type = await self._download(url)
        except Exception as exc:  # pragma: no cover - depends on OneBot/NapCat network state
            logger.warning("Sticker download failed for {}: {}", url, exc)
            return None

        digest = hashlib.sha256(data).hexdigest()
        suffix = _image_suffix(content_type, url)
        target_dir = self.root / _safe_path_part(context.group_id)
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / f"{digest[:32]}{suffix}"
        if not target.exists():
            target.write_bytes(data)
        return SavedStickerFile(
            local_path=str(target.resolve()),
            sha256=digest,
            content_type=content_type,
        )

    async def _download(self, url: str) -> tuple[bytes, str]:
        max_bytes = self.config.stickers.max_download_bytes
        timeout = self.config.stickers.download_timeout_seconds
        chunks: list[bytes] = []
        total = 0

        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            async with client.stream("GET", url) as response:
                response.raise_for_status()
                content_type = response.headers.get("content-type", "").split(";", 1)[0].strip()
                content_length = response.headers.get("content-length", "").strip()
                if content_length:
                    try:
                        parsed_length = int(content_length)
                    except ValueError:
                        parsed_length = 0
                    if parsed_length > max_bytes:
                        raise ValueError("sticker image is too large")

                async for chunk in response.aiter_bytes():
                    if not chunk:
                        continue
                    total += len(chunk)
                    if total > max_bytes:
                        raise ValueError("sticker image is too large")
                    chunks.append(chunk)

        return b"".join(chunks), content_type


def _image_suffix(content_type: str, url: str) -> str:
    mime_suffixes = {
        "image/jpeg": ".jpg",
        "image/jpg": ".jpg",
        "image/png": ".png",
        "image/gif": ".gif",
        "image/webp": ".webp",
        "image/bmp": ".bmp",
    }
    content_suffix = mime_suffixes.get(content_type.lower())
    if content_suffix:
        return content_suffix

    url_suffix = Path(urlparse(url).path).suffix.lower()
    if url_suffix in {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"}:
        return ".jpg" if url_suffix == ".jpeg" else url_suffix
    return ".img"


def _safe_path_part(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(value))
    return cleaned[:80] or "unknown"
