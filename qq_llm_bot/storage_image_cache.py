from __future__ import annotations

import json
import time
from collections.abc import Iterable
from dataclasses import replace

from qq_llm_bot.models import ImageVisionCacheRecord
from qq_llm_bot.storage_record_rows import _image_vision_cache_record
from qq_llm_bot.storage_record_values import _compact_string_list


def get_image_vision_cache(storage: object, url: str) -> ImageVisionCacheRecord | None:
    normalized_url = str(url).strip()
    if not normalized_url:
        return None
    now = int(time.time())
    with storage._connect() as conn:
        row = conn.execute(
            """
            SELECT url, description, ocr_text, topics, memory, confidence, importance,
                   model, created_at, updated_at, last_seen_at, hit_count
            FROM image_vision_cache
            WHERE url = ?
            """,
            (normalized_url,),
        ).fetchone()
        if row is None:
            return None
        conn.execute(
            """
            UPDATE image_vision_cache
            SET last_seen_at = ?,
                hit_count = hit_count + 1
            WHERE url = ?
            """,
            (now, normalized_url),
        )
    record = _image_vision_cache_record(row)
    return replace(record, last_seen_at=now, hit_count=record.hit_count + 1)

def upsert_image_vision_cache(
    storage: object,
    *,
    url: str,
    description: str,
    ocr_text: str = "",
    topics: Iterable[str] = (),
    memory: str = "",
    confidence: float = 0.0,
    importance: float = 0.5,
    model: str = "",
) -> None:
    normalized_url = str(url).strip()
    description = str(description).strip()
    ocr_text = str(ocr_text).strip()
    memory = str(memory).strip()
    if not normalized_url or not (description or ocr_text or memory):
        return
    now = int(time.time())
    topics_json = json.dumps(_compact_string_list(topics, limit=10), ensure_ascii=False)
    with storage._connect() as conn:
        conn.execute(
            """
            INSERT INTO image_vision_cache (
                url, description, ocr_text, topics, memory, confidence, importance,
                model, created_at, updated_at, last_seen_at, hit_count
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
            ON CONFLICT(url) DO UPDATE SET
                description = excluded.description,
                ocr_text = excluded.ocr_text,
                topics = excluded.topics,
                memory = excluded.memory,
                confidence = excluded.confidence,
                importance = excluded.importance,
                model = excluded.model,
                updated_at = excluded.updated_at,
                last_seen_at = excluded.last_seen_at,
                hit_count = image_vision_cache.hit_count + 1
            """,
            (
                normalized_url,
                description[:1000],
                ocr_text[:1000],
                topics_json,
                memory[:1000],
                float(confidence),
                float(importance),
                str(model).strip()[:80],
                now,
                now,
                now,
            ),
        )
