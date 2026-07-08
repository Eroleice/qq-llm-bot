from __future__ import annotations

from qq_llm_bot.models import FactRecord, MemoryRecord, StickerAssetRecord


def format_memory_record(record: MemoryRecord) -> str:
    return (
        f"#{record.id} [{record.kind}/{record.status}/{record.claim_scope}] "
        f"{record.content} -> {record.owner_type}:{record.owner_id} "
        f"(src={record.source_user_id}, subject={record.subject_user_id}, "
        f"conf={record.confidence:.2f}, imp={record.importance:.2f})"
    )

def format_fact_record(record: FactRecord) -> str:
    return (
        f"#{record.id} [{record.fact_type}/{record.status}/{record.claim_scope}] "
        f"{record.claim_text} -> user:{record.subject_user_id} "
        f"(topic={record.topic}, stance={record.stance or '-'}, "
        f"src={record.source_user_id}, conf={record.confidence:.2f})"
    )

def format_sticker_asset(asset: StickerAssetRecord) -> str:
    status = "enabled" if asset.enabled else "disabled"
    tags = "、".join(asset.tags[:5]) or "(no tags)"
    usage = asset.usage or asset.description or "(no usage)"
    return (
        f"#{asset.id} [{status}] mood={asset.mood or '(unknown)'} "
        f"tags={tags} conf={asset.confidence:.2f} hits={asset.hit_count}\n"
        f"用途：{usage}\n"
        f"本地：{asset.local_path}"
    )
