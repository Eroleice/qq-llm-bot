from __future__ import annotations

import json
import re

from qq_llm_bot.models import (
    ConversationSnapshot,
    FactRecord,
    MemoryCandidate,
    MemoryRecord,
    MessageContext,
    SemanticContext,
    StickerAssetRecord,
    TargetUserContext,
    UserProfileRecord,
)
from qq_llm_bot.onebot_messages import format_mention_label
from qq_llm_bot.web_search import SearchResult


def format_memory_candidates(memories: list[MemoryCandidate]) -> str:
    if not memories:
        return "(none)"
    return "\n".join(f"[{item.kind}] {item.content}" for item in memories)


def format_fact_records(facts: list[FactRecord]) -> str:
    if not facts:
        return "(none)"
    return "\n".join(
        f"#{fact.id} [{fact.fact_type}/{fact.stance or 'unknown'}] "
        f"{fact.claim_text} (topic={fact.topic}, conf={fact.confidence:.2f}, "
        f"evidence={fact.evidence_text})"
        for fact in facts
    )


def semantic_context_has_content(context: SemanticContext | None) -> bool:
    return bool(
        context
        and (
            context.current_intent
            or context.relevant_messages
            or context.resolved_references
            or context.member_context
            or context.uncertain_references
            or context.ignored_noise
        )
    )


def format_semantic_context(context: SemanticContext | None) -> str:
    if not semantic_context_has_content(context):
        return "(none)"
    assert context is not None
    sections: list[str] = []
    if context.current_intent:
        sections.append(f"当前用户意图：{context.current_intent}")
    if context.relevant_messages:
        sections.append("相关聊天记录：\n" + join_lines(context.relevant_messages))
    if context.resolved_references:
        sections.append("指代/称呼解析：\n" + join_lines(context.resolved_references))
    if context.member_context:
        sections.append("与话题相关的成员认知：\n" + join_lines(context.member_context))
    if context.uncertain_references:
        sections.append("不确定项：\n" + join_lines(context.uncertain_references))
    if context.ignored_noise:
        sections.append("可忽略噪音：\n" + join_lines(context.ignored_noise))
    return "\n\n".join(sections)


def compact_target_context(target: TargetUserContext) -> str:
    aliases = "、".join(target.aliases[:5])
    facts = [
        fact.claim_text
        for fact in target.facts[:3]
        if fact.fact_type in {"identity", "alias", "preference", "dislike", "boundary", "experience"}
    ]
    parts = [f"QQ:{target.user_id} status={target.resolution_status} reason={target.match_reason}"]
    if aliases:
        parts.append(f"aliases={aliases}")
    if target.profile and target.profile.summary:
        parts.append(f"profile={target.profile.summary}")
    if facts:
        parts.append("facts=" + "；".join(facts))
    return " ".join(parts)


def format_user_profile_record(profile: UserProfileRecord | None) -> str:
    if profile is None:
        return "(none)"
    traits = json.dumps(profile.traits, ensure_ascii=False) if profile.traits else "{}"
    return (
        f"v{profile.version} facts={profile.fact_count}\n"
        f"{profile.summary}\ntraits={traits}"
    )


def format_target_user_contexts(snapshot: ConversationSnapshot) -> str:
    lines: list[str] = []
    for target in snapshot.target_users:
        aliases = "、".join(target.aliases[:8]) or "(none)"
        lines.append(
            f"QQ:{target.user_id} status={target.resolution_status} reason={target.match_reason}\n"
            f"aliases={aliases}\n"
            f"profile={format_user_profile_record(target.profile)}\n"
            f"facts=\n{format_fact_records(target.facts[:8])}"
        )
    if snapshot.unknown_name_refs:
        lines.append("unknown_names=" + "、".join(snapshot.unknown_name_refs))
    if snapshot.ambiguous_name_refs:
        ambiguous = [
            f"{name}: {', '.join(user_ids)}"
            for name, user_ids in snapshot.ambiguous_name_refs.items()
        ]
        lines.append("ambiguous_names=" + "；".join(ambiguous))
    return "\n\n".join(lines) if lines else "(none)"


def format_sticker_assets(assets: list[StickerAssetRecord]) -> str:
    if not assets:
        return "(none)"
    lines = []
    for asset in assets:
        tags = "、".join(asset.tags[:6])
        lines.append(
            f"#{asset.id} mood={asset.mood or '(unknown)'} "
            f"tags={tags or '(none)'} usage={asset.usage or asset.description}"
        )
    return "\n".join(lines)


def format_search_results(results: list[SearchResult]) -> str:
    lines = []
    for index, result in enumerate(results[:5], start=1):
        title = " ".join(result.title.split())[:120]
        url = result.url.strip()
        snippet = " ".join(result.snippet.split())[:240]
        lines.append(f"{index}. {title}\nURL: {url}\n摘要: {snippet or '(empty)'}")
    return "\n".join(lines) if lines else "(none)"


def fallback_search_definition(results: list[SearchResult]) -> str:
    for result in results:
        snippet = clean_lexicon_definition(result.snippet)
        if snippet:
            return snippet
    if results:
        return clean_lexicon_definition(results[0].title)
    return ""


def clean_lexicon_definition(value: str) -> str:
    text = " ".join(str(value).strip().split())
    text = re.sub(r"^释义[:：]\s*", "", text)
    return text[:100].strip()


def format_memories(memories: list[MemoryRecord]) -> str:
    if not memories:
        return "(none)"
    return "\n".join(f"#{m.id} [{m.kind}] {m.content}" for m in memories)


def format_mentions(context: MessageContext) -> str:
    if not context.mentions:
        return "(none)"
    lines = []
    for mention in context.mentions:
        suffix = " bot" if mention.is_bot else ""
        lines.append(f"{format_mention_label(mention)} -> QQ:{mention.user_id}{suffix}")
    return "\n".join(lines)


def format_relationship(snapshot: ConversationSnapshot) -> str:
    relation = snapshot.relationship
    if relation is None:
        return "(none)"
    return (
        f"closeness={relation.closeness}, trust={relation.trust}, "
        f"familiarity={relation.familiarity}, tension={relation.tension}, "
        f"summary={relation.summary or '(empty)'}"
    )


def join_lines(lines: list[str]) -> str:
    return "\n".join(lines) if lines else "(none)"


def format_recent_context(snapshot: ConversationSnapshot) -> str:
    if not snapshot.speaker_recent_messages and not snapshot.other_recent_messages:
        return join_lines(snapshot.recent_messages)

    sections = [
        "优先根据“当前发言人近期主线”理解当前消息；其他发言只作为话题背景参考。"
    ]
    if snapshot.speaker_recent_messages:
        sections.append(
            "当前发言人近期主线：\n"
            f"{join_lines(snapshot.speaker_recent_messages)}"
        )
    if snapshot.other_recent_messages:
        sections.append(
            "其他发言近期话题参考：\n"
            f"{join_lines(snapshot.other_recent_messages)}"
        )
    return "\n\n".join(sections)
