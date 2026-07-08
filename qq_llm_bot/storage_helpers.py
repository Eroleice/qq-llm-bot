from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

from qq_llm_bot.models import FactCandidate, FactRecord, MessageContext
from qq_llm_bot.relationship_summary import merge_relationship_summary
from qq_llm_bot.storage_records import _dashboard_user_id


RELATIONAL_ALIAS_TERMS = {
    "主人",
    "主子",
    "老板",
    "老板娘",
    "领导",
    "管理员",
    "管理",
    "群主",
    "版主",
    "爸爸",
    "爸",
    "爹",
    "父亲",
    "妈妈",
    "妈",
    "母亲",
    "哥哥",
    "哥",
    "姐姐",
    "姐",
    "弟弟",
    "弟",
    "妹妹",
    "妹",
    "儿子",
    "女儿",
    "老婆",
    "老公",
    "媳妇",
    "丈夫",
    "妻子",
    "对象",
    "男朋友",
    "女朋友",
}


@dataclass(frozen=True)
class NameResolutionMatch:
    user_id: str
    reason: str
    score: float
    status: str = "resolved"


def clamp_float(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def safe_claim_scope(value: str) -> str:
    return value if value in {"self_report", "third_party", "bot_directed", "group_fact"} else "self_report"


def safe_verification_status(value: str) -> str:
    return value if value in {"accepted", "pending_confirmation", "conflict", "rejected"} else "pending_confirmation"


def safe_fact_status(value: str) -> str:
    return (
        value
        if value in {"candidate", "accepted", "pending_confirmation", "rejected", "superseded", "forgotten"}
        else "candidate"
    )


def clean_fact_field(value: str, limit: int) -> str:
    return " ".join(str(value or "").strip().split())[:limit]


def is_complete_fact(item: FactCandidate) -> bool:
    return bool(
        item.subject_user_id
        and item.fact_type
        and item.claim_text
        and item.topic
        and item.evidence_message_id
        and item.evidence_text
    )


def looks_low_value_fact(item: FactCandidate) -> bool:
    text = f"{item.claim_text} {item.topic} {item.evidence_text}"
    if len(item.claim_text) < 6 or len(item.topic) < 2:
        return True
    low_value_markers = (
        "继续聊",
        "随口",
        "发了",
        "发送",
        "分享图片",
        "分享截图",
        "空消息",
        "表情包",
        "聊天",
        "参与讨论",
        "表达情绪",
    )
    return any(marker in text for marker in low_value_markers) and not any(
        signal in text for signal in ("认为", "觉得", "喜欢", "讨厌", "支持", "反对", "评价", "倾向")
    )


def extract_explicit_target_user_ids(context: MessageContext) -> list[str]:
    candidates: list[str] = []
    for mention in context.mentions:
        if mention.is_bot:
            continue
        user_id = _dashboard_user_id(mention.user_id)
        if user_id.isdigit():
            candidates.append(user_id)

    text = context.plain_text
    for match in re.finditer(r"(?i)(?:@?\s*QQ\s*[:：]\s*|qq=)(\d{5,20})", text):
        candidates.append(match.group(1))
    if looks_like_identity_query(text):
        for match in re.finditer(r"(?<!\d)(\d{5,20})(?!\d)", text):
            candidates.append(match.group(1))
    return dedupe_short_strings(candidates)


def extract_identity_name_refs(text: str) -> list[str]:
    if not looks_like_identity_query(text):
        return []
    direct_qq_pattern = re.compile(r"(?i)QQ\s*[:：]\s*\d{5,20}|\d{5,20}")
    refs: list[str] = []
    patterns = (
        re.compile(r"谁是\s*([^?？,，。!！\s@]{1,30})"),
        re.compile(r"([^?？,，。!！\s@]{1,30})\s*是谁"),
        re.compile(r"([^?？,，。!！\s@]{1,30})\s*(?:叫啥|叫什么|叫什麼)"),
        re.compile(r"(?:怎么|如何|要怎么|该怎么)\s*称呼\s*([^?？,，。!！\s@]{1,30})"),
        re.compile(r"(?:名字|昵称|外号|称呼)\s*(?:是|叫|为)\s*([^?？,，。!！\s@]{1,30})"),
    )
    for pattern in patterns:
        for match in pattern.finditer(text):
            raw = match.group(1)
            if direct_qq_pattern.fullmatch(raw.strip()):
                continue
            cleaned = clean_alias(raw)
            if cleaned:
                refs.append(cleaned)
    return dedupe_short_strings(refs)


def looks_like_identity_query(text: str) -> bool:
    return bool(
        re.search(
            r"(谁是|是谁|叫啥|叫什么|叫什麼|怎么称呼|如何称呼|要怎么称呼|该怎么称呼|名字|昵称|外号|称呼)",
            text,
        )
    )


def extract_aliases_from_fact(fact: FactRecord) -> list[tuple[str, str]]:
    if fact.fact_type not in {"identity", "alias"}:
        return []
    candidates = extract_alias_candidates_from_fact(fact)
    return dedupe_alias_pairs(
        (alias, alias_type)
        for alias, alias_type in candidates
        if is_reasonable_member_alias(alias)
    )


def extract_alias_candidates_from_fact(fact: FactCandidate | FactRecord) -> list[tuple[str, str]]:
    if fact.fact_type not in {"identity", "alias"}:
        return []
    texts = (fact.claim_text, fact.evidence_text, fact.topic)
    denied = set(extract_denied_aliases(fact.claim_text, fact.evidence_text))
    aliases: list[tuple[str, str]] = []
    patterns = (
        ("alias", re.compile(r"(?:称呼|昵称|外号|名字)\s*(?:是|叫|为|：|:)?\s*[“\"']?([^，,。；;、\s”\"']{1,30})")),
        ("alias", re.compile(r"(?<!不)(?<!别)(?<!要)(?<!再)(?:叫做|称作|称为|叫)\s*(?:我|他|她|ta|TA)?\s*[“\"']?([^，,。；;、\s”\"']{1,30})")),
        ("identity", re.compile(r"(?:自称|身份是|是)\s*[“\"']?([^，,。；;、\s”\"']{1,30})")),
        ("alias", re.compile(r"(?:改叫|以后叫)\s*[“\"']?([^，,。；;、\s”\"']{1,30})")),
    )
    for alias_type, pattern in patterns:
        for text in texts:
            for match in pattern.finditer(text):
                alias = clean_alias(match.group(1))
                if not alias or alias in denied:
                    continue
                aliases.append((alias, alias_type))
    return dedupe_alias_pairs(aliases)


def is_unreasonable_alias_fact(fact: FactCandidate | FactRecord) -> bool:
    if fact.fact_type not in {"identity", "alias"}:
        return False
    aliases = extract_alias_candidates_from_fact(fact)
    return bool(aliases) and all(not is_reasonable_member_alias(alias) for alias, _ in aliases)


def is_reasonable_member_alias(value: str) -> bool:
    alias = clean_alias(value)
    if not alias:
        return False
    compact = re.sub(r"\s+", "", alias)
    if compact in RELATIONAL_ALIAS_TERMS:
        return False
    relation_core = (
        "主人|主子|老板|老板娘|领导|管理员|管理|群主|版主|"
        "爸爸|爸|爹|父亲|妈妈|妈|母亲|哥哥|哥|姐姐|姐|"
        "弟弟|弟|妹妹|妹|儿子|女儿|老婆|老公|媳妇|丈夫|妻子|"
        "对象|男朋友|女朋友"
    )
    if re.fullmatch(rf"(?:小|老|大|阿)?(?:{relation_core})(?:大人)?", compact):
        return False
    if re.fullmatch(r"(?:第?[一二三四五六七八九十\d]+)?(?:管理员|管理|群主|版主)", compact):
        return False
    return True


def display_name_match_score(ref: str, candidate: str) -> float:
    left = normalize_display_name(ref)
    right = normalize_display_name(candidate)
    if not left or not right:
        return 0.0
    if left == right:
        return 1.0
    if len(left) < 2 or len(right) < 2:
        return 0.0
    if left.startswith(right) or right.startswith(left):
        return 0.88
    if left in right or right in left:
        return 0.82
    common = len(set(left) & set(right))
    coverage = common / max(len(set(left)), 1)
    if len(left) >= 3 and coverage >= 0.75:
        return 0.78
    return 0.0


def select_name_resolution_matches(
    matches: list[NameResolutionMatch],
    limit: int,
) -> list[NameResolutionMatch]:
    if not matches:
        return []
    top_score = matches[0].score
    if top_score < 0.78:
        return []
    close = [match for match in matches if top_score - match.score <= 0.04]
    if len(close) > 1:
        return close[: max(1, int(limit))]
    return matches[:1]


def normalize_display_name(value: str) -> str:
    return re.sub(r"\s+", "", clean_alias(value)).casefold()


def extract_denied_aliases(*texts: str) -> list[str]:
    combined = "\n".join(str(text or "") for text in texts)
    aliases: list[str] = []
    patterns = (
        re.compile(r"(?:不是|不叫|别叫|不要叫|别再叫|不要再叫)\s*(?:我|他|她|ta|TA)?\s*[“\"']?([^，,。；;、\s”\"']{1,30})"),
        re.compile(r"[“\"']?([^，,。；;、\s”\"']{1,30})[”\"']?\s*(?:不是我|不是他|不是她|不是ta|不对)"),
    )
    for pattern in patterns:
        for match in pattern.finditer(combined):
            alias = clean_alias(match.group(1))
            if alias:
                aliases.append(alias)
    return dedupe_short_strings(aliases)


def clean_alias(value: str) -> str:
    alias = " ".join(str(value or "").strip().split())
    alias = alias.strip("「」『』“”\"'`.,，。:：;；!?！？()（）[]【】")
    if not 1 <= len(alias) <= 30:
        return ""
    if re.fullmatch(r"(?i)qq[:：]?\d+", alias):
        return ""
    if alias in {"谁", "你", "我", "他", "她", "它", "ta", "TA", "这个", "那个", "哪位"}:
        return ""
    return alias


def dedupe_alias_pairs(values: Iterable[tuple[str, str]]) -> list[tuple[str, str]]:
    result: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for alias, alias_type in values:
        key = (alias, alias_type)
        if alias and key not in seen:
            result.append(key)
            seen.add(key)
    return result


def dedupe_short_strings(values: Iterable[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        item = " ".join(str(value or "").strip().split())
        if not item or item in seen:
            continue
        result.append(item)
        seen.add(item)
    return result


def fact_importance(item: FactCandidate) -> float:
    base = clamp_float(getattr(item, "importance", 0.5))
    if item.fact_type in {"identity", "boundary"}:
        base = max(base, 0.9)
    elif item.fact_type in {"skill", "habit"}:
        base = max(base, 0.75)
    elif item.fact_type in {"preference", "dislike"}:
        base = max(base, 0.55)
    if str(item.evidence_text or "").startswith("image_index="):
        base = min(base, 0.3)
    return base


def looks_like_self_direct_conflict(new_content: str, old_content: str) -> bool:
    positive_tokens = ("喜欢", "想", "会", "习惯", "可以")
    negative_tokens = ("不喜欢", "讨厌", "怕", "不会", "不太", "不能")
    new_positive = any(token in new_content for token in positive_tokens)
    new_negative = any(token in new_content for token in negative_tokens)
    old_positive = any(token in old_content for token in positive_tokens)
    old_negative = any(token in old_content for token in negative_tokens)
    shared = self_object_terms(new_content) & self_object_terms(old_content)
    return bool(shared and ((new_positive and old_negative) or (new_negative and old_positive)))


def self_object_terms(content: str) -> set[str]:
    cleaned = content
    for token in ("不喜欢", "喜欢", "讨厌", "害怕", "怕", "我", "很", "比较", "一点", "有点"):
        cleaned = cleaned.replace(token, "")
    terms: set[str] = set()
    for phrase in re.findall(r"[\u4e00-\u9fffA-Za-z0-9]{2,}", cleaned):
        terms.add(phrase)
        if len(phrase) <= 12:
            terms.update(phrase[index : index + 2] for index in range(len(phrase) - 1))
    return {term for term in terms if len(term) >= 2}


def merge_summary(current: str, patch: str) -> str:
    return merge_relationship_summary(current, patch)
