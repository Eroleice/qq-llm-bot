from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Protocol

from loguru import logger

from qq_llm_bot.web_search import SearchResult


@dataclass(frozen=True)
class DrawReferenceCandidate:
    work_name: str
    character_name: str
    query: str
    confidence: float
    reason: str = ""


@dataclass(frozen=True)
class DrawResolvedReference:
    candidate: DrawReferenceCandidate
    status: str
    visual_summary: str = ""
    source_urls: tuple[str, ...] = ()
    confidence: float = 0.0
    cached_at: int = 0


class DrawReferenceLLM(Protocol):
    async def complete_text(
        self,
        system_prompt: str,
        user_prompt: str,
        purpose: str = "",
        model_tier: str = "",
    ) -> str | None:
        ...


class DrawReferenceSearch(Protocol):
    async def search(self, query: str, max_results: int | None = None) -> list[SearchResult]:
        ...


_NAME_TOKEN = r"[\u4e00-\u9fffA-Za-z0-9][\u4e00-\u9fffA-Za-z0-9_·・:：.\-]{0,29}"
_CHAR_TOKEN = r"[\u4e00-\u9fffA-Za-z0-9][\u4e00-\u9fffA-Za-z0-9_·・:：.\-]{0,23}"
_REF_PREFIX = r"(?P<ref_prefix>参考|参照|照着|像|按|来自|以|借鉴|基于|cos|COS|同款|二创)?"
_WORK_PART = rf"(?:《(?P<work_quoted>[^》]{{1,30}})》|(?P<work>{_NAME_TOKEN}))"
_CHAR_PART = (
    rf"(?:「(?P<char_corner>[^」]{{1,24}})」|"
    rf"“(?P<char_quote>[^”]{{1,24}})”|(?P<char>{_CHAR_TOKEN}))"
)

_WORK_IN_CHARACTER_PATTERNS = (
    re.compile(
        rf"{_REF_PREFIX}\s*{_WORK_PART}\s*"
        rf"(?:中|里|中的|里的|里面的)\s*"
        rf"(?:角色|人物|女主|男主|角色名|人物名)?\s*"
        rf"(?:叫|名叫|名为)?\s*{_CHAR_PART}",
        re.I,
    ),
    re.compile(
        rf"(?:游戏|手游|动漫|动画|番剧|作品)\s*{_WORK_PART}\s*"
        rf"(?:的|中|里|中的|里的)?\s*"
        rf"(?:角色|人物|女主|男主|角色名|人物名)\s*"
        rf"(?:叫|名叫|名为|是)?\s*{_CHAR_PART}",
        re.I,
    ),
    re.compile(
        rf"{_REF_PREFIX}\s*(?:来自|出自)\s*{_WORK_PART}\s*(?:的)?\s*"
        rf"(?:角色|人物|女主|男主)?\s*{_CHAR_PART}",
        re.I,
    ),
)

_QUOTED_CHARACTER_ONLY_PATTERN = re.compile(
    rf"(?:参考|参照|照着|像|按|借鉴|基于|cos|COS|二创)\s*"
    rf"(?:角色|人物)?\s*{_CHAR_PART}\s*"
    rf"(?:这个|这一)?(?:角色|人物|人设|设定|外观|同款|cos|COS|二创)",
    re.I,
)

_BAD_NAME_VALUES = {
    "角色",
    "人物",
    "女主",
    "男主",
    "一个",
    "一位",
    "这个",
    "那个",
    "龙娘",
    "女孩",
    "男孩",
    "少女",
    "少年",
}

_REFERENCE_CUE_PATTERN = re.compile(
    r"(参考|参照|照着|像|按|借鉴|基于|二创|同人|cos|COS|同款|角色|人物|设定|外观|立绘|"
    r"游戏|手游|动漫|动画|番剧|作品)"
)


class DrawReferenceResolver:
    def __init__(
        self,
        llm: DrawReferenceLLM,
        web_search: DrawReferenceSearch,
        *,
        max_results: int = 5,
        cache_ttl_seconds: int = 6 * 60 * 60,
        min_search_confidence: float = 0.7,
    ) -> None:
        self.llm = llm
        self.web_search = web_search
        self.max_results = max(1, max_results)
        self.cache_ttl_seconds = max(60, cache_ttl_seconds)
        self.min_search_confidence = min_search_confidence
        self._cache: dict[str, DrawResolvedReference] = {}

    async def resolve(self, draw_request: str) -> DrawResolvedReference | None:
        candidate = detect_draw_reference(draw_request)
        if candidate is None:
            return None

        key = draw_reference_cache_key(candidate)
        cached = self._cache.get(key)
        now = int(time.time())
        if cached and now - cached.cached_at <= self.cache_ttl_seconds:
            return cached

        results: list[SearchResult] = []
        if candidate.confidence >= self.min_search_confidence:
            try:
                results = await self.web_search.search(candidate.query, self.max_results)
            except Exception as exc:  # pragma: no cover - third-party search boundary
                logger.warning("Draw reference search failed: {}", exc)
                results = []

        resolved = await self._resolve_from_results(candidate, results, now)
        self._cache[key] = resolved
        return resolved

    async def _resolve_from_results(
        self,
        candidate: DrawReferenceCandidate,
        results: list[SearchResult],
        now: int,
    ) -> DrawResolvedReference:
        source_urls = tuple(_dedupe_urls(result.url for result in results))
        if not results:
            return DrawResolvedReference(
                candidate=candidate,
                status="unresolved",
                source_urls=source_urls,
                cached_at=now,
            )

        summary, confidence = await _summarize_reference(
            self.llm,
            candidate,
            results,
        )
        if not summary:
            return DrawResolvedReference(
                candidate=candidate,
                status="unresolved",
                source_urls=source_urls,
                cached_at=now,
            )

        return DrawResolvedReference(
            candidate=candidate,
            status="resolved",
            visual_summary=summary,
            source_urls=source_urls,
            confidence=confidence,
            cached_at=now,
        )


def detect_draw_reference(draw_request: str) -> DrawReferenceCandidate | None:
    text = " ".join(str(draw_request or "").split())
    if not text:
        return None

    candidates: list[DrawReferenceCandidate] = []
    for index, pattern in enumerate(_WORK_IN_CHARACTER_PATTERNS):
        for match in pattern.finditer(text):
            candidate = _candidate_from_match(
                match,
                confidence=0.92,
                reason="work_character",
                full_text=text,
                require_reference_cue=index == 0,
            )
            if candidate:
                candidates.append(candidate)

    for match in _QUOTED_CHARACTER_ONLY_PATTERN.finditer(text):
        candidate = _candidate_from_match(match, confidence=0.72, reason="character_only")
        if candidate:
            candidates.append(candidate)

    if not candidates:
        return None
    candidates.sort(key=lambda item: item.confidence, reverse=True)
    return candidates[0]


def draw_reference_cache_key(candidate: DrawReferenceCandidate) -> str:
    return (
        f"work:{_normalize_cache_part(candidate.work_name)}|"
        f"character:{_normalize_cache_part(candidate.character_name)}"
    )


def format_draw_reference_context(
    reference: DrawResolvedReference | None,
    *,
    has_reference_image_context: bool = False,
) -> str:
    if reference is None:
        return "(none)"

    candidate = reference.candidate
    work = f"《{candidate.work_name}》" if candidate.work_name else "未指定作品"
    character = f"「{candidate.character_name}」"
    lines = []
    if has_reference_image_context:
        lines.append(
            "用户本次请求附近包含图片理解结果；如果图片与请求相关，优先使用图片视觉内容，"
            "联网参考只作补充。"
        )

    if reference.status == "resolved" and reference.visual_summary:
        lines.extend(
            [
                f"已查证参考：{work}角色{character}。",
                f"联网查询：{candidate.query}",
                f"可画视觉事实：{reference.visual_summary}",
                "使用方式：只把这些已查证外观特征当作二创锚点；不要 1:1 复刻，"
                "也不要补充来源没有支持的服装、武器或发色。",
            ]
        )
    else:
        lines.extend(
            [
                f"疑似专名参考：{work}角色{character}，但未取得可靠外观摘要。",
                f"尝试查询：{candidate.query}",
                "处理要求：不要把角色名按字面解释成颜色或普通词，"
                "不要编造该角色的发色、服装、武器、性格或标志物；"
                "如果仍要出图，只生成更泛化的原创版本，并保留“参考未确认”的不确定性。",
            ]
        )

    if reference.source_urls:
        lines.append("来源URL：" + "；".join(reference.source_urls[:3]))
    return "\n".join(lines)


def _candidate_from_match(
    match: re.Match[str],
    *,
    confidence: float,
    reason: str,
    full_text: str = "",
    require_reference_cue: bool = False,
) -> DrawReferenceCandidate | None:
    if require_reference_cue and not _match_has_reference_cue(match, full_text):
        return None
    work = _clean_work_name(match.groupdict().get("work_quoted") or match.groupdict().get("work") or "")
    character = _clean_character_name(
        match.groupdict().get("char_corner")
        or match.groupdict().get("char_quote")
        or match.groupdict().get("char")
        or ""
    )
    if not _valid_reference_names(work, character):
        return None
    query = build_draw_reference_query(work, character)
    return DrawReferenceCandidate(
        work_name=work,
        character_name=character,
        query=query,
        confidence=confidence,
        reason=reason,
    )


def _match_has_reference_cue(match: re.Match[str], full_text: str) -> bool:
    groups = match.groupdict()
    if groups.get("ref_prefix") or groups.get("work_quoted"):
        return True
    return bool(_REFERENCE_CUE_PATTERN.search(full_text))


def build_draw_reference_query(work_name: str, character_name: str) -> str:
    parts = [work_name, character_name, "角色", "外观", "设定"]
    return " ".join(part for part in parts if part)


async def _summarize_reference(
    llm: DrawReferenceLLM,
    candidate: DrawReferenceCandidate,
    results: list[SearchResult],
) -> tuple[str, float]:
    text = await llm.complete_text(
        "你是生图参考资料提取器。只输出 JSON，不要解释。",
        (
            "根据搜索结果判断用户请求的作品/角色引用是否匹配，并提取可画的视觉事实。"
            "只能依据搜索结果里的标题和摘要；不要补充来源没有明确支持的外观。"
            "visual_summary 只写发型、服装、配色、体态、标志物等可画信息，"
            "不写剧情评价或无关背景。没有可靠外观就 is_match=false。"
            "输出 JSON："
            '{"is_match":bool,"visual_summary":"可画视觉摘要","confidence":0.0}\n'
            f"作品：{candidate.work_name or '(unknown)'}\n"
            f"角色：{candidate.character_name}\n"
            f"搜索查询：{candidate.query}\n"
            f"搜索结果：\n{_format_reference_search_results(results)}"
        ),
        purpose="draw_reference",
    )
    data = _extract_json_object(text)
    if not data:
        return "", 0.0
    if not _as_bool(data.get("is_match"), False):
        return "", 0.0
    summary = _clean_visual_summary(str(data.get("visual_summary", "")))
    confidence = _clamp_float(data.get("confidence", 0.0))
    if not summary or confidence < 0.55:
        return "", 0.0
    return summary, confidence


def _format_reference_search_results(results: list[SearchResult]) -> str:
    lines = []
    for index, result in enumerate(results[:5], start=1):
        title = " ".join(result.title.split())[:120]
        url = result.url.strip()
        snippet = " ".join(result.snippet.split())[:240]
        lines.append(f"{index}. {title}\nURL: {url}\n摘要: {snippet or '(empty)'}")
    return "\n".join(lines) if lines else "(none)"


def _extract_json_object(text: str | None) -> dict[str, object] | None:
    if not text:
        return None
    raw = text.strip()
    if raw.startswith("```"):
        raw = raw.strip("`").strip()
        if raw.lower().startswith("json"):
            raw = raw[4:].strip()
    start = raw.find("{")
    end = raw.rfind("}")
    if start >= 0 and end > start:
        raw = raw[start : end + 1]
    try:
        data = json.loads(raw)
    except ValueError:
        return None
    return data if isinstance(data, dict) else None


def _clean_work_name(value: str) -> str:
    text = _clean_name(value)
    text = re.sub(r"^(?:参考|参照|照着|像|按|来自|以|借鉴|基于)", "", text, flags=re.I)
    return _clean_name(text)


def _clean_character_name(value: str) -> str:
    text = _clean_name(value)
    text = re.sub(r"^(?:角色|人物|女主|男主|叫|名叫|名为|是|的)", "", text, flags=re.I)
    text = re.split(
        r"(?:的(?:二创|同人|设定|外观|风格|同款)?|，|,|。|！|!|？|\?|画|绘|做成|变成|改成)",
        text,
        maxsplit=1,
    )[0]
    return _clean_name(text)


def _clean_name(value: str) -> str:
    text = str(value or "").strip()
    text = text.strip("「」“”\"'《》[]（）()【】")
    return " ".join(text.split())[:30]


def _valid_reference_names(work: str, character: str) -> bool:
    if not character or character in _BAD_NAME_VALUES:
        return False
    if work and work == character:
        return False
    if work in _BAD_NAME_VALUES:
        return False
    if character.endswith("色") and len(character) <= 4 and not work:
        return False
    return True


def _normalize_cache_part(value: str) -> str:
    return re.sub(r"\s+", "", str(value or "").strip().lower())


def _dedupe_urls(urls: object) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for raw in urls:
        url = str(raw or "").strip()
        if not url or url in seen:
            continue
        seen.add(url)
        deduped.append(url)
    return deduped


def _as_bool(value: object, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes", "1", "是"}:
            return True
        if lowered in {"false", "no", "0", "否"}:
            return False
    return default


def _clamp_float(value: object) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, parsed))


def _clean_visual_summary(value: str) -> str:
    text = " ".join(str(value or "").split())
    return text[:240]
