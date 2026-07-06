from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class DrawRequestPlan:
    original_request: str
    cleaned_draw_request: str
    bot_mention_role: str = "none"
    include_bot_appearance: bool = False
    reference_notes: str = ""


class DrawIntentLLM(Protocol):
    async def complete_text(
        self,
        system_prompt: str,
        user_prompt: str,
        purpose: str = "",
        model_tier: str = "",
    ) -> str | None:
        ...


class DrawIntentPlanner:
    def __init__(
        self,
        llm: DrawIntentLLM,
        *,
        bot_names: tuple[str, ...] = (),
    ) -> None:
        self.llm = llm
        self.bot_names = tuple(name for name in bot_names if str(name).strip())

    async def plan(self, draw_request: str) -> DrawRequestPlan:
        cleaned_fallback = _clean_draw_request_text(draw_request)
        if not cleaned_fallback:
            return DrawRequestPlan(draw_request, "")

        reply = await self.llm.complete_text(
            "你是生图请求意图规划器。只输出 JSON，不要解释。",
            (
                "请规划用户的画图请求：清理开头对机器人的称呼，判断是否真的要画机器人形象，"
                "并保留用户提到的作品、角色、种族、服饰、风格或参考图要求。"
                "重点规则："
                "1. 如果请求开头只是喊机器人昵称，例如“可可，参考...”或“可可 帮我画...”，"
                "bot_mention_role=addressing_bot，cleaned_draw_request 要去掉这个称呼，include_bot_appearance=false。"
                "2. 只有用户明确要求画机器人/可可/bot 的形象、拟人、头像、或和机器人同框时，"
                "bot_mention_role=draw_bot 或 draw_with_bot，include_bot_appearance=true。"
                "3. 不要改写、删除用户明确给出的颜色、妆容、肤色、服饰、构图和参考对象。"
                "4. reference_notes 只简短整理用户显式要求保留的参考关系；不要联网、不要补编外观。"
                "输出 JSON："
                '{"cleaned_draw_request":"去掉机器人称呼后的画图请求",'
                '"bot_mention_role":"none|addressing_bot|draw_bot|draw_with_bot|unclear",'
                '"include_bot_appearance":bool,'
                '"reference_notes":"用户显式参考要求的简短整理，可空"}\n'
                f"机器人昵称：{', '.join(self.bot_names) or '(none)'}\n"
                f"用户请求：{cleaned_fallback}"
            ),
            purpose="draw_intent",
        )
        data = _extract_json_object(reply)
        if not data:
            return DrawRequestPlan(
                original_request=draw_request,
                cleaned_draw_request=cleaned_fallback,
                bot_mention_role="unknown",
                include_bot_appearance=False,
            )

        cleaned = _clean_draw_request_text(str(data.get("cleaned_draw_request", "") or cleaned_fallback))
        return DrawRequestPlan(
            original_request=draw_request,
            cleaned_draw_request=cleaned or cleaned_fallback,
            bot_mention_role=_safe_bot_mention_role(str(data.get("bot_mention_role", "none") or "none")),
            include_bot_appearance=_as_bool(data.get("include_bot_appearance"), False),
            reference_notes=_clean_draw_request_text(str(data.get("reference_notes", "") or "")),
        )


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


def _safe_bot_mention_role(value: str) -> str:
    normalized = str(value or "").strip().lower()
    allowed = {"none", "addressing_bot", "draw_bot", "draw_with_bot", "unclear", "unknown"}
    return normalized if normalized in allowed else "none"


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


def _clean_draw_request_text(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()[:500]
