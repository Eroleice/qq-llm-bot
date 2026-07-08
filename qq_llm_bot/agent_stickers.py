from __future__ import annotations

import time
from typing import Any

from qq_llm_bot.agent_formatters import (
    format_recent_context as _format_recent_context,
    format_semantic_context as _format_semantic_context,
    format_sticker_assets as _format_sticker_assets,
)
from qq_llm_bot.config import AppConfig
from qq_llm_bot.llm import LLMClient
from qq_llm_bot.llm_json_helpers import complete_json as _complete_json
from qq_llm_bot.models import (
    ConversationSnapshot,
    MessageContext,
    ParticipationDecision,
    StickerAssetRecord,
)


class StickerSelectorAgent:
    def __init__(self, config: AppConfig, llm: LLMClient) -> None:
        self.config = config
        self.llm = llm
        self._last_sent_at: dict[str, int] = {}

    async def select(
        self,
        context: MessageContext,
        decision: ParticipationDecision,
        snapshot: ConversationSnapshot,
        reply: str | None,
    ) -> StickerAssetRecord | None:
        if not self.config.stickers.enabled:
            return None
        if decision.action == "observe" or not reply:
            return None
        if not snapshot.sticker_assets:
            return None
        if self._cooldown_active(context.group_id):
            return None

        fallback = self._heuristic_select(context, reply, snapshot.sticker_assets)
        data = await _complete_json(
            self.llm,
            "你是 QQ 群拟人角色的表情包选择器。只输出 JSON，不要解释。",
            (
                "判断这次回复是否适合附带一个表情包。"
                "只有明显契合语气和上下文时才选；严肃、敏感、争吵、安慰过重时不要发。"
                "asset_id 为 0 表示不发。"
                "输出 JSON："
                '{"asset_id":0,"confidence":0.0,"reason":"短原因"}\n'
                f"第一阶段语义上下文：\n{_format_semantic_context(snapshot.semantic_context)}\n"
                f"最近群聊：\n{_format_recent_context(snapshot)}\n"
                f"对方消息：{context.plain_text}\n"
                f"机器人文字回复：{reply}\n"
                f"可用表情：\n{_format_sticker_assets(snapshot.sticker_assets)}"
            ),
            purpose="sticker_select",
        )
        selected = self._asset_from_json(data, snapshot.sticker_assets) if data else fallback
        if selected is None:
            return None
        self._last_sent_at[context.group_id] = int(time.time())
        return selected

    def _cooldown_active(self, group_id: str) -> bool:
        last = self._last_sent_at.get(group_id, 0)
        return int(time.time()) - last < self.config.stickers.send_cooldown_seconds

    def _asset_from_json(
        self,
        data: dict[str, Any] | None,
        assets: list[StickerAssetRecord],
    ) -> StickerAssetRecord | None:
        if not data:
            return None
        confidence = _clamp_float(data.get("confidence", 0.0))
        if confidence < self.config.stickers.selection_threshold:
            return None
        try:
            asset_id = int(data.get("asset_id", 0))
        except (TypeError, ValueError):
            return None
        if asset_id <= 0:
            return None
        by_id = {asset.id: asset for asset in assets}
        return by_id.get(asset_id)

    def _heuristic_select(
        self,
        context: MessageContext,
        reply: str,
        assets: list[StickerAssetRecord],
    ) -> StickerAssetRecord | None:
        text = f"{context.plain_text}\n{reply}"
        if any(token in text for token in ("哈哈", "笑死", "乐", "好好笑")):
            return _first_matching_sticker(assets, ("笑", "好笑", "乐", "开心"))
        if any(token in text for token in ("离谱", "震惊", "啊？", "啊?", "真的假的")):
            return _first_matching_sticker(assets, ("震惊", "惊讶", "离谱"))
        if any(token in text for token in ("无语", "沉默", "尴尬")):
            return _first_matching_sticker(assets, ("无语", "沉默", "尴尬"))
        if any(token in text for token in ("困", "累", "下班", "不想动")):
            return _first_matching_sticker(assets, ("困", "累", "疲惫", "下班"))
        return None



def _first_matching_sticker(
    assets: list[StickerAssetRecord],
    tokens: tuple[str, ...],
) -> StickerAssetRecord | None:
    for asset in assets:
        haystack = " ".join([asset.mood, asset.usage, asset.description, *asset.tags])
        if any(token in haystack for token in tokens):
            return asset
    return None



def _clamp_float(value: object) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, numeric))
