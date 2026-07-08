from __future__ import annotations

import time

from nonebot.adapters.onebot.v11 import Bot, GroupMessageEvent

from plugins.llm_group_bot import admin_command_state as _state
from plugins.llm_group_bot import admin_command_utils as _utils
from qq_llm_bot.llm_usage_report import format_llm_token_report


async def handle_stickers(rest: list[str], group_id: str) -> None:
    action = rest[0].lower() if rest else "list"
    if action == "list":
        limit = _utils.parse_memory_id(rest[1]) if len(rest) >= 2 else 20
        stickers = _state.storage.list_stickers(group_id, limit=limit or 20)
        await _state.finish_command(_state.admin_cmd, "\n\n".join(stickers) if stickers else "本群暂无已保存表情包。")
    if len(rest) >= 2 and action in {"enable", "disable"}:
        sticker_id = _utils.parse_memory_id(rest[1])
        if sticker_id is None:
            await _state.finish_command(_state.admin_cmd, "sticker_id 必须是数字，例如：#bot stickers disable 12")
        ok = _state.storage.set_sticker_enabled(sticker_id, action == "enable")
        await _state.finish_command(_state.admin_cmd, "已更新表情状态。" if ok else "没有找到该表情。")
    if len(rest) >= 2 and action in {"delete", "remove", "del", "rm"}:
        sticker_id = _utils.parse_memory_id(rest[1])
        if sticker_id is None:
            await _state.finish_command(_state.admin_cmd, "sticker_id 必须是数字，例如：#bot stickers delete 12")
        asset = _state.storage.delete_sticker_asset(sticker_id)
        if asset is None:
            await _state.finish_command(_state.admin_cmd, "没有找到该表情。")
        deleted_file = _state.sticker_store.delete_saved_file(asset.local_path)
        suffix = "，本地图片也已删除。" if deleted_file else "，但没有找到可删除的本地图片。"
        await _state.finish_command(_state.admin_cmd, f"已删除表情 #{sticker_id}{suffix}")
    await _state.finish_command(_state.admin_cmd, "用法：#bot stickers list [数量]|enable <id>|disable <id>|delete <id>")

async def handle_llm(bot: Bot, event: GroupMessageEvent, rest: list[str]) -> None:
    action = rest[0].lower() if rest else "status"
    if action == "status":
        routing = _state.config.llm.routing
        await _state.finish_command(
            _state.admin_cmd,
            "LLM 模型路由：\n"
            f"路由状态：{'启用' if routing.enabled else '关闭'}\n"
            f"聊天预处理：{_model_display(routing.chat_preprocess_model)}\n"
            f"聊天生成：{_model_display(routing.chat_generation_model)}\n"
            f"最终问答校验：{_model_display(routing.qa_model)}\n"
            f"聊天 FACT 提取：{_model_display(routing.fact_extraction_model)}\n"
            f"根据 FACT 整理认知：{_model_display(routing.cognition_model)}\n"
            f"简单识图：{_model_display(routing.simple_vision_model)}\n"
            f"详细识图：{_model_display(routing.detailed_vision_model)}\n"
            f"图片生成：{_model_display(routing.image_generation_model)}"
        )

    if action == "test":
        prompt = " ".join(rest[1:]).strip() or "用一句话自然地打个招呼。"
        reply = await _state.llm.complete_text(
            "你是 QQ 群里的拟人角色，说话自然、简短。",
            prompt,
            purpose="llm_test",
        )
        await _state.finish_command(_state.admin_cmd, reply or "LLM 没有返回内容，请检查 provider/base_url/model/key。")

    await _state.finish_command(_state.admin_cmd, "用法：#bot _state.llm status|test [prompt]")

async def handle_token_usage() -> None:
    now = int(time.time())
    data = _state.storage.list_dashboard_llm_usage(since=now - 24 * 3600, limit=1)
    await _state.finish_command(_state.admin_cmd, format_llm_token_report(data, hours=24))


def _model_display(model: str) -> str:
    clean_model = model.strip()
    if not clean_model:
        return "未配置"
    _, separator, model_name = clean_model.partition(":")
    return model_name if separator else clean_model
