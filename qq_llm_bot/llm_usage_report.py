from __future__ import annotations

from collections.abc import Mapping

LLM_PURPOSE_LABELS = {
    "(unspecified)": "未指定功能",
    "answer_question": "问题回答",
    "batch_observation": "批量旁观总结",
    "draw_intent": "生图意图判断",
    "draw_prompt": "生图提示词整理",
    "fact_extract": "成员事实提取",
    "final_qa": "最终质检",
    "followup_gate": "追聊判断",
    "lexicon_detect": "群词识别",
    "lexicon_summarize": "群词总结",
    "llm_test": "LLM 测试",
    "memory_curator": "记忆整理",
    "perception": "消息理解",
    "profile_aggregate": "成员画像汇总",
    "reflection": "复盘反思",
    "relationship": "关系更新",
    "response": "最终回复",
    "self_claim_rewrite": "自称改写",
    "self_narrative_check": "自我叙述校验",
    "self_narrative_draft": "自我叙述草稿",
    "self_narrative_plan": "自我叙述规划",
    "sticker_select": "表情选择",
    "structured_json": "结构化解析",
    "text": "文本补全",
    "topic_join": "话题加入",
    "participation_policy": "参与判断",
    "vision": "图片理解",
}


def format_llm_token_report(
    data: Mapping[str, object],
    *,
    hours: int = 24,
    feature_limit: int = 3,
) -> str:
    summary = data.get("summary")
    summary_data: Mapping[str, object] = summary if isinstance(summary, Mapping) else {}
    total_tokens = _int_value(summary_data.get("total_tokens"))
    calls = _int_value(summary_data.get("calls"))
    prompt_tokens = _int_value(summary_data.get("prompt_tokens"))
    completion_tokens = _int_value(summary_data.get("completion_tokens"))

    rows = _usage_rows(data)
    by_model = _aggregate_usage(rows, "model")
    by_purpose = _aggregate_usage(rows, "purpose")
    feature_limit = max(1, int(feature_limit))

    lines = [
        f"过去 {max(1, int(hours))} 小时 token 消耗：",
        (
            f"总计：{_format_integer(total_tokens)} token"
            f"（调用 {_format_integer(calls)} 次，"
            f"prompt {_format_integer(prompt_tokens)}，"
            f"completion {_format_integer(completion_tokens)}）"
        ),
        "按模型：",
        *_format_usage_lines(by_model, total_tokens),
        f"消耗量 Top {feature_limit} 功能：",
        *_format_usage_lines(by_purpose[:feature_limit], total_tokens),
    ]
    if calls > 0 and total_tokens == 0:
        lines.append("提示：有调用记录但 token 为 0，可能是 provider 未返回 usage。")
    return "\n".join(lines)


def _usage_rows(data: Mapping[str, object]) -> list[Mapping[str, object]]:
    rows = data.get("by_purpose")
    if not isinstance(rows, list):
        return []
    return [row for row in rows if isinstance(row, Mapping)]


def _aggregate_usage(
    rows: list[Mapping[str, object]],
    key: str,
) -> list[tuple[str, int]]:
    totals: dict[str, int] = {}
    for row in rows:
        tokens = _int_value(row.get("total_tokens"))
        if tokens <= 0:
            continue
        raw_label = str(row.get(key) or "").strip()
        label = _model_label(raw_label) if key == "model" else _purpose_label(raw_label)
        totals[label] = totals.get(label, 0) + tokens
    return sorted(totals.items(), key=lambda item: (-item[1], item[0]))


def _format_usage_lines(
    rows: list[tuple[str, int]],
    total_tokens: int,
) -> list[str]:
    if not rows:
        return ["- 暂无非零 token 记录"]
    return [
        f"- {label}：{_format_integer(tokens)} token（占比 {_format_percent(tokens, total_tokens)}）"
        for label, tokens in rows
    ]


def _purpose_label(purpose: str) -> str:
    normalized = purpose.strip()
    if not normalized:
        return "未指定功能"
    return LLM_PURPOSE_LABELS.get(normalized, f"未知功能（{normalized}）")


def _model_label(model: str) -> str:
    return model.strip() or "未指定模型"


def _format_integer(value: int) -> str:
    return f"{value:,}"


def _format_percent(value: int, total: int) -> str:
    if total <= 0:
        return "0.0%"
    return f"{value * 100 / total:.1f}%"


def _int_value(value: object) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0
