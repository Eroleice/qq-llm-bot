from __future__ import annotations

import unittest

from qq_llm_bot.llm_usage_report import format_llm_token_report


class LLMUsageReportTests(unittest.TestCase):
    def test_format_llm_token_report_groups_models_and_chinese_features(self) -> None:
        report = format_llm_token_report(
            {
                "summary": {
                    "calls": 5,
                    "prompt_tokens": 100,
                    "completion_tokens": 40,
                    "total_tokens": 140,
                },
                "by_purpose": [
                    {"purpose": "response", "model": "pro:gpt-5.5", "total_tokens": 60},
                    {"purpose": "perception", "model": "pro:gpt-5.4-mini", "total_tokens": 30},
                    {"purpose": "response", "model": "gpt-5.4-mini", "total_tokens": 20},
                    {"purpose": "draw_prompt", "model": "backup:gpt-5.5", "total_tokens": 30},
                    {"purpose": "vision", "model": "pro:gpt-5.5", "total_tokens": 0},
                ],
            }
        )

        self.assertIn("过去 24 小时 token 消耗：", report)
        self.assertIn("总计：140 token（调用 5 次，prompt 100，completion 40）", report)
        self.assertIn("- gpt-5.5：90 token（占比 64.3%）", report)
        self.assertIn("- gpt-5.4-mini：50 token（占比 35.7%）", report)
        self.assertNotIn("pro:", report)
        self.assertNotIn("backup:", report)
        self.assertIn("- 最终回复：80 token（占比 57.1%）", report)
        self.assertIn("- 生图提示词整理：30 token（占比 21.4%）", report)
        self.assertIn("- 消息理解：30 token（占比 21.4%）", report)
        self.assertNotIn("图片理解", report)

    def test_format_llm_token_report_notes_zero_token_calls(self) -> None:
        report = format_llm_token_report(
            {
                "summary": {
                    "calls": 2,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                },
                "by_purpose": [
                    {"purpose": "response", "model": "gpt-5.5", "total_tokens": 0},
                ],
            }
        )

        self.assertIn("总计：0 token（调用 2 次，prompt 0，completion 0）", report)
        self.assertIn("- 暂无非零 token 记录", report)
        self.assertIn("provider 未返回 usage", report)
