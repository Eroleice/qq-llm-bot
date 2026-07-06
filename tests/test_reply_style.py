from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from qq_llm_bot.cognitive_storage import BotStorage
from qq_llm_bot.config import AppConfig, BotConfig, LLMConfig, NapCatConfig, StorageConfig, load_config
from qq_llm_bot.reply_style import ReplyStyleSettings, split_reply_bubbles, style_reply_text


def test_short_reply_collapses_newlines_and_drops_period() -> None:
    settings = ReplyStyleSettings()

    text = style_reply_text(
        "懂了。\n\n我先记一下。",
        settings,
        action="reply",
        value_type="direct_reply",
    )

    assert text == "懂了。 我先记一下"


def test_short_reply_keeps_question_exclaim_and_emoji() -> None:
    settings = ReplyStyleSettings()

    assert style_reply_text("真的假的？", settings) == "真的假的？"
    assert style_reply_text("好耶！", settings) == "好耶！"
    assert style_reply_text("可以😌", settings) == "可以😌"


def test_disabled_style_keeps_single_reply_shape() -> None:
    settings = ReplyStyleSettings(enabled=False)

    assert style_reply_text("懂了。", settings) == "懂了。"
    assert split_reply_bubbles("懂了。", settings) == ("懂了。",)


def test_direct_reply_compacts_when_not_explanation() -> None:
    settings = ReplyStyleSettings()

    text = style_reply_text(
        "别急别急，trust不是十连抽出来的啦。稳定好相处、不越界、有事说清楚，慢慢就涨了。",
        settings,
        action="reply",
        value_type="direct_reply",
        trigger_text="看看你跟我的好感度",
    )

    assert len(text) <= 25
    assert text.startswith("别急别急")
    assert "trust" in text


def test_explanation_reply_can_split_into_multiple_bubbles() -> None:
    settings = ReplyStyleSettings(bubble_trigger_chars=45, bubble_target_chars=28, bubble_max_parts=3)
    text = style_reply_text(
        "可以做成画图前置链路：先判断需不需要搜，需要就搜关键词和风格参考，再压成一小段 prompt，最后丢给 image2.0。",
        settings,
        action="proactive_reply",
        value_type="useful_context",
        trigger_text="怎么能让机器人画图之前有联网搜索能力",
    )

    parts = split_reply_bubbles(text, settings)

    assert 2 <= len(parts) <= 3
    assert "".join(parts).replace(" ", "") == text.rstrip("。").replace(" ", "")
    assert not parts[-1].endswith("。")


def test_proactive_reply_compacts_to_short_interjection() -> None:
    settings = ReplyStyleSettings()

    text = style_reply_text(
        "这个角度有点绕，但核心就是先把主语补上，不然大家会各聊各的。",
        settings,
        action="proactive_reply",
        value_type="clarifying_question",
        trigger_text="有啥成名作",
    )

    assert len(text) <= 30


def test_emoji_cooldown_removes_recent_emoji_and_extra_emoji() -> None:
    settings = ReplyStyleSettings(emoji_cooldown_messages=10)

    text = style_reply_text(
        "这个可以😌😼",
        settings,
        recent_bot_replies=["上一条用了😌"],
    )

    assert text == "这个可以😼"


def test_sticker_reply_gets_short_or_empty_text() -> None:
    settings = ReplyStyleSettings()

    empty_text = style_reply_text("哈哈哈😌", settings, has_sticker=True)
    short_text = style_reply_text("这个方案真的挺靠谱，可以先这么走。", settings, has_sticker=True)

    assert empty_text == ""
    assert len(short_text) <= 20


def test_split_does_not_break_mentions_urls_or_cq_segments() -> None:
    settings = ReplyStyleSettings(bubble_trigger_chars=12, bubble_target_chars=18, bubble_max_parts=3)
    text = "可以问 @QQ:123456789 这条 https://example.test/a/b?c=1 先别拆坏。[CQ:face,id=14]"

    parts = split_reply_bubbles(text, settings)
    joined = "".join(parts)

    assert "@QQ:123456789" in joined
    assert "https://example.test/a/b?c=1" in joined
    assert "[CQ:face,id=14]" in joined


def test_reply_style_config_defaults_and_validation(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text('[napcat]\nws_url = "ws://example.test"\n', encoding="utf-8")

    config = load_config(config_path)

    assert config.bot.reply_style_enabled is True
    assert config.bot.reply_bubbles_enabled is True
    assert config.bot.reply_bubble_max_parts == 3
    assert config.bot.reply_bubble_delay_seconds == 0.9

    bad_config_path = tmp_path / "bad-config.toml"
    bad_config_path.write_text(
        "\n".join(
            [
                "[napcat]",
                'ws_url = "ws://example.test"',
                "",
                "[bot]",
                "reply_bubble_max_parts = 4",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="bot.reply_bubble_max_parts"):
        load_config(bad_config_path)


def test_storage_records_bot_reply_parts_and_recent_texts(tmp_path: Path) -> None:
    db_path = tmp_path / "bot.sqlite3"
    config = AppConfig(
        napcat=NapCatConfig(ws_url="ws://example.test"),
        bot=BotConfig(),
        storage=StorageConfig(sqlite_path=str(db_path)),
        llm=LLMConfig(provider="disabled"),
        project_root=tmp_path,
    )
    storage = BotStorage.from_config(config)
    storage.setup()

    storage.record_bot_reply_parts("100", "999", ["第一条", "第二条"])

    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            "SELECT plain_text, sender_role FROM messages ORDER BY id",
        ).fetchall()

    assert rows == [("第一条", "bot"), ("第二条", "bot")]
    assert storage.get_recent_bot_reply_texts("100", 2) == ["第二条", "第一条"]


def test_plugin_source_keeps_bubble_send_contract() -> None:
    source = (Path(__file__).resolve().parents[1] / "plugins" / "llm_group_bot" / "__init__.py").read_text(
        encoding="utf-8"
    )

    assert "allow_bubbles=not bool(conflict_reply)" in source
    assert "for index, part in enumerate(parts[:-1])" in source
    assert "last_reply_to = None" in source
    assert "storage.record_bot_reply_parts" in source


def test_plugin_reply_message_only_adds_newline_before_image() -> None:
    source = (Path(__file__).resolve().parents[1] / "plugins" / "llm_group_bot" / "__init__.py").read_text(
        encoding="utf-8"
    )

    assert 'if file_ref:\n            message += MessageSegment.text("\\n")' in source
