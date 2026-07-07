from __future__ import annotations

from qq_llm_bot.directness import looks_like_bot_address, text_mentions_bot_name


def test_nickname_call_patterns_are_direct() -> None:
    nicknames = ["可可"]

    assert looks_like_bot_address("可可，帮我看下这个", nicknames)
    assert looks_like_bot_address("可可你觉得呢", nicknames)
    assert looks_like_bot_address("这个怎么处理？可可", nicknames)


def test_object_mentions_are_not_direct_calls() -> None:
    nicknames = ["可可"]

    assert text_mentions_bot_name("顺便一提可可的形象是固定的", nicknames)
    assert not looks_like_bot_address("顺便一提可可的形象是固定的", nicknames)
    assert not looks_like_bot_address("让可可给你cos一个", nicknames)
    assert not looks_like_bot_address("你也好可可这口是吧", nicknames)
    assert not looks_like_bot_address("噢对，我好像限制可可讨论政治话题了", nicknames)
