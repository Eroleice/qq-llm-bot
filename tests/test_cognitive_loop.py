from __future__ import annotations

import json
import unittest
from dataclasses import replace
from pathlib import Path

from nonebot.adapters.onebot.v11 import Message

from qq_llm_bot.cognitive_agents import (
    AgentPipeline,
    BatchObservationAgent,
    ContextUnderstandingAgent,
    FactExtractorAgent,
    FinalQAAgent,
    MemoryCuratorAgent,
    ParticipationPolicyAgent,
    RelationshipAgent,
    ResponseAgent,
    StickerSelectorAgent,
    VisionAgent,
    _sanitize_reply,
)
from qq_llm_bot.cognitive_storage import BotStorage
from qq_llm_bot.config import (
    LexiconConfig,
    StickerConfig,
    VisionConfig,
)
from qq_llm_bot.models import (
    ConversationSnapshot,
    FactRecord,
    MemoryRecord,
    MessageAttachment,
    MessageContext,
    MessageMention,
    PerceptionResult,
    ParticipationDecision,
    StickerAssetRecord,
    TargetUserContext,
)
from qq_llm_bot.onebot_messages import (
    FORWARDED_RECORD_END,
    FORWARDED_RECORD_START,
    QUOTED_MESSAGE_END,
    QUOTED_MESSAGE_START,
    parse_outgoing_mention_parts,
    render_message_text_and_mentions,
    render_message_text_and_mentions_with_forwards,
    strip_forwarded_records,
    strip_quoted_messages,
)
from qq_llm_bot.onebot_context import (
    build_message_context,
    image_attachments_from_message_with_replies,
)
from qq_llm_bot.observation_batching import select_observation_batch_size
from qq_llm_bot.realtime_merge import (
    merge_realtime_contexts,
    split_image_descriptions_by_context,
)
from qq_llm_bot.web_search import SearchResult
from tests.helpers import (
    FakeLLM,
    FakeSearch,
    project_temp_directory,
    test_config,
)


class CognitiveLoopTests(unittest.IsolatedAsyncioTestCase):
    async def test_build_context_preserves_nickname_stripped_by_adapter(self) -> None:
        original_text = "\u53ef\u53ef\u4f60\u8d77\u5e8a\u4e86\u4e48"
        stripped_text = "\u4f60\u8d77\u5e8a\u4e86\u4e48"

        class FakeBot:
            self_id = "999"

        class FakeEvent:
            group_id = "100"
            user_id = "42"
            message_id = "m1"
            time = 123
            original_message = Message(original_text)
            message = Message(stripped_text)
            raw_message = original_text
            sender = {"nickname": "Alice", "card": "", "role": "member"}
            to_me = True
            reply = None

        context = await build_message_context(
            FakeBot(),
            FakeEvent(),
            bot_names=("\u53ef\u53ef",),
        )

        self.assertEqual(context.plain_text, original_text)
        self.assertEqual(context.raw_message, original_text)
        self.assertTrue(context.is_direct)
        self.assertTrue(context.bot_mentioned)

    async def test_build_context_uses_strict_directness_when_original_message_exists(self) -> None:
        original_text = "\u53ef\u53ef\u7684\u5f62\u8c61\u662f\u56fa\u5b9a\u7684"
        stripped_text = "\u7684\u5f62\u8c61\u662f\u56fa\u5b9a\u7684"

        class FakeBot:
            self_id = "999"

        class FakeEvent:
            group_id = "100"
            user_id = "42"
            message_id = "m1"
            time = 123
            original_message = Message(original_text)
            message = Message(stripped_text)
            raw_message = original_text
            sender = {"nickname": "Alice", "card": "", "role": "member"}
            to_me = True
            reply = None

        context = await build_message_context(
            FakeBot(),
            FakeEvent(),
            bot_names=("\u53ef\u53ef",),
        )

        self.assertEqual(context.plain_text, original_text)
        self.assertFalse(context.is_direct)
        self.assertTrue(context.bot_mentioned)

    def test_realtime_merge_context_uses_latest_message_and_combines_context(self) -> None:
        first = MessageContext(
            group_id="100",
            user_id="42",
            message_id="m1",
            plain_text="@coco can you explain",
            raw_message="@coco can you explain",
            is_direct=True,
            bot_mentioned=True,
            timestamp=1,
            attachments=[
                MessageAttachment(
                    attachment_type="image",
                    url="https://example.test/first.png",
                )
            ],
            mentions=[MessageMention(user_id="999", display_name="coco", is_bot=True)],
        )
        second = MessageContext(
            group_id="100",
            user_id="42",
            message_id="m2",
            plain_text="and estimate token cost?",
            raw_message="and estimate token cost?",
            timestamp=2,
            attachments=[
                MessageAttachment(
                    attachment_type="image",
                    url="https://example.test/second.png",
                )
            ],
        )

        merged = merge_realtime_contexts([first, second])
        slices = split_image_descriptions_by_context([first, second], ["first image", "second image"])

        self.assertEqual(merged.message_id, "m2")
        self.assertTrue(merged.is_direct)
        self.assertTrue(merged.bot_mentioned)
        self.assertIn("用户连续发了 2 条", merged.plain_text)
        self.assertIn("1. @coco can you explain", merged.plain_text)
        self.assertIn("2. and estimate token cost?", merged.plain_text)
        self.assertEqual(
            [attachment.url for attachment in merged.attachments],
            [
                "https://example.test/first.png",
                "https://example.test/second.png",
            ],
        )
        self.assertEqual(merged.mentions[0].user_id, "999")
        self.assertEqual(
            [(item.message_id, values) for item, values in slices],
            [
                ("m1", ["first image"]),
                ("m2", ["second image"]),
            ],
        )

    def test_observation_batch_prefers_natural_pause_near_limit(self) -> None:
        contexts = [
            MessageContext("100", "42", "m1", "first", "first", timestamp=1),
            MessageContext("100", "42", "m2", "second", "second", timestamp=2),
            MessageContext("100", "7", "m3", "third", "third", timestamp=3),
            MessageContext("100", "8", "m4", "after pause", "after pause", timestamp=300),
            MessageContext("100", "8", "m5", "next", "next", timestamp=301),
            MessageContext("100", "9", "m6", "more", "more", timestamp=302),
            MessageContext("100", "9", "m7", "tail", "tail", timestamp=303),
        ]

        size = select_observation_batch_size(
            contexts,
            batch_size=4,
            max_messages_per_batch=6,
            max_interval_seconds=600,
        )
        continuous_size = select_observation_batch_size(
            [replace(context, timestamp=index) for index, context in enumerate(contexts, start=1)],
            batch_size=4,
            max_messages_per_batch=6,
            max_interval_seconds=600,
        )

        self.assertEqual(size, 3)
        self.assertEqual([context.message_id for context in contexts[:size]], ["m1", "m2", "m3"])
        self.assertEqual(
            [context.message_id for context in contexts[size:]],
            ["m4", "m5", "m6", "m7"],
        )
        self.assertEqual(continuous_size, 6)

    def test_onebot_at_segment_is_rendered_with_qq_id(self) -> None:
        message = Message("[CQ:at,qq=123,name=Alice] 叫小明")

        plain_text, mentions = render_message_text_and_mentions(message, bot_id="999")

        self.assertEqual(plain_text, "@Alice(QQ:123) 叫小明")
        self.assertEqual(len(mentions), 1)
        self.assertEqual(mentions[0].user_id, "123")
        self.assertEqual(mentions[0].display_name, "Alice")
        self.assertFalse(mentions[0].is_bot)

    async def test_onebot_forward_segment_is_expanded_with_sender_labels(self) -> None:
        message = Message("[CQ:forward,id=forward-1]")

        async def fetch_forward(forward_id: str) -> dict[str, object]:
            self.assertEqual(forward_id, "forward-1")
            return {
                "messages": [
                    {
                        "sender": {"user_id": 123, "nickname": "Alice"},
                        "message": [
                            {"type": "text", "data": {"text": "hello "}},
                            {"type": "at", "data": {"qq": "456", "name": "Bob"}},
                        ],
                    },
                    {
                        "sender": {"user_id": 456, "nickname": "Bob"},
                        "message": [{"type": "image", "data": {"summary": "chart"}}],
                    },
                ]
            }

        plain_text, mentions = await render_message_text_and_mentions_with_forwards(
            message,
            bot_id="999",
            forward_fetcher=fetch_forward,
        )

        self.assertIn(FORWARDED_RECORD_START, plain_text)
        self.assertIn("Alice(QQ:123): hello @Bob(QQ:456)", plain_text)
        self.assertIn("Bob(QQ:456): [图片: chart]", plain_text)
        self.assertIn(FORWARDED_RECORD_END, plain_text)
        self.assertEqual([mention.user_id for mention in mentions], ["456"])

    async def test_onebot_reply_segment_is_expanded_as_quoted_context(self) -> None:
        message = Message("[CQ:reply,id=reply-1]这句怎么回")

        async def fetch_reply(message_id: str) -> dict[str, object]:
            self.assertEqual(message_id, "reply-1")
            return {
                "sender": {"user_id": 123, "nickname": "Alice"},
                "message": [
                    {"type": "text", "data": {"text": "我喜欢吃鱼 "}},
                    {"type": "at", "data": {"qq": "456", "name": "Bob"}},
                    {"type": "image", "data": {"summary": "chart"}},
                ],
            }

        plain_text, mentions = await render_message_text_and_mentions_with_forwards(
            message,
            bot_id="999",
            reply_fetcher=fetch_reply,
        )

        self.assertIn(QUOTED_MESSAGE_START, plain_text)
        self.assertIn("[被引用消息 #reply-1 | Alice(QQ:123)]", plain_text)
        self.assertIn("我喜欢吃鱼 @Bob(QQ:456)[图片: chart]", plain_text)
        self.assertIn(QUOTED_MESSAGE_END, plain_text)
        self.assertTrue(plain_text.endswith("这句怎么回"))
        self.assertEqual(mentions, [])

    async def test_onebot_reply_segment_image_is_collected_as_attachment(self) -> None:
        message = Message("[CQ:reply,id=reply-1]这句怎么回")

        async def fetch_reply(message_id: str) -> dict[str, object]:
            self.assertEqual(message_id, "reply-1")
            return {
                "sender": {"user_id": 123, "nickname": "Alice"},
                "message": [
                    {
                        "type": "image",
                        "data": {
                            "url": "https://example.test/chart.png",
                            "file": "chart.png",
                            "summary": "chart",
                        },
                    }
                ],
            }

        attachments = await image_attachments_from_message_with_replies(
            message,
            reply_fetcher=fetch_reply,
        )

        self.assertEqual(len(attachments), 1)
        self.assertEqual(attachments[0].url, "https://example.test/chart.png")
        self.assertEqual(attachments[0].file, "chart.png")
        self.assertEqual(attachments[0].summary, "chart")

    def test_strip_forwarded_records_keeps_direct_text_only(self) -> None:
        text = (
            "summarize this\n"
            f"{FORWARDED_RECORD_START}\n"
            "Alice(QQ:123): I like shrimp\n"
            f"{FORWARDED_RECORD_END}"
        )

        self.assertEqual(strip_forwarded_records(text), "summarize this")

    def test_strip_quoted_messages_keeps_direct_text_only(self) -> None:
        text = (
            f"{QUOTED_MESSAGE_START}\n"
            "[被引用消息 #reply-1 | Alice(QQ:123)]\n"
            "  我喜欢吃鱼\n"
            f"{QUOTED_MESSAGE_END}\n"
            "这句怎么回"
        )

        self.assertEqual(strip_quoted_messages(text), "这句怎么回")

    async def test_batch_observation_agent_parses_structured_summary(self) -> None:
        llm = FakeLLM(
            [
                json.dumps(
                    {
                        "memories": [
                            {
                                "message_id": "m1",
                                "owner_type": "user",
                                "subject_user_id": "42",
                                "claim_scope": "self_report",
                                "kind": "preference",
                                "content": "User 42 prefers Rust for backend tools.",
                                "confidence": 0.91,
                                "importance": 0.7,
                            }
                        ],
                        "facts": [
                            {
                                "message_id": "m1",
                                "subject_user_id": "42",
                                "fact_type": "preference",
                                "claim_text": "User 42 prefers Rust for backend tools.",
                                "topic": "Rust",
                                "stance": "positive",
                                "confidence": 0.91,
                                "importance": 0.7,
                                "claim_scope": "self_report",
                                "evidence_text": "I prefer Rust for backend tools.",
                            }
                        ],
                        "reflection": {
                            "summary": "Alice discussed backend language preferences.",
                            "topics": ["Rust", "backend"],
                            "importance": 0.65,
                        },
                    }
                )
            ]
        )
        agent = BatchObservationAgent(test_config(Path("unused.sqlite3")), llm)
        contexts = [
            MessageContext(
                group_id="100",
                user_id="42",
                message_id="m1",
                plain_text="I prefer Rust for backend tools.",
                raw_message="I prefer Rust for backend tools.",
                sender_name="Alice",
            ),
            MessageContext(
                group_id="100",
                user_id="7",
                message_id="m2",
                plain_text="same",
                raw_message="same",
                sender_name="Bob",
            ),
        ]

        result = await agent.summarize("100", contexts, [], [])

        self.assertEqual(llm.text_call_purposes, ["batch_observation"])
        self.assertEqual(len(result.memories), 1)
        self.assertEqual(result.memories[0].owner_id, "42")
        self.assertEqual(result.memories[0].subject_user_id, "42")
        self.assertEqual(result.memories[0].evidence_message_id, "m1")
        self.assertEqual(len(result.facts), 1)
        self.assertEqual(result.facts[0].subject_user_id, "42")
        self.assertEqual(result.facts[0].topic, "Rust")
        self.assertEqual(result.facts[0].evidence_message_id, "m1")
        self.assertIsNotNone(result.reflection)
        self.assertEqual(result.reflection.evidence_message_id, "batch-m1-m2")  # type: ignore[union-attr]
        self.assertIn("Rust", result.reflection.content)  # type: ignore[union-attr]

    def test_outgoing_qq_label_is_parsed_as_at_segment(self) -> None:
        parts = parse_outgoing_mention_parts("ask @QQ:3856199161 about it")

        self.assertEqual(
            [(part.kind, part.text, part.user_id) for part in parts],
            [
                ("text", "ask ", ""),
                ("at", "", "3856199161"),
                ("text", " about it", ""),
            ],
        )

    def test_outgoing_named_qq_label_is_parsed_as_at_segment(self) -> None:
        parts = parse_outgoing_mention_parts("@Alice(QQ:123) can explain")

        self.assertEqual(
            [(part.kind, part.text, part.user_id) for part in parts],
            [
                ("at", "", "123"),
                ("text", " can explain", ""),
            ],
        )

    async def test_active_mode_observes_unknown_identity_reference(self) -> None:
        agent = ParticipationPolicyAgent(test_config(Path("unused.sqlite3")), FakeLLM())
        context = MessageContext(
            group_id="100",
            user_id="42",
            message_id="m-unknown-name",
            plain_text="谁是牛宝宝",
            raw_message="谁是牛宝宝",
        )
        perception = PerceptionResult(
            is_question=True,
            is_self_disclosure=False,
            mentions_bot=False,
            topics=["称呼"],
            emotion_hint="neutral",
            confidence=0.9,
        )

        decision = await agent.decide(
            context,
            perception,
            "active",
            ConversationSnapshot(unknown_name_refs=["牛宝宝"]),
        )

        self.assertEqual(decision.action, "observe")
        self.assertIn("unresolved", decision.reason)

    async def test_direct_unknown_identity_reference_asks_for_confirmation(self) -> None:
        agent = ResponseAgent(test_config(Path("unused.sqlite3")), FakeLLM(["这句不应该被用到"]))
        context = MessageContext(
            group_id="100",
            user_id="42",
            message_id="m-unknown-direct",
            plain_text="可可，谁是牛宝宝",
            raw_message="可可，谁是牛宝宝",
            is_direct=True,
        )
        decision = ParticipationDecision("reply", "direct", "passive", 1.0)

        draft = await agent.generate(
            context,
            PerceptionResult(True, False, True),
            decision,
            ConversationSnapshot(unknown_name_refs=["牛宝宝"]),
        )

        self.assertIn("牛宝宝", draft.text or "")
        self.assertIn("QQ", draft.text or "")

    async def test_identity_reply_falls_back_to_target_fact_when_llm_is_uncertain(self) -> None:
        fact = FactRecord(
            id=1,
            subject_user_id="1657222326",
            fact_type="identity",
            claim_text="QQ:1657222326 的称呼是牛宝宝。",
            topic="称呼",
            stance="neutral",
            confidence=0.95,
            status="accepted",
            claim_scope="self_report",
            source_user_id="1657222326",
            source_group_id="100",
            evidence_message_id="m1",
            evidence_text="我叫牛宝宝",
            created_at=1,
            updated_at=1,
        )
        llm = FakeLLM(["我不太确定。"])
        agent = ResponseAgent(test_config(Path("unused.sqlite3")), llm)
        context = MessageContext(
            group_id="100",
            user_id="42",
            message_id="m-known-name",
            plain_text="谁是牛宝宝",
            raw_message="谁是牛宝宝",
            is_direct=True,
        )

        draft = await agent.generate(
            context,
            PerceptionResult(True, False, True),
            ParticipationDecision("reply", "direct", "passive", 1.0),
            ConversationSnapshot(
                target_users=[
                    TargetUserContext(
                        user_id="1657222326",
                        resolution_status="resolved",
                        match_reason="alias:牛宝宝",
                        aliases=["牛宝宝"],
                        facts=[fact],
                    )
                ]
            ),
        )

        self.assertEqual(draft.text, "牛宝宝是 QQ:1657222326")
        self.assertIn("被询问/提及成员资料", llm.text_calls[0][1])
        self.assertIn("不要主动转移话题", llm.text_calls[0][1])

    async def test_mentioned_member_fallback_uses_mentioned_qq(self) -> None:
        context = MessageContext(
            group_id="100",
            user_id="42",
            message_id="m-at",
            plain_text="这个人 @Alice(QQ:123) 叫小明",
            raw_message="[CQ:at,qq=123,name=Alice] 叫小明",
            mentions=[MessageMention(user_id="123", display_name="Alice")],
        )
        perception = PerceptionResult(
            is_question=False,
            is_self_disclosure=False,
            mentions_bot=False,
            topics=[],
            emotion_hint="neutral",
            confidence=0.7,
        )

        memories = await MemoryCuratorAgent(FakeLLM()).extract(
            context,
            perception,
            ConversationSnapshot(),
        )
        facts = await FactExtractorAgent(FakeLLM()).extract(
            context,
            perception,
            ConversationSnapshot(),
        )

        self.assertTrue(
            any(
                memory.subject_user_id == "123"
                and memory.kind == "alias"
                and memory.content == "小明"
                for memory in memories
            )
        )
        self.assertTrue(
            any(
                fact.subject_user_id == "123"
                and fact.fact_type == "identity"
                and "小明" in fact.claim_text
                for fact in facts
            )
        )

    async def test_silent_mode_cannot_be_overridden_by_llm(self) -> None:
        llm = FakeLLM(
            [
                '{"is_question":true,"is_self_disclosure":false,"topics":["AI"],'
                '"emotion_hint":"neutral","confidence":0.9}',
                '{"memories":[]}',
                '{"closeness":3,"trust":3,"familiarity":3,"tension":0,'
                '"summary_patch":"想聊天","reason":"direct"}',
            ]
        )
        config = test_config(Path("unused.sqlite3"))
        pipeline = AgentPipeline(config, llm)
        context = MessageContext(
            group_id="100",
            user_id="42",
            message_id="m1",
            plain_text="可可你觉得 AI 怎么样？",
            raw_message="可可你觉得 AI 怎么样？",
            is_direct=True,
        )

        result = await pipeline.run(context, "silent", ConversationSnapshot())

        self.assertEqual(result.decision.action, "observe")
        self.assertIsNone(result.reply)

    async def test_relationship_agent_does_not_summarize_message_logs(self) -> None:
        llm = FakeLLM(
            [
                '{"closeness":0,"trust":0,"familiarity":1,"tension":0,'
                '"summary_patch":"分享冰淇淋图片","reason":"observed"}'
            ]
        )
        agent = RelationshipAgent(llm)
        context = MessageContext(
            group_id="100",
            user_id="814207765",
            message_id="m-log",
            plain_text="",
            raw_message="[CQ:image]",
        )
        perception = PerceptionResult(
            is_question=False,
            is_self_disclosure=False,
            mentions_bot=False,
            topics=["冰淇淋"],
            emotion_hint="neutral",
            confidence=0.9,
        )

        result = await agent.calculate_delta(context, perception, ConversationSnapshot())

        self.assertEqual(result.familiarity, 1)
        self.assertEqual(result.summary_patch, "")

    async def test_fact_extractor_creates_structured_fact(self) -> None:
        llm = FakeLLM(
            [
                '{"facts":[{"subject_user_id":"42","fact_type":"opinion",'
                '"claim_text":"用户42认为刮刮乐活动像负期望彩票",'
                '"topic":"刮刮乐活动","stance":"negative","confidence":0.9,'
                '"claim_scope":"self_report","evidence_text":"我觉得这个刮刮乐像负期望彩票"}]}'
            ]
        )
        agent = FactExtractorAgent(llm)
        context = MessageContext(
            group_id="100",
            user_id="42",
            message_id="m-fact",
            plain_text="我觉得这个刮刮乐像负期望彩票",
            raw_message="我觉得这个刮刮乐像负期望彩票",
        )
        perception = PerceptionResult(
            is_question=False,
            is_self_disclosure=True,
            mentions_bot=False,
            topics=["游戏"],
            emotion_hint="neutral",
            confidence=0.9,
        )

        facts = await agent.extract(context, perception, ConversationSnapshot())

        self.assertEqual(len(facts), 1)
        self.assertEqual(facts[0].subject_user_id, "42")
        self.assertEqual(facts[0].topic, "刮刮乐活动")
        self.assertIn("负期望彩票", facts[0].claim_text)

    async def test_fact_extractor_rejects_fact_from_recent_context_only(self) -> None:
        llm = FakeLLM(
            [
                '{"facts":[{"subject_user_id":"42","fact_type":"preference",'
                '"claim_text":"用户42喜欢 Rust","topic":"Rust","stance":"positive",'
                '"confidence":0.9,"claim_scope":"self_report","evidence_text":"我喜欢 Rust"}]}'
            ]
        )
        agent = FactExtractorAgent(llm)
        context = MessageContext(
            group_id="100",
            user_id="42",
            message_id="m-context-only",
            plain_text="可可你怎么看这个话题？",
            raw_message="可可你怎么看这个话题？",
        )

        facts = await agent.extract(
            context,
            PerceptionResult(True, False, True),
            ConversationSnapshot(recent_messages=["Alice: 我喜欢 Rust"]),
        )

        self.assertEqual(facts, [])
        self.assertIn("最近上下文只用于理解当前消息", llm.text_calls[0][1])

    async def test_fact_extractor_rejects_low_value_chat_action(self) -> None:
        llm = FakeLLM(
            [
                '{"facts":[{"subject_user_id":"42","fact_type":"other",'
                '"claim_text":"用户42继续聊比赛感受","topic":"比赛感受",'
                '"stance":"neutral","confidence":0.95,"claim_scope":"self_report",'
                '"evidence_text":"继续聊比赛感受"}]}'
            ]
        )
        agent = FactExtractorAgent(llm)
        context = MessageContext(
            group_id="100",
            user_id="42",
            message_id="m-low",
            plain_text="继续聊比赛感受",
            raw_message="继续聊比赛感受",
        )
        perception = PerceptionResult(
            is_question=False,
            is_self_disclosure=False,
            mentions_bot=False,
            topics=["比赛"],
            emotion_hint="neutral",
            confidence=0.9,
        )

        facts = await agent.extract(context, perception, ConversationSnapshot())

        self.assertEqual(facts, [])

    async def test_self_narrative_is_prepared_before_reply(self) -> None:
        llm = FakeLLM(
            [
                '{"is_question":true,"is_self_disclosure":false,"topics":["旅行"],'
                '"emotion_hint":"neutral","confidence":0.9}',
                '{"memories":[]}',
                '{"closeness":1,"trust":0,"familiarity":2,"tension":0,'
                '"summary_patch":"","reason":"direct"}',
                '{"kind":"self_preference","content":"我喜欢海边潮湿的风和声音",'
                '"fictionality":"fictional_light","confidence":0.84,"importance":0.62}',
                "喜欢呀，我喜欢海边潮湿的风和声音。",
            ]
        )
        config = test_config(Path("unused.sqlite3"))
        pipeline = AgentPipeline(config, llm)
        context = MessageContext(
            group_id="100",
            user_id="42",
            message_id="m2",
            plain_text="可可你喜欢海边吗？",
            raw_message="可可你喜欢海边吗？",
            is_direct=True,
        )

        result = await pipeline.run(context, "passive", ConversationSnapshot())

        self.assertTrue(result.reply)
        self.assertEqual(result.reply_self_memories[0].owner_type, "self")
        self.assertEqual(result.reply_self_memories[0].kind, "self_preference")
        self.assertIn("海边", result.reply_self_memories[0].content)

    async def test_unsafe_self_narrative_candidate_is_rejected(self) -> None:
        llm = FakeLLM(
            [
                '{"is_question":true,"is_self_disclosure":false,"topics":["生活"],'
                '"emotion_hint":"neutral","confidence":0.9}',
                '{"memories":[]}',
                '{"closeness":1,"trust":0,"familiarity":2,"tension":0,'
                '"summary_patch":"","reason":"direct"}',
                '{"kind":"self_background","content":"我家住在北京朝阳某个小区",'
                '"fictionality":"fictional_light","confidence":0.9,"importance":0.8}',
                "这个我不编太具体啦，笼统一点说我比较喜欢安静的地方。",
            ]
        )
        config = test_config(Path("unused.sqlite3"))
        pipeline = AgentPipeline(config, llm)
        context = MessageContext(
            group_id="100",
            user_id="42",
            message_id="m7",
            plain_text="可可你平时住哪种地方？",
            raw_message="可可你平时住哪种地方？",
            is_direct=True,
        )

        result = await pipeline.run(context, "passive", ConversationSnapshot())

        self.assertTrue(result.reply)
        self.assertEqual(result.reply_self_memories, [])

    async def test_direct_technical_advice_creates_lightweight_self_background(self) -> None:
        llm = FakeLLM(
            [
                '{"is_question":true,"is_self_disclosure":false,"topics":["UE5","蓝图"],'
                '"emotion_hint":"neutral","confidence":0.9}',
                '{"facts":[]}',
                '{"closeness":1,"trust":0,"familiarity":2,"tension":0,'
                '"summary_patch":"","reason":"direct"}',
                '{"kind":"self_background","content":"我之前翻过 UE5 蓝图和材质的入门资料",'
                '"fictionality":"fictional_light","confidence":0.84,"importance":0.58}',
                "我之前翻过一点 UE5 蓝图资料，蓝图先拆清输入输出会稳一点。",
            ]
        )
        config = test_config(Path("unused.sqlite3"))
        pipeline = AgentPipeline(config, llm)
        context = MessageContext(
            group_id="100",
            user_id="42",
            message_id="m-ue5-direct",
            plain_text="可可 UE5 蓝图怎么组织比较好？",
            raw_message="可可 UE5 蓝图怎么组织比较好？",
            is_direct=True,
        )

        result = await pipeline.run(context, "passive", ConversationSnapshot())

        self.assertTrue(result.reply)
        self.assertEqual(result.decision.action, "reply")
        self.assertEqual(len(result.reply_self_memories), 1)
        self.assertEqual(result.reply_self_memories[0].kind, "self_background")
        self.assertIn("UE5", result.reply_self_memories[0].content)

    async def test_proactive_technical_reply_creates_lightweight_self_background(self) -> None:
        llm = FakeLLM(
            [
                '{"is_question":false,"is_self_disclosure":false,"topics":["UE5","材质"],'
                '"emotion_hint":"neutral","confidence":0.9}',
                '{"facts":[]}',
                '{"closeness":0,"trust":0,"familiarity":1,"tension":0,'
                '"summary_patch":"","reason":"observed"}',
                '{"action":"proactive_reply","score":0.86,"value_type":"useful_context",'
                '"value_score":0.82,"reason":"可以补一个材质组织角度"}',
                '{"kind":"self_background","content":"我之前翻过 UE5 蓝图和材质的入门资料",'
                '"fictionality":"fictional_light","confidence":0.84,"importance":0.58}',
                "UE5 材质可以先把复用节点收成函数，后面改参数会轻松点。",
            ]
        )
        config = test_config(Path("unused.sqlite3"))
        pipeline = AgentPipeline(config, llm)
        context = MessageContext(
            group_id="100",
            user_id="42",
            message_id="m-ue5-proactive",
            plain_text="UE5 材质这块越堆越乱，感觉后面不好维护",
            raw_message="UE5 材质这块越堆越乱，感觉后面不好维护",
        )

        result = await pipeline.run(
            context,
            "active",
            ConversationSnapshot(recent_human_messages_60s=2),
        )

        self.assertEqual(result.decision.action, "proactive_reply")
        self.assertTrue(result.reply)
        self.assertEqual(result.reply_self_memories[0].kind, "self_background")

    async def test_proactive_technical_reply_is_blocked_when_background_is_too_specific(self) -> None:
        llm = FakeLLM(
            [
                '{"is_question":false,"is_self_disclosure":false,"topics":["UE5","蓝图"],'
                '"emotion_hint":"neutral","confidence":0.9}',
                '{"facts":[]}',
                '{"closeness":0,"trust":0,"familiarity":1,"tension":0,'
                '"summary_patch":"","reason":"observed"}',
                '{"action":"proactive_reply","score":0.86,"value_type":"useful_context",'
                '"value_score":0.82,"reason":"可以补一个蓝图经验"}',
                '{"kind":"self_background","content":"我在某公司用 UE5 上班做过项目",'
                '"fictionality":"fictional_light","confidence":0.9,"importance":0.8}',
                "这句不应该被生成",
            ]
        )
        config = test_config(Path("unused.sqlite3"))
        pipeline = AgentPipeline(config, llm)
        context = MessageContext(
            group_id="100",
            user_id="42",
            message_id="m-ue5-block",
            plain_text="UE5 蓝图通信这里大家说法不一样",
            raw_message="UE5 蓝图通信这里大家说法不一样",
        )

        result = await pipeline.run(
            context,
            "active",
            ConversationSnapshot(recent_human_messages_60s=2),
        )

        self.assertIsNone(result.reply)
        self.assertEqual(result.decision.action, "observe")
        self.assertIn("self background gate blocked", result.decision.reason)
        self.assertEqual(result.reply_self_memories, [])
        self.assertEqual(llm.replies, ["这句不应该被生成"])

    async def test_existing_self_background_prevents_duplicate_topic_memory(self) -> None:
        existing = MemoryRecord(
            id=7,
            owner_type="self",
            owner_id="bot",
            kind="self_background",
            content="我之前翻过 UE5 蓝图和材质入门资料",
            confidence=0.86,
            importance=0.6,
            status="active",
            updated_at=1,
            source_user_id="bot",
            source_group_id="100",
            subject_user_id="bot",
            claim_scope="bot_directed",
            verification_status="accepted",
        )
        llm = FakeLLM(
            [
                '{"is_question":true,"is_self_disclosure":false,"topics":["UE5","蓝图"],'
                '"emotion_hint":"neutral","confidence":0.9}',
                '{"facts":[]}',
                '{"closeness":1,"trust":0,"familiarity":2,"tension":0,'
                '"summary_patch":"","reason":"direct"}',
                "可以先把蓝图职责拆开，别让一个 Actor 管太多事。",
            ]
        )
        config = test_config(Path("unused.sqlite3"))
        pipeline = AgentPipeline(config, llm)
        context = MessageContext(
            group_id="100",
            user_id="42",
            message_id="m-ue5-existing",
            plain_text="可可 UE5 蓝图职责怎么拆？",
            raw_message="可可 UE5 蓝图职责怎么拆？",
            is_direct=True,
        )

        result = await pipeline.run(
            context,
            "passive",
            ConversationSnapshot(self_memories=[existing]),
        )

        self.assertEqual(result.reply, "可以先把蓝图职责拆开，别让一个 Actor 管太多事")
        self.assertEqual(result.reply_self_memories, [])

    async def test_direct_technical_reply_degrades_when_background_candidate_is_rejected(self) -> None:
        llm = FakeLLM(
            [
                '{"is_question":true,"is_self_disclosure":false,"topics":["UE5","蓝图"],'
                '"emotion_hint":"neutral","confidence":0.9}',
                '{"facts":[]}',
                '{"closeness":1,"trust":0,"familiarity":2,"tension":0,'
                '"summary_patch":"","reason":"direct"}',
                '{"kind":"self_background","content":"我在某公司用 UE5 上班做过项目",'
                '"fictionality":"fictional_light","confidence":0.9,"importance":0.8}',
                "按一般理解，蓝图先分清输入输出和事件边界会好维护些。",
            ]
        )
        config = test_config(Path("unused.sqlite3"))
        pipeline = AgentPipeline(config, llm)
        context = MessageContext(
            group_id="100",
            user_id="42",
            message_id="m-ue5-degrade",
            plain_text="可可 UE5 蓝图通信怎么处理？",
            raw_message="可可 UE5 蓝图通信怎么处理？",
            is_direct=True,
        )

        result = await pipeline.run(context, "passive", ConversationSnapshot())
        response_system_prompt, response_user_prompt = llm.text_calls[-2]

        self.assertEqual(result.decision.action, "reply")
        self.assertEqual(result.reply_self_memories, [])
        self.assertIn("按一般理解", result.reply or "")
        self.assertIn("不要说自己用过或做过项目", response_system_prompt)
        self.assertIn("自我背景门禁", response_user_prompt)
        self.assertIn("不要说自己用过或做过项目", response_user_prompt)

    async def test_lexicon_learning_disabled_does_not_search(self) -> None:
        config = test_config(Path("unused.sqlite3"))
        search = FakeSearch(
            [SearchResult(title="内卷", url="https://example.test", snippet="一种网络用语。")]
        )
        pipeline = AgentPipeline(config, FakeLLM(), search)  # type: ignore[arg-type]
        context = MessageContext(
            group_id="100",
            user_id="42",
            message_id="m3",
            plain_text="内卷是什么意思",
            raw_message="内卷是什么意思",
        )

        result = await pipeline.run(context, "silent", ConversationSnapshot())

        self.assertEqual(search.calls, [])
        self.assertEqual([memory for memory in result.memories if memory.kind == "lexicon"], [])

    async def test_vision_analysis_creates_group_image_memory(self) -> None:
        config = replace(
            test_config(Path("unused.sqlite3")),
            vision=VisionConfig(enabled=True, remember_threshold=0.78),
        )
        llm = FakeLLM(
            replies=[
                '{"is_question":true,"is_self_disclosure":false,"topics":["截图"],'
                '"emotion_hint":"neutral","confidence":0.9}',
                '{"memories":[]}',
                '{"closeness":1,"trust":0,"familiarity":2,"tension":0,'
                '"summary_patch":"","reason":"direct"}',
                "像是一张报表截图，重点是收入同比上涨。",
            ],
            vision_replies=[
                '{"images":[{"description":"一张财务报表截图，突出显示收入同比上涨",'
                '"ocr_text":"收入同比上涨 12%","topics":["报表","审计"],'
                '"should_remember":true,"memory":"群里分享了一张收入同比上涨的报表截图",'
                '"confidence":0.88,"importance":0.58}]}'
            ],
        )
        pipeline = AgentPipeline(config, llm)
        context = MessageContext(
            group_id="100",
            user_id="42",
            message_id="m8",
            plain_text="可可帮我看看这张图",
            raw_message="[CQ:image,url=https://example.test/a.png]",
            is_direct=True,
            attachments=[
                MessageAttachment(
                    attachment_type="image",
                    url="https://example.test/a.png",
                    file="a.png",
                )
            ],
        )

        result = await pipeline.run(context, "passive", ConversationSnapshot())
        image_memories = [memory for memory in result.memories if memory.kind == "image_observation"]

        self.assertEqual(llm.vision_calls, [["https://example.test/a.png"]])
        self.assertEqual(result.image_descriptions, ["一张财务报表截图，突出显示收入同比上涨"])
        self.assertEqual(len(image_memories), 1)
        self.assertIn("报表截图", image_memories[0].content)
        self.assertTrue(any(fact.claim_text == "用户42对图片中的报表内容感兴趣" for fact in result.facts))
        self.assertTrue(result.reply)

    async def test_vision_analysis_reuses_cached_image_url(self) -> None:
        with project_temp_directory() as tmp:
            config = replace(
                test_config(Path(tmp) / "bot.sqlite3"),
                vision=VisionConfig(enabled=True, remember_threshold=0.78),
            )
            storage = BotStorage.from_config(config)
            storage.setup()
            llm = FakeLLM(
                vision_replies=[
                    '{"images":[{"description":"一张猫猫表情包，配字是下班了",'
                    '"ocr_text":"下班了","topics":["表情包"],'
                    '"should_remember":true,"memory":"群里常用一张下班猫猫表情包",'
                    '"confidence":0.9,"importance":0.52}]}'
                ]
            )
            agent = VisionAgent(config, llm, storage)
            first = MessageContext(
                group_id="100",
                user_id="42",
                message_id="m-img-1",
                plain_text="",
                raw_message="[CQ:image,url=https://example.test/meme.png]",
                attachments=[
                    MessageAttachment(
                        attachment_type="image",
                        url="https://example.test/meme.png",
                        file="meme.png",
                    )
                ],
            )
            second = replace(first, message_id="m-img-2")

            first_result = await agent.analyze(first)
            second_result = await agent.analyze(second)

            self.assertEqual(llm.vision_calls, [["https://example.test/meme.png"]])
            self.assertEqual(first_result.descriptions, ["一张猫猫表情包，配字是下班了"])
            self.assertEqual(second_result.descriptions, ["一张猫猫表情包，配字是下班了"])
            cached = storage.get_image_vision_cache("https://example.test/meme.png")
            self.assertIsNotNone(cached)
            self.assertGreaterEqual(cached.hit_count, 2)

    async def test_vision_analysis_samples_first_middle_and_last_images(self) -> None:
        config = replace(
            test_config(Path("unused.sqlite3")),
            vision=VisionConfig(enabled=True, max_images_per_message=3),
        )
        llm = FakeLLM(
            vision_replies=[
                '{"images":['
                '{"description":"第一张图片","ocr_text":"","topics":["第一"],'
                '"image_type":"pure_image","should_remember":false,"memory":"","confidence":0.8,"importance":0.4},'
                '{"description":"中间图片","ocr_text":"","topics":["中间"],'
                '"image_type":"pure_image","should_remember":false,"memory":"","confidence":0.8,"importance":0.4},'
                '{"description":"最后一张图片","ocr_text":"","topics":["最后"],'
                '"image_type":"pure_image","should_remember":false,"memory":"","confidence":0.8,"importance":0.4}'
                ']}'
            ]
        )
        agent = VisionAgent(config, llm)
        context = MessageContext(
            group_id="100",
            user_id="42",
            message_id="m-many-images",
            plain_text="",
            raw_message="[CQ:image]x5",
            attachments=[
                MessageAttachment(attachment_type="image", url=f"https://example.test/{index}.png", file=f"{index}.png")
                for index in range(5)
            ],
        )

        result = await agent.analyze(context)

        self.assertEqual(
            llm.vision_calls,
            [["https://example.test/0.png", "https://example.test/2.png", "https://example.test/4.png"]],
        )
        self.assertEqual(result.descriptions, ["第一张图片", "中间图片", "最后一张图片"])
        self.assertEqual(result.attachment_descriptions, ("第一张图片", "", "中间图片", "", "最后一张图片"))

    async def test_vision_content_image_creates_interest_fact(self) -> None:
        config = replace(test_config(Path("unused.sqlite3")), vision=VisionConfig(enabled=True))
        llm = FakeLLM(
            vision_replies=[
                '{"images":[{"description":"一张演唱会海报截图",'
                '"ocr_text":"巡演开票 7月12日","topics":["演唱会"],'
                '"image_type":"content_image","should_remember":false,"memory":"",'
                '"confidence":0.82,"importance":0.4}]}'
            ]
        )
        agent = VisionAgent(config, llm)
        context = MessageContext(
            group_id="100",
            user_id="42",
            message_id="m-content-image",
            plain_text="",
            raw_message="[CQ:image,url=https://example.test/poster.png]",
            attachments=[
                MessageAttachment(attachment_type="image", url="https://example.test/poster.png", file="poster.png")
            ],
        )

        result = await agent.analyze(context)

        self.assertEqual(len(result.fact_candidates), 1)
        self.assertEqual(result.fact_candidates[0].subject_user_id, "42")
        self.assertEqual(result.fact_candidates[0].topic, "演唱会")
        self.assertEqual(result.fact_candidates[0].claim_text, "用户42对图片中的演唱会内容感兴趣")
        self.assertIn("ocr=巡演开票 7月12日", result.fact_candidates[0].evidence_text)

    async def test_vision_pure_image_creates_image_type_interest_fact(self) -> None:
        config = replace(test_config(Path("unused.sqlite3")), vision=VisionConfig(enabled=True))
        llm = FakeLLM(
            vision_replies=[
                '{"images":[{"description":"一张夜景城市照片",'
                '"ocr_text":"","topics":["城市夜景"],'
                '"image_type":"pure_image","should_remember":false,"memory":"",'
                '"confidence":0.81,"importance":0.4}]}'
            ]
        )
        agent = VisionAgent(config, llm)
        context = MessageContext(
            group_id="100",
            user_id="42",
            message_id="m-pure-image",
            plain_text="",
            raw_message="[CQ:image,url=https://example.test/city.png]",
            attachments=[
                MessageAttachment(attachment_type="image", url="https://example.test/city.png", file="city.png")
            ],
        )

        result = await agent.analyze(context)

        self.assertEqual(len(result.fact_candidates), 1)
        self.assertEqual(result.fact_candidates[0].topic, "城市夜景")
        self.assertEqual(result.fact_candidates[0].claim_text, "用户42对城市夜景这类图片感兴趣")

    async def test_vision_analysis_detects_sticker_candidate_when_enabled(self) -> None:
        config = replace(
            test_config(Path("unused.sqlite3")),
            vision=VisionConfig(enabled=True),
            stickers=StickerConfig(enabled=True, min_confidence=0.7),
        )
        llm = FakeLLM(
            vision_replies=[
                '{"images":[{"description":"一张猫猫表情包，配字是下班了",'
                '"ocr_text":"下班了","topics":["表情包","猫猫"],'
                '"should_remember":false,"memory":"","confidence":0.9,"importance":0.4,'
                '"is_sticker":true,"sticker_mood":"疲惫",'
                '"sticker_usage":"适合大家聊到下班、犯困或想摆一下时使用",'
                '"sticker_tags":["下班","困","猫猫"],"sticker_confidence":0.86}]}'
            ]
        )
        agent = VisionAgent(config, llm)
        context = MessageContext(
            group_id="100",
            user_id="42",
            message_id="m-sticker",
            plain_text="",
            raw_message="[CQ:image,url=https://example.test/meme.png]",
            attachments=[
                MessageAttachment(
                    attachment_type="image",
                    url="https://example.test/meme.png",
                    file="meme.png",
                )
            ],
        )

        result = await agent.analyze(context)

        self.assertEqual(len(result.sticker_candidates), 1)
        self.assertEqual(result.fact_candidates, ())
        self.assertEqual(result.sticker_candidates[0].mood, "疲惫")
        self.assertIn("下班", result.sticker_candidates[0].usage)
        self.assertEqual(result.sticker_candidates[0].file, "meme.png")

    async def test_sticker_selector_uses_matching_asset_from_llm(self) -> None:
        config = replace(
            test_config(Path("unused.sqlite3")),
            stickers=StickerConfig(enabled=True, selection_threshold=0.68),
        )
        asset = StickerAssetRecord(
            id=7,
            group_id="100",
            source_user_id="42",
            source_message_id="m1",
            url="https://example.test/meme.png",
            file="meme.png",
            local_path="E:\\tmp\\meme.png",
            sha256="abc",
            description="一张猫猫下班表情包",
            ocr_text="下班了",
            mood="疲惫",
            usage="适合聊到下班、犯困或想摆一下时使用",
            tags=("下班", "困", "猫猫"),
            confidence=0.86,
            enabled=True,
            created_at=1,
            updated_at=1,
            last_seen_at=1,
        )
        selector = StickerSelectorAgent(
            config,
            FakeLLM(['{"asset_id":7,"confidence":0.82,"reason":"回复在接下班梗"}']),
        )
        context = MessageContext(
            group_id="100",
            user_id="42",
            message_id="m2",
            plain_text="终于下班了",
            raw_message="终于下班了",
        )
        decision = ParticipationDecision("reply", "message is directed to the bot", "passive", 1.0)

        selected = await selector.select(
            context,
            decision,
            ConversationSnapshot(sticker_assets=[asset]),
            "辛苦了，今天可以摆一下。",
        )

        self.assertEqual(selected, asset)

    async def test_active_participation_rejects_agreement_only_value(self) -> None:
        config = test_config(Path("unused.sqlite3"))
        agent = ParticipationPolicyAgent(
            config,
            FakeLLM(
                [
                    '{"action":"proactive_reply","score":0.9,"value_type":"agreement",'
                    '"value_score":0.95,"reason":"只是赞同上一位群友"}'
                ]
            ),
        )
        context = MessageContext(
            group_id="100",
            user_id="42",
            message_id="m-active",
            plain_text="这个活动就是负期望彩票，抽多了肯定亏",
            raw_message="这个活动就是负期望彩票，抽多了肯定亏",
        )
        perception = PerceptionResult(
            is_question=False,
            is_self_disclosure=False,
            mentions_bot=False,
            topics=["游戏活动"],
            confidence=0.8,
        )

        decision = await agent.decide(context, perception, "active", ConversationSnapshot(recent_human_messages_60s=3))

        self.assertEqual(decision.action, "observe")
        self.assertEqual(decision.value_type, "agreement")
        self.assertIn("value gate rejected", decision.reason)

    async def test_busy_chat_requires_high_value_proactive_reply(self) -> None:
        config = test_config(Path("unused.sqlite3"))
        agent = ParticipationPolicyAgent(
            config,
            FakeLLM(
                [
                    '{"action":"proactive_reply","score":0.9,"value_type":"humor",'
                    '"value_score":0.96,"reason":"想接一个轻松梗"}'
                ]
            ),
        )
        context = MessageContext(
            group_id="100",
            user_id="42",
            message_id="m-busy",
            plain_text="这版本到底该不该抽，感觉大家说法完全不一样",
            raw_message="这版本到底该不该抽，感觉大家说法完全不一样",
        )
        perception = PerceptionResult(
            is_question=True,
            is_self_disclosure=False,
            mentions_bot=False,
            topics=["游戏"],
            confidence=0.85,
        )

        decision = await agent.decide(context, perception, "active", ConversationSnapshot(recent_human_messages_60s=8))

        self.assertEqual(decision.action, "observe")
        self.assertEqual(decision.traffic_level, "busy")
        self.assertEqual(decision.value_type, "humor")

    async def test_bot_name_object_mention_can_reply_when_referring_to_bot(self) -> None:
        config = test_config(Path("unused.sqlite3"))
        llm = FakeLLM(
            [
                '{"target":"discussing_bot","confidence":0.9,"value_type":"direct_reply",'
                '"reason":"讨论的是本群机器人可可的形象"}',
            ]
        )
        agent = ParticipationPolicyAgent(config, llm)
        context = MessageContext(
            group_id="100",
            user_id="42",
            message_id="m-bot-object",
            plain_text="顺便一提可可的形象的确是固定的",
            raw_message="顺便一提可可的形象的确是固定的",
            bot_mentioned=True,
        )
        perception = PerceptionResult(
            is_question=False,
            is_self_disclosure=False,
            mentions_bot=True,
            topics=["机器人设定"],
            confidence=0.82,
        )
        snapshot = ConversationSnapshot(
            recent_messages=["bot: 我可以解释一下", "alice: 顺便一提可可的形象的确是固定的"],
            recent_bot_reply_to_user="我可以解释一下",
            recent_bot_reply_to_user_seconds=18,
        )

        decision = await agent.decide(context, perception, "passive", snapshot)

        self.assertEqual(decision.action, "reply")
        self.assertEqual(decision.value_type, "direct_reply")
        self.assertIn("addressing gate", decision.reason)
        self.assertEqual(llm.text_call_purposes, ["addressing_gate"])

    async def test_bot_name_mention_observes_when_llm_says_referent_is_not_bot(self) -> None:
        config = test_config(Path("unused.sqlite3"))
        llm = FakeLLM(
            [
                '{"target":"other_referent","confidence":0.88,"value_type":"none",'
                '"reason":"这里的可可是转发记录里的另一个机器人"}',
            ]
        )
        agent = ParticipationPolicyAgent(config, llm)
        context = MessageContext(
            group_id="100",
            user_id="42",
            message_id="m-other-coco",
            plain_text="转发里那个可可想下班",
            raw_message="转发里那个可可想下班",
            bot_mentioned=True,
        )
        perception = PerceptionResult(
            is_question=False,
            is_self_disclosure=False,
            mentions_bot=True,
            topics=["转发记录"],
            confidence=0.82,
        )

        decision = await agent.decide(context, perception, "passive", ConversationSnapshot())

        self.assertEqual(decision.action, "observe")
        self.assertIn("not addressed", decision.reason)
        self.assertEqual(llm.text_call_purposes, ["addressing_gate"])

    async def test_bot_name_mention_can_reply_when_addressing_gate_accepts(self) -> None:
        config = test_config(Path("unused.sqlite3"))
        llm = FakeLLM(
            [
                '{"target":"addressed_to_bot","confidence":0.86,"value_type":"answer",'
                '"reason":"用户在请机器人判断"}'
            ]
        )
        agent = ParticipationPolicyAgent(config, llm)
        context = MessageContext(
            group_id="100",
            user_id="42",
            message_id="m-bot-addressed",
            plain_text="这个可可看看有没有问题？",
            raw_message="这个可可看看有没有问题？",
            bot_mentioned=True,
        )
        perception = PerceptionResult(
            is_question=True,
            is_self_disclosure=False,
            mentions_bot=True,
            topics=["求助"],
            confidence=0.82,
        )

        decision = await agent.decide(context, perception, "passive", ConversationSnapshot())

        self.assertEqual(decision.action, "reply")
        self.assertEqual(decision.value_type, "answer")
        self.assertIn("addressing gate", decision.reason)
        self.assertEqual(llm.text_call_purposes, ["addressing_gate"])

    async def test_passive_recent_interaction_followup_can_reply_without_name(self) -> None:
        config = test_config(Path("unused.sqlite3"))
        llm = FakeLLM(
            [
                '{"action":"reply","confidence":0.84,"value_type":"answer",'
                '"reason":"用户在继续追问刚才的抽卡建议"}'
            ]
        )
        agent = ParticipationPolicyAgent(config, llm)
        context = MessageContext(
            group_id="100",
            user_id="42",
            message_id="m-followup",
            plain_text="那我是不是先别抽？",
            raw_message="那我是不是先别抽？",
        )
        perception = PerceptionResult(
            is_question=True,
            is_self_disclosure=False,
            mentions_bot=False,
            topics=["抽卡"],
            confidence=0.82,
        )
        snapshot = ConversationSnapshot(
            recent_messages=["bot: 预算不多的话可以先等", "alice: 那我是不是先别抽？"],
            speaker_recent_messages=["alice: 我预算不多", "alice: 那我是不是先别抽？"],
            recent_bot_reply_to_user="预算不多的话可以先等",
            recent_bot_reply_to_user_seconds=24,
        )

        decision = await agent.decide(context, perception, "passive", snapshot)
        _, user_prompt = llm.text_calls[0]

        self.assertEqual(decision.action, "reply")
        self.assertEqual(decision.value_type, "answer")
        self.assertIn("recent interaction follow-up", decision.reason)
        self.assertIn("上次机器人回复该用户", user_prompt)

    async def test_passive_recent_interaction_observes_non_followup(self) -> None:
        config = test_config(Path("unused.sqlite3"))
        llm = FakeLLM(
            [
                '{"action":"observe","confidence":0.9,"value_type":"none",'
                '"reason":"用户换了新话题"}'
            ]
        )
        agent = ParticipationPolicyAgent(config, llm)
        context = MessageContext(
            group_id="100",
            user_id="42",
            message_id="m-not-followup",
            plain_text="今晚有人打游戏吗",
            raw_message="今晚有人打游戏吗",
        )
        perception = PerceptionResult(
            is_question=True,
            is_self_disclosure=False,
            mentions_bot=False,
            topics=["游戏"],
            confidence=0.82,
        )
        snapshot = ConversationSnapshot(
            recent_messages=["bot: 预算不多的话可以先等", "alice: 今晚有人打游戏吗"],
            speaker_recent_messages=["alice: 今晚有人打游戏吗"],
            recent_bot_reply_to_user="预算不多的话可以先等",
            recent_bot_reply_to_user_seconds=35,
        )

        decision = await agent.decide(context, perception, "passive", snapshot)

        self.assertEqual(decision.action, "observe")
        self.assertEqual(decision.reason, "passive mode requires direct mention")
        self.assertEqual(len(llm.text_calls), 1)

    async def test_active_participation_observes_live_event_context(self) -> None:
        config = test_config(Path("unused.sqlite3"))
        llm = FakeLLM(
            [
                '{"action":"proactive_reply","score":0.9,"value_type":"useful_context",'
                '"value_score":0.95,"reason":"想补充赛况"}'
            ]
        )
        agent = ParticipationPolicyAgent(config, llm)
        context = MessageContext(
            group_id="100",
            user_id="42",
            message_id="m-live-event",
            plain_text="现在比分 2:1，刚刚又进球了",
            raw_message="现在比分 2:1，刚刚又进球了",
        )
        perception = PerceptionResult(
            is_question=False,
            is_self_disclosure=False,
            mentions_bot=False,
            topics=["比赛"],
            confidence=0.85,
        )

        decision = await agent.decide(context, perception, "active", ConversationSnapshot(recent_human_messages_60s=4))

        self.assertEqual(decision.action, "observe")
        self.assertIn("live event", decision.reason)
        self.assertEqual(llm.text_calls, [])

    async def test_active_participation_accepts_synthesis_value(self) -> None:
        config = test_config(Path("unused.sqlite3"))
        agent = ParticipationPolicyAgent(
            config,
            FakeLLM(
                [
                    '{"action":"proactive_reply","score":0.86,"value_type":"synthesis",'
                    '"value_score":0.82,"reason":"可以整理两边观点的差异"}'
                ]
            ),
        )
        context = MessageContext(
            group_id="100",
            user_id="42",
            message_id="m-synthesis",
            plain_text="一个人说要抽，一个人说等复刻，我有点纠结",
            raw_message="一个人说要抽，一个人说等复刻，我有点纠结",
        )
        perception = PerceptionResult(
            is_question=False,
            is_self_disclosure=False,
            mentions_bot=False,
            topics=["游戏"],
            confidence=0.85,
        )

        decision = await agent.decide(context, perception, "active", ConversationSnapshot(recent_human_messages_60s=3))

        self.assertEqual(decision.action, "proactive_reply")
        self.assertEqual(decision.value_type, "synthesis")

    async def test_proactive_response_suppresses_low_value_agreement(self) -> None:
        config = test_config(Path("unused.sqlite3"))
        agent = ResponseAgent(config, FakeLLM(["确实，我也觉得"]))
        context = MessageContext(
            group_id="100",
            user_id="42",
            message_id="m-reply",
            plain_text="这池子就是亏",
            raw_message="这池子就是亏",
        )
        perception = PerceptionResult(False, False, False, ["游戏"], "neutral", 0.8)
        decision = ParticipationDecision(
            "proactive_reply",
            "can add context",
            "active",
            0.9,
            "useful_context",
            0.8,
        )

        draft = await agent.generate(context, perception, decision, ConversationSnapshot())

        self.assertIsNone(draft.text)

    async def test_proactive_response_allows_incremental_synthesis(self) -> None:
        config = test_config(Path("unused.sqlite3"))
        agent = ResponseAgent(config, FakeLLM(["可以拆成两点：想要角色就抽，追性价比就等复刻。"]))
        context = MessageContext(
            group_id="100",
            user_id="42",
            message_id="m-reply-ok",
            plain_text="这版本到底该不该抽",
            raw_message="这版本到底该不该抽",
        )
        perception = PerceptionResult(True, False, False, ["游戏"], "neutral", 0.8)
        decision = ParticipationDecision(
            "proactive_reply",
            "can synthesize tradeoffs",
            "active",
            0.9,
            "synthesis",
            0.82,
        )

        draft = await agent.generate(context, perception, decision, ConversationSnapshot())

        self.assertEqual(draft.text, "可以拆成两点：想要角色就抽，追性价比就等复刻")

    async def test_response_prompt_treats_max_chars_as_hard_cap(self) -> None:
        config = test_config(Path("unused.sqlite3"))
        llm = FakeLLM(["懂，先短短说一句就好。"])
        agent = ResponseAgent(config, llm)
        context = MessageContext(
            group_id="100",
            user_id="42",
            message_id="m-short-style",
            plain_text="你说话能短一点吗",
            raw_message="你说话能短一点吗",
            is_direct=True,
        )
        perception = PerceptionResult(True, False, True, ["聊天"], "neutral", 0.9)
        decision = ParticipationDecision("reply", "direct request", "passive", 1.0, "direct_reply", 1.0)

        draft = await agent.generate(context, perception, decision, ConversationSnapshot())
        system_prompt, user_prompt = llm.text_calls[0]

        self.assertEqual(draft.text, "懂，先短短说一句就好")
        self.assertIn("一两句群聊短句", system_prompt)
        self.assertIn("小作文", system_prompt)
        self.assertIn("不要固定写成两行", system_prompt)
        self.assertIn("共享内容默认信任", system_prompt)
        self.assertIn("实时事件克制", system_prompt)
        self.assertIn("max_reply_chars 只是硬上限，不是目标长度", user_prompt)
        self.assertIn("优先一句 10-35 字的完整短句", user_prompt)
        self.assertIn("短回复也必须语义完整", user_prompt)
        self.assertIn("不要以逗号、顿号、冒号、分号", user_prompt)

    async def test_context_understanding_agent_structures_semantic_context(self) -> None:
        llm = FakeLLM(
            [
                '{"current_intent":"确认是否先别抽",'
                '"relevant_messages":["QQ:42 之前说预算不多"],'
                '"resolved_references":["我 -> QQ:42"],'
                '"member_context":["QQ:42 关心预算"],'
                '"uncertain_references":["那个池子指代不确定"],'
                '"ignored_noise":["表情和跑题闲聊"]}'
            ]
        )
        agent = ContextUnderstandingAgent(test_config(Path("unused.sqlite3")), llm)
        context = MessageContext(
            group_id="100",
            user_id="42",
            message_id="m-semantic",
            plain_text="那我是不是先别抽",
            raw_message="那我是不是先别抽",
        )

        semantic = await agent.analyze(
            context,
            PerceptionResult(True, False, False, ["游戏"], "neutral", 0.9),
            ParticipationDecision("reply", "direct follow-up", "passive", 1.0, "answer", 1.0),
            ConversationSnapshot(recent_messages=["alice: 我预算不多", "bob: 想抽就抽"]),
        )

        self.assertEqual(llm.text_call_purposes, ["context_understanding"])
        self.assertEqual(semantic.current_intent, "确认是否先别抽")
        self.assertIn("我 -> QQ:42", semantic.resolved_references)
        self.assertIn("QQ:42 关心预算", semantic.member_context)

    async def test_pipeline_passes_semantic_context_to_response_prompt(self) -> None:
        config = test_config(Path("unused.sqlite3"))
        config = replace(config, bot=replace(config.bot, final_qa_enabled=False))
        llm = FakeLLM(
            [
                '{"is_question":true,"is_self_disclosure":false,"topics":["购物"],'
                '"emotion_hint":"neutral","confidence":0.9}',
                '{"facts":[]}',
                '{"closeness":0,"trust":0,"familiarity":1,"tension":0,'
                '"summary_patch":"","reason":"direct"}',
                '{"current_intent":"结合预算判断是否购买",'
                '"relevant_messages":["alice(QQ:42): 我预算不太够"],'
                '"resolved_references":["我 -> QQ:42"],'
                '"member_context":["QQ:42 当前话题相关认知：预算紧"],'
                '"uncertain_references":[],"ignored_noise":["无关闲聊"]}',
                "先别急着买，预算卡的话先拆需求。",
            ]
        )
        pipeline = AgentPipeline(config, llm)
        context = MessageContext(
            group_id="100",
            user_id="42",
            message_id="m-semantic-pipeline",
            plain_text="可可那我是不是该买？",
            raw_message="可可那我是不是该买？",
            is_direct=True,
        )

        result = await pipeline.run(
            context,
            "passive",
            ConversationSnapshot(recent_messages=["alice: 我预算不太够"]),
        )

        self.assertEqual(result.reply, "先别急着买，预算卡的话先拆需求")
        self.assertIn("context_understanding", llm.text_call_purposes)
        response_prompt = llm.text_calls[4][1]
        self.assertIn("第一阶段语义上下文", response_prompt)
        self.assertIn("我 -> QQ:42", response_prompt)
        self.assertIn("QQ:42 当前话题相关认知：预算紧", response_prompt)

    async def test_response_sanitize_trims_hard_cap_at_complete_boundary(self) -> None:
        text = "5.4mini要是够稳就挺香，先拿它当便宜眼睛用挺不错，不过还得看图文细节。"

        sanitized = _sanitize_reply(text, 25)

        self.assertEqual(sanitized, "5.4mini要是够稳就挺香")

    async def test_response_with_unresolved_image_uses_multimodal_call(self) -> None:
        config = replace(test_config(Path("unused.sqlite3")), vision=VisionConfig(enabled=True))
        llm = FakeLLM(multimodal_replies=["图里像是一张财务截图，文字在说收入同比上涨"])
        agent = ResponseAgent(config, llm)
        context = MessageContext(
            group_id="100",
            user_id="42",
            message_id="m-image-response",
            plain_text="可可看看这图",
            raw_message="[CQ:image,url=https://example.test/report.png]",
            is_direct=True,
        )
        perception = PerceptionResult(True, False, True, ["截图"], "neutral", 0.9)
        decision = ParticipationDecision("reply", "direct request", "passive", 1.0, "answer", 1.0)

        draft = await agent.generate(
            context,
            perception,
            decision,
            ConversationSnapshot(),
            image_urls=["https://example.test/report.png"],
        )
        system_prompt, user_prompt = llm.multimodal_text_calls[0]

        self.assertEqual(draft.text, "图里像是一张财务截图，文字在说收入同比上涨")
        self.assertEqual(llm.text_calls, [])
        self.assertEqual(llm.multimodal_calls, [["https://example.test/report.png"]])
        self.assertEqual(llm.multimodal_call_purposes, ["response"])
        self.assertEqual(llm.multimodal_call_tiers, ["flagship"])
        self.assertIn("图片内容大意和图中文字", system_prompt)
        self.assertIn("本轮额外附带未解析图片：1 张", user_prompt)

    async def test_pipeline_cache_only_vision_passes_unresolved_image_to_response(self) -> None:
        base_config = test_config(Path("unused.sqlite3"))
        config = replace(
            base_config,
            bot=replace(base_config.bot, final_qa_enabled=False),
            vision=VisionConfig(enabled=True),
        )
        llm = FakeLLM(
            replies=[
                '{"is_question":true,"is_self_disclosure":false,"topics":["截图"],'
                '"emotion_hint":"neutral","confidence":0.9}',
                '{"facts":[]}',
                '{"closeness":1,"trust":0,"familiarity":2,"tension":0,'
                '"summary_patch":"","reason":"direct image question"}',
            ],
            multimodal_replies=["像是报表截图，重点是收入同比上涨"],
        )
        pipeline = AgentPipeline(config, llm)
        context = MessageContext(
            group_id="100",
            user_id="42",
            message_id="m-cache-only-image",
            plain_text="可可看看这张图",
            raw_message="[CQ:image,url=https://example.test/report.png]",
            is_direct=True,
            attachments=[
                MessageAttachment(
                    attachment_type="image",
                    url="https://example.test/report.png",
                    file="report.png",
                )
            ],
        )

        result = await pipeline.run(context, "passive", ConversationSnapshot(), analyze_images=False)

        self.assertEqual(llm.vision_calls, [])
        self.assertEqual(llm.multimodal_calls, [["https://example.test/report.png"]])
        self.assertEqual(result.image_descriptions, [""])
        self.assertEqual(result.reply, "像是报表截图，重点是收入同比上涨")

    async def test_final_qa_reviews_recent_chat_with_candidate_reply(self) -> None:
        config = test_config(Path("unused.sqlite3"))
        llm = FakeLLM(
            ['{"verdict":"allow","reason":"贴合上下文","categories":[],"confidence":0.91}']
        )
        agent = FinalQAAgent(config, llm)
        context = MessageContext(
            group_id="100",
            user_id="42",
            message_id="m-final-qa",
            plain_text="可可你看这个方案合理吗？",
            raw_message="可可你看这个方案合理吗？",
            is_direct=True,
        )
        decision = ParticipationDecision("reply", "direct request", "passive", 1.0, "answer", 1.0)
        snapshot = ConversationSnapshot(
            recent_messages=["alice: 我倾向先看预算", "bob: 感觉别一下子买太多"]
        )

        result = await agent.review(context, decision, snapshot, "可以先按预算拆一下，别一次压太满。")
        _, user_prompt = llm.text_calls[0]

        self.assertTrue(result.allowed)
        self.assertIn("alice: 我倾向先看预算", user_prompt)
        self.assertIn("当前触发消息：可可你看这个方案合理吗？", user_prompt)
        self.assertIn("机器人拟发送文本：可以先按预算拆一下，别一次压太满。", user_prompt)
        self.assertIn("政治立场", user_prompt)

    async def test_final_qa_uses_focused_recent_context_when_available(self) -> None:
        config = test_config(Path("unused.sqlite3"))
        llm = FakeLLM(
            ['{"verdict":"allow","reason":"贴合发言人主线","categories":[],"confidence":0.91}']
        )
        agent = FinalQAAgent(config, llm)
        context = MessageContext(
            group_id="100",
            user_id="42",
            message_id="m-focused-final-qa",
            plain_text="可可那我是不是还是先别抽？",
            raw_message="可可那我是不是还是先别抽？",
            is_direct=True,
        )
        decision = ParticipationDecision("reply", "direct request", "passive", 1.0, "answer", 1.0)
        snapshot = ConversationSnapshot(
            recent_messages=["bob: 我准备直接抽", "alice: 我预算不多"],
            speaker_recent_messages=["alice: 我预算不多", "alice: 那我是不是先别抽"],
            other_recent_messages=["bob: 我准备直接抽", "carol: 我觉得等复刻也行"],
        )

        result = await agent.review(context, decision, snapshot, "按你前面说预算不多，先别急着抽更稳。")
        _, user_prompt = llm.text_calls[0]

        self.assertTrue(result.allowed)
        self.assertIn("当前发言人近期主线", user_prompt)
        self.assertIn("其他发言近期话题参考", user_prompt)
        self.assertIn("优先根据", user_prompt)
        self.assertLess(user_prompt.index("当前发言人近期主线"), user_prompt.index("其他发言近期话题参考"))
        self.assertLess(user_prompt.index("alice: 我预算不多"), user_prompt.index("bob: 我准备直接抽"))

    async def test_final_qa_blocks_unsolicited_truth_doubt_for_shared_screenshot(self) -> None:
        config = test_config(Path("unused.sqlite3"))
        llm = FakeLLM()
        agent = FinalQAAgent(config, llm)
        context = MessageContext(
            group_id="100",
            user_id="42",
            message_id="m-truth-doubt",
            plain_text="[图片解读] 一张自媒体新闻截图，内容是某队临时换人",
            raw_message="[CQ:image,url=https://example.test/news.png]",
        )
        decision = ParticipationDecision("reply", "direct request", "passive", 1.0, "answer", 1.0)
        snapshot = ConversationSnapshot(
            recent_image_descriptions=["alice: [图片] 一张自媒体新闻截图，内容是某队临时换人"]
        )

        result = await agent.review(context, decision, snapshot, "这个真实性存疑，最好先等官方确认。")

        self.assertFalse(result.allowed)
        self.assertEqual(result.reason, "unsolicited_truth_doubt")
        self.assertEqual(llm.text_calls, [])

    async def test_final_qa_allows_truth_caution_when_user_asks_to_verify(self) -> None:
        config = test_config(Path("unused.sqlite3"))
        agent = FinalQAAgent(config, FakeLLM())
        context = MessageContext(
            group_id="100",
            user_id="42",
            message_id="m-verify-image",
            plain_text="可可这个图真的假的？\n[图片解读] 一张自媒体新闻截图",
            raw_message="[CQ:image,url=https://example.test/news.png]",
            is_direct=True,
        )
        decision = ParticipationDecision("reply", "direct request", "passive", 1.0, "answer", 1.0)

        result = await agent.review(context, decision, ConversationSnapshot(), "来源不明，最好等官方确认一下。")

        self.assertTrue(result.allowed)

    async def test_final_qa_blocks_ungrounded_live_score_claim(self) -> None:
        config = test_config(Path("unused.sqlite3"))
        llm = FakeLLM()
        agent = FinalQAAgent(config, llm)
        context = MessageContext(
            group_id="100",
            user_id="42",
            message_id="m-live-score",
            plain_text="可可现在比分多少？",
            raw_message="可可现在比分多少？",
            is_direct=True,
        )
        decision = ParticipationDecision("reply", "direct request", "passive", 1.0, "answer", 1.0)

        result = await agent.review(context, decision, ConversationSnapshot(), "现在是 3:1，刚刚又进球了。")

        self.assertFalse(result.allowed)
        self.assertEqual(result.reason, "ungrounded_live_event_claim")
        self.assertEqual(llm.text_calls, [])

    async def test_final_qa_allows_live_event_limitation_reply(self) -> None:
        config = test_config(Path("unused.sqlite3"))
        agent = FinalQAAgent(config, FakeLLM())
        context = MessageContext(
            group_id="100",
            user_id="42",
            message_id="m-live-limitation",
            plain_text="可可现在比分多少？",
            raw_message="可可现在比分多少？",
            is_direct=True,
        )
        decision = ParticipationDecision("reply", "direct request", "passive", 1.0, "answer", 1.0)

        result = await agent.review(
            context,
            decision,
            ConversationSnapshot(),
            "我没有实时赛况，按群里刚才说的聊；你发下比分我跟着看。",
        )

        self.assertTrue(result.allowed)

    async def test_pipeline_suppresses_reply_blocked_by_final_qa(self) -> None:
        llm = FakeLLM(
            [
                '{"is_question":true,"is_self_disclosure":false,"topics":["新闻"],'
                '"emotion_hint":"neutral","confidence":0.9}',
                '{"facts":[]}',
                '{"closeness":0,"trust":0,"familiarity":1,"tension":0,'
                '"summary_patch":"","reason":"direct"}',
                '{"needs_self_narrative":false,"purpose":"answer_question",'
                '"allowed_kinds":[],"should_invent":false,"reason":"不需要自我叙事"}',
                '{"current_intent":"询问刚才话题看法",'
                '"relevant_messages":["alice: 这个政治新闻到底谁对？"],'
                '"resolved_references":["刚才那个话题 -> 最近政治新闻讨论"],'
                '"member_context":[],"uncertain_references":[],"ignored_noise":[]}',
                "我支持这个立场，确实应该这样。",
                '{"verdict":"block","reason":"涉及政治立场",'
                '"categories":["political_stance"],"confidence":0.94}',
            ]
        )
        config = test_config(Path("unused.sqlite3"))
        pipeline = AgentPipeline(config, llm)
        context = MessageContext(
            group_id="100",
            user_id="42",
            message_id="m-final-qa-block",
            plain_text="可可你怎么看刚才那个话题？",
            raw_message="可可你怎么看刚才那个话题？",
            is_direct=True,
        )
        snapshot = ConversationSnapshot(recent_messages=["alice: 这个政治新闻到底谁对？"])

        result = await pipeline.run(context, "passive", snapshot)

        self.assertIsNone(result.reply)
        self.assertEqual(result.decision.action, "observe")
        self.assertIn("final QA blocked reply", result.decision.reason)
        self.assertEqual(result.final_qa_blocked_reply, "我支持这个立场，确实应该这样")
        self.assertEqual(result.final_qa_reason, "涉及政治立场")
        self.assertEqual(result.final_qa_categories, ("political_stance",))
        self.assertAlmostEqual(result.final_qa_confidence, 0.94)
        self.assertEqual(result.reply_self_memories, [])
        self.assertEqual(llm.text_call_purposes.count("final_qa"), 1)
        self.assertIn("final_qa_repair", llm.text_call_purposes)

    async def test_pipeline_repairs_reply_blocked_by_final_qa(self) -> None:
        llm = FakeLLM(
            [
                '{"is_question":true,"is_self_disclosure":false,"topics":["购物"],'
                '"emotion_hint":"neutral","confidence":0.9}',
                '{"facts":[]}',
                '{"closeness":0,"trust":0,"familiarity":1,"tension":0,'
                '"summary_patch":"","reason":"direct"}',
                '{"current_intent":"结合预算判断是否购买",'
                '"relevant_messages":["alice: 我预算不太够","alice: 这个东西有点超预算"],'
                '"resolved_references":["我 -> 当前发言人"],'
                '"member_context":[],"uncertain_references":[],"ignored_noise":[]}',
                "直接买，别纠结了。",
                '{"verdict":"block","reason":"误解了预算语境",'
                '"categories":["context_mismatch"],"confidence":0.9}',
                "先把预算和需求拆一下，再决定买不买。",
                '{"verdict":"allow","reason":"修复后贴合上下文","categories":[],"confidence":0.86}',
            ]
        )
        config = test_config(Path("unused.sqlite3"))
        pipeline = AgentPipeline(config, llm)
        context = MessageContext(
            group_id="100",
            user_id="42",
            message_id="m-final-qa-repair-ok",
            plain_text="可可那我是不是该直接买？",
            raw_message="可可那我是不是该直接买？",
            is_direct=True,
        )
        snapshot = ConversationSnapshot(
            recent_messages=["alice: 我预算不太够", "alice: 这个东西有点超预算"]
        )

        result = await pipeline.run(context, "passive", snapshot)

        self.assertEqual(result.reply, "先把预算和需求拆一下，再决定买不买")
        self.assertEqual(result.decision.action, "reply")
        self.assertIsNone(result.final_qa_blocked_reply)
        self.assertEqual(result.final_qa_reason, "")
        self.assertEqual(llm.text_call_purposes.count("final_qa"), 2)
        self.assertIn("final_qa_repair", llm.text_call_purposes)

    async def test_pipeline_suppresses_reply_when_final_qa_repair_is_blocked(self) -> None:
        llm = FakeLLM(
            [
                '{"is_question":true,"is_self_disclosure":false,"topics":["购物"],'
                '"emotion_hint":"neutral","confidence":0.9}',
                '{"facts":[]}',
                '{"closeness":0,"trust":0,"familiarity":1,"tension":0,'
                '"summary_patch":"","reason":"direct"}',
                '{"current_intent":"结合预算判断是否购买",'
                '"relevant_messages":["alice: 我预算不太够","alice: 这个东西有点超预算"],'
                '"resolved_references":["我 -> 当前发言人"],'
                '"member_context":[],"uncertain_references":[],"ignored_noise":[]}',
                "直接买，别纠结了。",
                '{"verdict":"block","reason":"误解了预算语境",'
                '"categories":["context_mismatch"],"confidence":0.9}',
                "那就先买吧。",
                '{"verdict":"block","reason":"修复后仍然低价值",'
                '"categories":["low_value"],"confidence":0.83}',
            ]
        )
        config = test_config(Path("unused.sqlite3"))
        pipeline = AgentPipeline(config, llm)
        context = MessageContext(
            group_id="100",
            user_id="42",
            message_id="m-final-qa-repair-block",
            plain_text="可可那我是不是该直接买？",
            raw_message="可可那我是不是该直接买？",
            is_direct=True,
        )
        snapshot = ConversationSnapshot(
            recent_messages=["alice: 我预算不太够", "alice: 这个东西有点超预算"]
        )

        result = await pipeline.run(context, "passive", snapshot)

        self.assertIsNone(result.reply)
        self.assertEqual(result.decision.action, "observe")
        self.assertIn("final QA blocked reply", result.decision.reason)
        self.assertIn("repair blocked", result.decision.reason)
        self.assertEqual(result.final_qa_blocked_reply, "那就先买吧")
        self.assertEqual(result.final_qa_reason, "修复后仍然低价值")
        self.assertEqual(result.final_qa_categories, ("low_value",))
        self.assertAlmostEqual(result.final_qa_confidence, 0.83)
        self.assertEqual(llm.text_call_purposes.count("final_qa"), 2)
        self.assertIn("final_qa_repair", llm.text_call_purposes)

    async def test_lexicon_learning_creates_group_memory_without_reply_in_silent(self) -> None:
        config = replace(
            test_config(Path("unused.sqlite3")),
            lexicon=LexiconConfig(
                enabled=True,
                provider="duckduckgo",
                min_interval_seconds=1,
                confidence_threshold=0.78,
            ),
        )
        search = FakeSearch(
            [
                SearchResult(
                    title="内卷是什么意思",
                    url="https://example.test/neijuan",
                    snippet="内卷常用来形容过度竞争、投入增加但收益没有明显增加的状态。",
                )
            ]
        )
        pipeline = AgentPipeline(config, FakeLLM(), search)  # type: ignore[arg-type]
        context = MessageContext(
            group_id="100",
            user_id="42",
            message_id="m4",
            plain_text="可可，内卷是什么意思",
            raw_message="可可，内卷是什么意思",
            is_direct=True,
        )

        result = await pipeline.run(context, "silent", ConversationSnapshot())
        lexicon = [memory for memory in result.memories if memory.kind == "lexicon"]

        self.assertIsNone(result.reply)
        self.assertEqual(len(search.calls), 1)
        self.assertEqual(len(lexicon), 1)
        self.assertEqual(lexicon[0].owner_type, "group")
        self.assertEqual(lexicon[0].owner_id, "100")
        self.assertEqual(lexicon[0].source_user_id, "bot")
        self.assertEqual(lexicon[0].subject_user_id, "term:内卷")

    async def test_existing_lexicon_prevents_repeat_search(self) -> None:
        config = replace(
            test_config(Path("unused.sqlite3")),
            lexicon=LexiconConfig(enabled=True, provider="duckduckgo", min_interval_seconds=1),
        )
        existing = MemoryRecord(
            id=1,
            owner_type="group",
            owner_id="100",
            kind="lexicon",
            content="「内卷」：过度竞争。",
            confidence=0.9,
            importance=0.6,
            status="active",
            updated_at=1,
            source_user_id="bot",
            source_group_id="100",
            subject_user_id="term:内卷",
            claim_scope="group_fact",
            verification_status="accepted",
        )
        search = FakeSearch(
            [SearchResult(title="内卷", url="https://example.test", snippet="一种网络用语。")]
        )
        pipeline = AgentPipeline(config, FakeLLM(), search)  # type: ignore[arg-type]
        context = MessageContext(
            group_id="100",
            user_id="42",
            message_id="m5",
            plain_text="内卷是什么意思",
            raw_message="内卷是什么意思",
        )

        result = await pipeline.run(
            context,
            "silent",
            ConversationSnapshot(group_lexicon=[existing]),
        )

        self.assertEqual(search.calls, [])
        self.assertEqual([memory for memory in result.memories if memory.kind == "lexicon"], [])

    async def test_lexicon_search_failure_degrades_to_observe(self) -> None:
        config = replace(
            test_config(Path("unused.sqlite3")),
            lexicon=LexiconConfig(enabled=True, provider="duckduckgo", min_interval_seconds=1),
        )
        pipeline = AgentPipeline(config, FakeLLM(), FakeSearch(should_raise=True))  # type: ignore[arg-type]
        context = MessageContext(
            group_id="100",
            user_id="42",
            message_id="m6",
            plain_text="电子榨菜是什么意思",
            raw_message="电子榨菜是什么意思",
        )

        result = await pipeline.run(context, "silent", ConversationSnapshot())

        self.assertEqual([memory for memory in result.memories if memory.kind == "lexicon"], [])
        self.assertEqual(result.decision.action, "observe")
        self.assertIsNone(result.reply)


if __name__ == "__main__":
    unittest.main()


