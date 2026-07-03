from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from qq_llm_bot.cognitive_agents import (
    AgentPipeline,
    FactExtractorAgent,
    RelationshipAgent,
    StickerSelectorAgent,
    VisionAgent,
)
from qq_llm_bot.cognitive_storage import BotStorage
from qq_llm_bot.config import (
    AppConfig,
    BotConfig,
    LexiconConfig,
    LLMConfig,
    NapCatConfig,
    PersonaConfig,
    ReflectionConfig,
    StorageConfig,
    StickerConfig,
    VisionConfig,
)
from qq_llm_bot.models import (
    ConversationSnapshot,
    FactCandidate,
    MemoryCandidate,
    MemoryRecord,
    MessageAttachment,
    MessageContext,
    PerceptionResult,
    RelationDelta,
    ParticipationDecision,
    StickerAssetRecord,
    StickerCandidate,
    UserProfileDraft,
)
from qq_llm_bot.stickers import StickerLocalStore
from qq_llm_bot.web_search import SearchResult


class FakeLLM:
    def __init__(
        self,
        replies: list[str] | None = None,
        vision_replies: list[str] | None = None,
    ) -> None:
        self.replies = replies or []
        self.vision_replies = vision_replies or []
        self.vision_calls: list[list[str]] = []

    async def complete_text(self, system_prompt: str, user_prompt: str) -> str | None:
        if self.replies:
            return self.replies.pop(0)
        return None

    async def complete_vision(
        self,
        system_prompt: str,
        user_prompt: str,
        image_urls: list[str],
        vision_config: VisionConfig,
    ) -> str | None:
        self.vision_calls.append(image_urls)
        if self.vision_replies:
            return self.vision_replies.pop(0)
        return None


class FakeSearch:
    def __init__(
        self,
        results: list[SearchResult] | None = None,
        should_raise: bool = False,
    ) -> None:
        self.results = results or []
        self.should_raise = should_raise
        self.calls: list[str] = []

    async def search(self, query: str, max_results: int | None = None) -> list[SearchResult]:
        self.calls.append(query)
        if self.should_raise:
            raise RuntimeError("search unavailable")
        return self.results[: max_results or len(self.results)]


def test_config(db_path: Path) -> AppConfig:
    return AppConfig(
        napcat=NapCatConfig(ws_url="ws://example.test"),
        bot=BotConfig(
            nicknames=["可可"],
            admin_ids=["1"],
            enabled_groups=["100"],
            default_group_mode="passive",
            proactive_cooldown_seconds=90,
            max_reply_chars=80,
        ),
        persona=PersonaConfig(self_name="可可"),
        reflection=ReflectionConfig(enabled=True, message_threshold=30),
        storage=StorageConfig(sqlite_path=str(db_path)),
        llm=LLMConfig(provider="disabled"),
        project_root=db_path.parent,
    )


class CognitiveLoopTests(unittest.IsolatedAsyncioTestCase):
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
        with tempfile.TemporaryDirectory() as tmp:
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


class MemoryStorageTests(unittest.TestCase):
    def test_sticker_asset_is_saved_and_can_be_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = test_config(Path(tmp) / "bot.sqlite3")
            storage = BotStorage.from_config(config)
            storage.setup()
            local_path = str(Path(tmp) / "meme.png")
            context = MessageContext(
                group_id="100",
                user_id="42",
                message_id="m-sticker",
                plain_text="",
                raw_message="[CQ:image]",
            )
            candidate = StickerCandidate(
                url="https://example.test/meme.png",
                file="meme.png",
                description="一张猫猫下班表情包",
                ocr_text="下班了",
                mood="疲惫",
                usage="适合聊到下班、犯困或想摆一下时使用",
                tags=("下班", "困", "猫猫"),
                confidence=0.86,
            )

            asset = storage.upsert_sticker_asset(context, candidate, local_path=local_path, sha256="abc")
            active = storage.list_sticker_assets("100")

            self.assertIsNotNone(asset)
            self.assertEqual(active[0].usage, "适合聊到下班、犯困或想摆一下时使用")
            self.assertEqual(active[0].tags, ("下班", "困", "猫猫"))

            self.assertTrue(storage.set_sticker_enabled(active[0].id, False))
            self.assertEqual(storage.list_sticker_assets("100"), [])
            self.assertEqual(len(storage.list_sticker_assets("100", enabled_only=False)), 1)

    def test_dashboard_stickers_include_delete_command(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = test_config(Path(tmp) / "bot.sqlite3")
            storage = BotStorage.from_config(config)
            storage.setup()
            local_path = str(Path(tmp) / "data" / "stickers" / "100" / "meme.png")
            context = MessageContext(
                group_id="100",
                user_id="42",
                message_id="m-sticker",
                plain_text="",
                raw_message="[CQ:image]",
            )
            candidate = StickerCandidate(
                url="https://example.test/meme.png",
                file="meme.png",
                description="一张猫猫下班表情包",
                mood="疲惫",
                usage="适合聊到下班、犯困或想摆一下时使用",
                tags=("下班", "困"),
                confidence=0.86,
            )

            asset = storage.upsert_sticker_asset(context, candidate, local_path=local_path, sha256="abc")
            items = storage.list_dashboard_stickers("100")

            self.assertIsNotNone(asset)
            self.assertEqual(items[0]["trigger"], "适合聊到下班、犯困或想摆一下时使用")
            self.assertEqual(items[0]["delete_command"], f"#bot stickers delete {asset.id}")
            self.assertIn("100", storage.list_dashboard_groups())

    def test_delete_sticker_asset_and_local_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = test_config(Path(tmp) / "bot.sqlite3")
            store = StickerLocalStore(config)
            sticker_dir = config.resolve_path(config.stickers.storage_dir) / "100"
            sticker_dir.mkdir(parents=True)
            sticker_file = sticker_dir / "meme.png"
            sticker_file.write_bytes(b"fake image")

            storage = BotStorage.from_config(config)
            storage.setup()
            context = MessageContext(
                group_id="100",
                user_id="42",
                message_id="m-sticker",
                plain_text="",
                raw_message="[CQ:image]",
            )
            candidate = StickerCandidate(
                url="https://example.test/meme.png",
                file="meme.png",
                description="一张猫猫下班表情包",
                mood="疲惫",
                usage="适合聊到下班、犯困或想摆一下时使用",
                tags=("下班",),
                confidence=0.86,
            )
            asset = storage.upsert_sticker_asset(
                context,
                candidate,
                local_path=str(sticker_file),
                sha256="abc",
            )

            deleted_asset = storage.delete_sticker_asset(asset.id if asset else 0)
            deleted_file = store.delete_saved_file(deleted_asset.local_path if deleted_asset else "")

            self.assertIsNotNone(deleted_asset)
            self.assertTrue(deleted_file)
            self.assertFalse(sticker_file.exists())
            self.assertEqual(storage.list_sticker_assets("100", enabled_only=False), [])

    def test_bot_sourced_lexicon_group_fact_is_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = test_config(Path(tmp) / "bot.sqlite3")
            storage = BotStorage.from_config(config)
            storage.setup()

            memory = MemoryCandidate(
                owner_type="group",
                owner_id="100",
                kind="lexicon",
                content="「内卷」：形容过度竞争。",
                confidence=0.84,
                importance=0.58,
                evidence_message_id="m1",
                source_user_id="bot",
                source_group_id="100",
                subject_user_id="term:内卷",
                claim_scope="group_fact",
            )

            write = storage.record_memory_candidates([memory])
            active = storage.list_group_lexicon_records("100")

            self.assertEqual(len(write.accepted), 1)
            self.assertEqual(active[0].content, "「内卷」：形容过度竞争。")
            self.assertTrue(storage.has_group_lexicon("100", "内卷"))

    def test_lexicon_duplicate_uses_term_subject_not_exact_content(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = test_config(Path(tmp) / "bot.sqlite3")
            storage = BotStorage.from_config(config)
            storage.setup()

            first = MemoryCandidate(
                owner_type="group",
                owner_id="100",
                kind="lexicon",
                content="「内卷」：形容过度竞争。",
                confidence=0.84,
                importance=0.58,
                evidence_message_id="m1",
                source_user_id="bot",
                source_group_id="100",
                subject_user_id="term:内卷",
                claim_scope="group_fact",
            )
            second = MemoryCandidate(
                owner_type="group",
                owner_id="100",
                kind="lexicon",
                content="「内卷」：网络语境里的竞争加剧。",
                confidence=0.9,
                importance=0.6,
                evidence_message_id="m2",
                source_user_id="bot",
                source_group_id="100",
                subject_user_id="term:内卷",
                claim_scope="group_fact",
            )

            storage.record_memory_candidates([first])
            storage.record_memory_candidates([second])
            active = storage.list_group_lexicon_records("100")

            self.assertEqual(len(active), 1)
            self.assertEqual(active[0].content, "「内卷」：形容过度竞争。")
            self.assertEqual(active[0].confidence, 0.9)

    def test_multiple_self_past_events_can_coexist(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = test_config(Path(tmp) / "bot.sqlite3")
            storage = BotStorage.from_config(config)
            storage.setup()

            first = MemoryCandidate(
                owner_type="self",
                owner_id="bot",
                kind="self_past_event",
                content="我以前也有过一阵子特别容易想太多",
                confidence=0.86,
                importance=0.55,
                evidence_message_id="m1",
                source_user_id="bot",
                source_group_id="100",
                subject_user_id="bot",
                claim_scope="bot_directed",
            )
            second = MemoryCandidate(
                owner_type="self",
                owner_id="bot",
                kind="self_past_event",
                content="我以前试着养过一盆薄荷",
                confidence=0.86,
                importance=0.55,
                evidence_message_id="m2",
                source_user_id="bot",
                source_group_id="100",
                subject_user_id="bot",
                claim_scope="bot_directed",
            )

            write = storage.record_memory_candidates([first, second])
            active = storage.list_memories("self", "bot", limit=10)

            self.assertEqual(len(write.accepted), 2)
            self.assertEqual(len(active), 2)

    def test_self_preference_conflict_is_marked_without_overwriting(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = test_config(Path(tmp) / "bot.sqlite3")
            storage = BotStorage.from_config(config)
            storage.setup()

            first = MemoryCandidate(
                owner_type="self",
                owner_id="bot",
                kind="self_preference",
                content="我喜欢安静一点的雨天",
                confidence=0.86,
                importance=0.6,
                evidence_message_id="m1",
                source_user_id="bot",
                source_group_id="100",
                subject_user_id="bot",
                claim_scope="bot_directed",
            )
            second = MemoryCandidate(
                owner_type="self",
                owner_id="bot",
                kind="self_preference",
                content="我不喜欢雨天",
                confidence=0.88,
                importance=0.6,
                evidence_message_id="m2",
                source_user_id="bot",
                source_group_id="100",
                subject_user_id="bot",
                claim_scope="bot_directed",
            )

            storage.record_memory_candidates([first])
            write = storage.record_memory_candidates([second])

            self.assertEqual(len(write.conflicts), 1)
            self.assertEqual(
                storage.list_memories("self", "bot", status="active")[0].content,
                "我喜欢安静一点的雨天",
            )
            self.assertEqual(
                storage.list_memories("self", "bot", status="conflict")[0].content,
                "我不喜欢雨天",
            )

    def test_different_self_preferences_can_coexist(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = test_config(Path(tmp) / "bot.sqlite3")
            storage = BotStorage.from_config(config)
            storage.setup()

            first = MemoryCandidate(
                owner_type="self",
                owner_id="bot",
                kind="self_preference",
                content="我喜欢海边潮湿的风和声音",
                confidence=0.86,
                importance=0.6,
                evidence_message_id="m1",
                source_user_id="bot",
                source_group_id="100",
                subject_user_id="bot",
                claim_scope="bot_directed",
            )
            second = MemoryCandidate(
                owner_type="self",
                owner_id="bot",
                kind="self_preference",
                content="我喜欢安静一点的雨天",
                confidence=0.88,
                importance=0.6,
                evidence_message_id="m2",
                source_user_id="bot",
                source_group_id="100",
                subject_user_id="bot",
                claim_scope="bot_directed",
            )

            write = storage.record_memory_candidates([first, second])

            self.assertEqual(len(write.accepted), 2)
            self.assertEqual(len(storage.list_memories("self", "bot", status="active")), 2)

    def test_self_report_fact_is_accepted_for_speaker(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = test_config(Path(tmp) / "bot.sqlite3")
            storage = BotStorage.from_config(config)
            storage.setup()

            fact = FactCandidate(
                subject_user_id="42",
                fact_type="preference",
                claim_text="用户42喜欢海边",
                topic="海边",
                stance="positive",
                confidence=0.82,
                evidence_message_id="m1",
                evidence_text="我喜欢海边",
                source_user_id="42",
                source_group_id="100",
                claim_scope="self_report",
            )

            write = storage.record_fact_candidates([fact])
            accepted_facts = storage.list_user_facts("42")

            self.assertEqual(len(write.accepted), 1)
            self.assertEqual(accepted_facts[0].claim_text, "用户42喜欢海边")
            self.assertEqual(accepted_facts[0].claim_scope, "self_report")
            self.assertEqual(accepted_facts[0].status, "accepted")
            self.assertEqual(storage.list_memories("user", "42", status="active"), [])

    def test_low_value_fact_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = test_config(Path(tmp) / "bot.sqlite3")
            storage = BotStorage.from_config(config)
            storage.setup()

            fact = FactCandidate(
                subject_user_id="42",
                fact_type="other",
                claim_text="用户42继续聊比赛感受",
                topic="比赛感受",
                stance="neutral",
                confidence=0.9,
                evidence_message_id="m-low",
                evidence_text="继续聊比赛感受",
                source_user_id="42",
                source_group_id="100",
                claim_scope="self_report",
            )

            write = storage.record_fact_candidates([fact])

            self.assertEqual(write.accepted, [])
            self.assertEqual(len(write.rejected), 1)
            self.assertEqual(storage.list_user_facts("42"), [])

    def test_low_trust_third_party_fact_is_pending_not_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = test_config(Path(tmp) / "bot.sqlite3")
            storage = BotStorage.from_config(config)
            storage.setup()

            fact = FactCandidate(
                subject_user_id="77",
                fact_type="preference",
                claim_text="用户77喜欢吃鱼",
                topic="吃鱼",
                stance="positive",
                confidence=0.9,
                evidence_message_id="m1",
                evidence_text="77喜欢吃鱼",
                source_user_id="42",
                source_group_id="100",
                claim_scope="third_party",
            )

            write = storage.record_fact_candidates([fact])

            self.assertEqual(write.accepted, [])
            self.assertEqual(len(write.pending), 1)
            self.assertEqual(storage.list_user_facts("77"), [])
            self.assertEqual(storage.list_user_facts("77", status="pending_confirmation")[0].claim_text, "用户77喜欢吃鱼")

    def test_high_trust_third_party_fact_is_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = test_config(Path(tmp) / "bot.sqlite3")
            storage = BotStorage.from_config(config)
            storage.setup()
            storage.apply_relationship_delta("100", "42", RelationDelta(trust=75))

            fact = FactCandidate(
                subject_user_id="77",
                fact_type="preference",
                claim_text="用户77喜欢吃鱼",
                topic="吃鱼",
                stance="positive",
                confidence=0.9,
                evidence_message_id="m1",
                evidence_text="77喜欢吃鱼",
                source_user_id="42",
                source_group_id="100",
                claim_scope="third_party",
            )

            write = storage.record_fact_candidates([fact])

            self.assertEqual(len(write.accepted), 1)
            self.assertEqual(storage.list_user_facts("77")[0].claim_text, "用户77喜欢吃鱼")

    def test_profile_updates_after_five_accepted_facts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = test_config(Path(tmp) / "bot.sqlite3")
            storage = BotStorage.from_config(config)
            storage.setup()
            facts = [
                FactCandidate(
                    subject_user_id="42",
                    fact_type="opinion",
                    claim_text=f"用户42认为话题{i}值得讨论",
                    topic=f"话题{i}",
                    stance="positive",
                    confidence=0.82,
                    evidence_message_id=f"m{i}",
                    evidence_text=f"我觉得话题{i}值得讨论",
                    source_user_id="42",
                    source_group_id="100",
                    claim_scope="self_report",
                )
                for i in range(5)
            ]

            write = storage.record_fact_candidates(facts[:4])
            self.assertEqual(len(write.accepted), 4)
            self.assertFalse(storage.should_update_user_profile("42"))

            storage.record_fact_candidates(facts[4:])
            self.assertTrue(storage.should_update_user_profile("42"))
            accepted_facts = storage.list_user_facts("42", limit=10)
            profile = storage.maybe_update_user_profile(
                "42",
                UserProfileDraft(
                    summary="用户42偏好讨论具体话题。",
                    traits={"opinions": ["愿意讨论具体话题"]},
                    supporting_fact_ids=tuple(fact.id for fact in accepted_facts),
                ),
                accepted_facts,
            )

            self.assertIsNotNone(profile)
            self.assertEqual(storage.get_user_profile("42").summary, "用户42偏好讨论具体话题。")  # type: ignore[union-attr]
            self.assertFalse(storage.should_update_user_profile("42"))

    def test_conflicting_memory_is_marked_without_overwriting_active_memory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = test_config(Path(tmp) / "bot.sqlite3")
            storage = BotStorage.from_config(config)
            storage.setup()

            first = MemoryCandidate(
                owner_type="user",
                owner_id="42",
                kind="alias",
                content="阿明",
                confidence=0.9,
                importance=0.7,
                evidence_message_id="m1",
                source_user_id="42",
                source_group_id="100",
                subject_user_id="42",
                claim_scope="self_report",
            )
            second = MemoryCandidate(
                owner_type="user",
                owner_id="42",
                kind="alias",
                content="小张",
                confidence=0.92,
                importance=0.7,
                evidence_message_id="m2",
                source_user_id="42",
                source_group_id="100",
                subject_user_id="42",
                claim_scope="self_report",
            )

            first_write = storage.record_memory_candidates([first])
            second_write = storage.record_memory_candidates([second])

            active = storage.list_memories("user", "42", status="active")
            conflicts = storage.list_memories("user", "42", status="conflict")
            self.assertEqual(len(first_write.accepted), 1)
            self.assertEqual(len(second_write.conflicts), 1)
            self.assertEqual(active[0].content, "阿明")
            self.assertEqual(conflicts[0].content, "小张")

    def test_conflict_confirmation_respects_mode_and_direct_mention(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = test_config(Path(tmp) / "bot.sqlite3")
            storage = BotStorage.from_config(config)
            storage.setup()
            first = MemoryCandidate(
                owner_type="user",
                owner_id="42",
                kind="preference",
                content="吃鱼",
                confidence=0.9,
                importance=0.7,
                evidence_message_id="m1",
                source_user_id="42",
                source_group_id="100",
                subject_user_id="42",
                claim_scope="self_report",
            )
            second = MemoryCandidate(
                owner_type="user",
                owner_id="42",
                kind="preference",
                content="吃虾",
                confidence=0.92,
                importance=0.7,
                evidence_message_id="m2",
                source_user_id="42",
                source_group_id="100",
                subject_user_id="42",
                claim_scope="self_report",
            )
            storage.record_memory_candidates([first])
            write = storage.record_memory_candidates([second])
            context = MessageContext(
                group_id="100",
                user_id="42",
                message_id="m2",
                plain_text="可可，我喜欢吃虾",
                raw_message="可可，我喜欢吃虾",
                is_direct=True,
            )

            self.assertIsNone(storage.build_conflict_confirmation(write.conflicts, context, "silent"))
            self.assertIn("吃鱼", storage.build_conflict_confirmation(write.conflicts, context, "passive") or "")

    def test_admin_can_approve_and_reject_pending_memory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = test_config(Path(tmp) / "bot.sqlite3")
            storage = BotStorage.from_config(config)
            storage.setup()
            pending = MemoryCandidate(
                owner_type="user",
                owner_id="name:小明",
                kind="preference",
                content="吃鱼",
                confidence=0.86,
                importance=0.5,
                evidence_message_id="m1",
                source_user_id="42",
                source_group_id="100",
                subject_user_id="name:小明",
                claim_scope="third_party",
            )
            storage.record_memory_candidates([pending])
            record = storage.list_memories("user", "name:小明", status="pending_confirmation")[0]

            self.assertTrue(storage.approve_memory(record.id))
            self.assertEqual(storage.list_memories("user", "name:小明", status="active")[0].content, "吃鱼")

            second = MemoryCandidate(
                owner_type="user",
                owner_id="name:小红",
                kind="preference",
                content="吃虾",
                confidence=0.86,
                importance=0.5,
                evidence_message_id="m2",
                source_user_id="42",
                source_group_id="100",
                subject_user_id="name:小红",
                claim_scope="third_party",
            )
            storage.record_memory_candidates([second])
            record = storage.list_memories("user", "name:小红", status="pending_confirmation")[0]

            self.assertTrue(storage.reject_memory(record.id))
            self.assertEqual(storage.list_memories("user", "name:小红", status="pending_confirmation"), [])

    def test_dashboard_user_cognition_includes_relationship_and_profile(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = test_config(Path(tmp) / "bot.sqlite3")
            storage = BotStorage.from_config(config)
            storage.setup()
            storage.apply_relationship_delta(
                "100",
                "42",
                RelationDelta(closeness=2, trust=3, familiarity=4, summary_patch="聊过海边"),
            )
            fact_write = storage.record_fact_candidates(
                [
                    FactCandidate(
                        subject_user_id="42",
                        fact_type="preference",
                        claim_text="用户42喜欢海边",
                        topic="海边",
                        stance="positive",
                        confidence=0.86,
                        evidence_message_id="m1",
                        evidence_text="我喜欢海边",
                        source_user_id="42",
                        source_group_id="100",
                        claim_scope="self_report",
                    )
                ]
            )
            storage.maybe_update_user_profile(
                "42",
                UserProfileDraft(summary="用户42喜欢海边。", supporting_fact_ids=(fact_write.accepted[0].id,)),
                fact_write.accepted,
                force=True,
            )

            items = storage.list_dashboard_user_cognition(group_id="100", user_id="42")

            self.assertEqual(items[0]["user_id"], "42")
            self.assertEqual(items[0]["relationship"]["trust"], 3)  # type: ignore[index]
            self.assertEqual(items[0]["profile"]["summary"], "用户42喜欢海边。")  # type: ignore[index]
            self.assertEqual(items[0]["facts"][0]["claim_text"], "用户42喜欢海边")  # type: ignore[index]

    def test_relationship_summary_filters_low_value_message_log_items(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = test_config(Path(tmp) / "bot.sqlite3")
            storage = BotStorage.from_config(config)
            storage.setup()
            with storage._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO relationships (
                        group_id, user_id, closeness, trust, familiarity, tension, summary, updated_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        "100",
                        "814207765",
                        0,
                        0,
                        1,
                        0,
                        "分享冰淇淋图片；发表负面群体类比；表达看阅兵时的无力感；"
                        "发送空消息；讨论技术与玩家便利性角度；解释因为是60帧",
                        10,
                    ),
                )

            self.assertEqual(storage.get_relationship("100", "814207765").summary, "")
            storage.apply_relationship_delta(
                "100",
                "814207765",
                RelationDelta(closeness=1, familiarity=1, summary_patch="主动找可可讨论技术问题"),
            )

            relation = storage.get_relationship("100", "814207765")
            items = storage.list_dashboard_user_cognition(group_id="100", user_id="814207765")

            self.assertEqual(relation.summary, "主动找可可讨论技术问题")
            self.assertEqual(items[0]["relationship"]["summary"], "主动找可可讨论技术问题")  # type: ignore[index]

    def test_dashboard_user_cognition_groups_by_qq_id_across_groups(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = test_config(Path(tmp) / "bot.sqlite3")
            storage = BotStorage.from_config(config)
            storage.setup()
            storage.apply_relationship_delta(
                "100",
                "42",
                RelationDelta(familiarity=1, summary_patch="一群聊过海边"),
            )
            storage.apply_relationship_delta(
                "200",
                "42",
                RelationDelta(trust=2, familiarity=3, summary_patch="二群聊过音乐"),
            )
            fact_write = storage.record_fact_candidates(
                [
                    FactCandidate(
                        subject_user_id="QQ:42",
                        fact_type="preference",
                        claim_text="用户42喜欢海边",
                        topic="海边",
                        stance="positive",
                        confidence=0.86,
                        evidence_message_id="m1",
                        evidence_text="我喜欢海边",
                        source_user_id="42",
                        source_group_id="200",
                        claim_scope="self_report",
                    )
                ]
            )
            storage.maybe_update_user_profile(
                "42",
                UserProfileDraft(summary="用户42喜欢海边。", supporting_fact_ids=(fact_write.accepted[0].id,)),
                fact_write.accepted,
                force=True,
            )

            items = storage.list_dashboard_user_cognition()
            user_items = [item for item in items if item["user_id"] == "42"]

            self.assertEqual(len(user_items), 1)
            self.assertEqual(user_items[0]["group_ids"], ["100", "200"])
            self.assertEqual(user_items[0]["relationship"]["trust"], 2)  # type: ignore[index]
            self.assertEqual(user_items[0]["relationship"]["familiarity"], 4)  # type: ignore[index]
            self.assertEqual(user_items[0]["profile"]["summary"], "用户42喜欢海边。")  # type: ignore[index]
            self.assertEqual(user_items[0]["facts"][0]["claim_text"], "用户42喜欢海边")  # type: ignore[index]

            filtered_items = storage.list_dashboard_user_cognition(group_id="100")
            filtered_user_items = [item for item in filtered_items if item["user_id"] == "42"]

            self.assertEqual(len(filtered_user_items), 1)
            self.assertEqual(filtered_user_items[0]["group_ids"], ["100", "200"])
            self.assertEqual(filtered_user_items[0]["profile"]["summary"], "用户42喜欢海边。")  # type: ignore[index]
            self.assertEqual(filtered_user_items[0]["facts"][0]["claim_text"], "用户42喜欢海边")  # type: ignore[index]

    def test_dashboard_user_cognition_uses_latest_qq_nickname(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = test_config(Path(tmp) / "bot.sqlite3")
            storage = BotStorage.from_config(config)
            storage.setup()
            storage.record_message(
                MessageContext(
                    group_id="100",
                    user_id="42",
                    message_id="m-old",
                    plain_text="旧昵称发言",
                    raw_message="旧昵称发言",
                    sender_name="旧群名片",
                    sender_nickname="旧昵称",
                    timestamp=10,
                )
            )
            storage.record_message(
                MessageContext(
                    group_id="100",
                    user_id="42",
                    message_id="m-new",
                    plain_text="新昵称发言",
                    raw_message="新昵称发言",
                    sender_name="新群名片",
                    sender_nickname="新昵称",
                    timestamp=20,
                )
            )
            storage.apply_relationship_delta("100", "42", RelationDelta(familiarity=1))

            items = storage.list_dashboard_user_cognition(user_id="42")

            self.assertEqual(items[0]["nickname"], "新昵称")
            self.assertEqual(items[0]["display_name"], "新群名片")

    def test_dashboard_messages_can_filter_by_group_user_and_date(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = test_config(Path(tmp) / "bot.sqlite3")
            storage = BotStorage.from_config(config)
            storage.setup()
            storage.record_message(
                MessageContext(
                    group_id="100",
                    user_id="42",
                    message_id="m1",
                    plain_text="今天聊海边",
                    raw_message="今天聊海边",
                    sender_name="alice",
                    timestamp=1_725_235_200,
                )
            )
            storage.record_message(
                MessageContext(
                    group_id="200",
                    user_id="42",
                    message_id="m2",
                    plain_text="别的群",
                    raw_message="别的群",
                    sender_name="alice",
                    timestamp=1_725_321_600,
                )
            )

            messages = storage.list_dashboard_messages(
                group_id="100",
                user_id="42",
                start_time=1_725_148_800,
                end_time=1_725_321_600,
            )

            self.assertEqual(len(messages), 1)
            self.assertEqual(messages[0]["plain_text"], "今天聊海边")

    def test_dashboard_messages_include_image_attachments_and_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = test_config(Path(tmp) / "bot.sqlite3")
            storage = BotStorage.from_config(config)
            storage.setup()
            storage.record_message(
                MessageContext(
                    group_id="100",
                    user_id="42",
                    message_id="m-img",
                    plain_text="看图",
                    raw_message="[CQ:image]",
                    timestamp=1_725_235_200,
                    attachments=[
                        MessageAttachment(
                            attachment_type="image",
                            file="a.png",
                            url="https://example.test/a.png",
                        )
                    ],
                )
            )
            storage.update_image_descriptions("100", "m-img", ["一张财务报表截图"])

            messages = storage.list_dashboard_messages(group_id="100")

            self.assertEqual(messages[0]["attachments"][0]["url"], "https://example.test/a.png")
            self.assertEqual(messages[0]["attachments"][0]["summary"], "一张财务报表截图")

    def test_image_description_update_keeps_unsampled_images_blank(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = test_config(Path(tmp) / "bot.sqlite3")
            storage = BotStorage.from_config(config)
            storage.setup()
            storage.record_message(
                MessageContext(
                    group_id="100",
                    user_id="42",
                    message_id="m-many-img",
                    plain_text="",
                    raw_message="[CQ:image]x5",
                    attachments=[
                        MessageAttachment(
                            attachment_type="image",
                            url=f"https://example.test/{index}.png",
                            file=f"{index}.png",
                        )
                        for index in range(5)
                    ],
                )
            )

            storage.update_image_descriptions("100", "m-many-img", ["第一张", "", "中间张", "", "最后张"])
            messages = storage.list_dashboard_messages(group_id="100")
            summaries = [item["summary"] for item in messages[0]["attachments"]]  # type: ignore[index]

            self.assertEqual(summaries, ["第一张", "", "中间张", "", "最后张"])

    def test_dashboard_pending_generates_group_commands(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = test_config(Path(tmp) / "bot.sqlite3")
            storage = BotStorage.from_config(config)
            storage.setup()
            storage.record_fact_candidates(
                [
                    FactCandidate(
                        subject_user_id="77",
                        fact_type="preference",
                        claim_text="用户77喜欢海边",
                        topic="海边",
                        stance="positive",
                        confidence=0.86,
                        evidence_message_id="m1",
                        evidence_text="77喜欢海边",
                        source_user_id="42",
                        source_group_id="100",
                        claim_scope="third_party",
                    )
                ]
            )

            pending = storage.list_dashboard_pending()

            self.assertEqual(pending[0]["status"], "pending_confirmation")
            self.assertEqual(pending[0]["item_type"], "fact")
            self.assertEqual(pending[0]["approve_command"], f"#bot facts approve {pending[0]['id']}")
            self.assertEqual(pending[0]["reject_command"], f"#bot facts reject {pending[0]['id']}")


if __name__ == "__main__":
    unittest.main()
