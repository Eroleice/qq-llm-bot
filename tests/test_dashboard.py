from __future__ import annotations

import unittest
from pathlib import Path


from qq_llm_bot.cognitive_storage import BotStorage
from qq_llm_bot.models import (
    ConversationSnapshot,
    FactCandidate,
    MemoryCandidate,
    MessageAttachment,
    MessageContext,
    MessageMention,
    RelationDelta,
    ParticipationDecision,
    StickerCandidate,
    UserProfileDraft,
)
from qq_llm_bot.llm import (
    LLMUsageRecord,
)
from tests.helpers import (
    FakeDashboardDriver,
    InMemoryBotStorage,
    dashboard_test_tools,
    project_temp_directory,
    test_config,
)


class DashboardTests(unittest.TestCase):
    def test_llm_usage_is_recorded_for_dashboard(self) -> None:
        storage = InMemoryBotStorage()
        try:
            storage.setup()

            storage.record_llm_usage(
                LLMUsageRecord(
                    purpose="perception",
                    model="gpt-5.4-mini",
                    prompt_chars=100,
                    completion_chars=20,
                    prompt_tokens=10,
                    completion_tokens=2,
                    total_tokens=12,
                    created_at=100,
                )
            )
            storage.record_llm_usage(
                LLMUsageRecord(
                    purpose="perception",
                    model="gpt-5.5",
                    prompt_chars=50,
                    completion_chars=10,
                    prompt_tokens=5,
                    completion_tokens=1,
                    total_tokens=6,
                    created_at=150,
                )
            )
            storage.record_llm_usage(
                LLMUsageRecord(
                    purpose="response",
                    model="gpt-5.5",
                    prompt_chars=300,
                    completion_chars=40,
                    prompt_tokens=30,
                    completion_tokens=4,
                    total_tokens=34,
                    created_at=200,
                )
            )

            data = storage.list_dashboard_llm_usage(since=50, limit=1)

            self.assertEqual(data["summary"]["calls"], 3)  # type: ignore[index]
            self.assertEqual(data["summary"]["total_tokens"], 52)  # type: ignore[index]
            self.assertEqual(data["summary"]["prompt_chars"], 450)  # type: ignore[index]
            by_purpose = {
                (str(item["purpose"]), str(item["model"])): item
                for item in data["by_purpose"]  # type: ignore[index]
            }
            self.assertEqual(by_purpose[("response", "gpt-5.5")]["total_tokens"], 34)
            self.assertEqual(by_purpose[("perception", "gpt-5.4-mini")]["calls"], 1)
            self.assertEqual(by_purpose[("perception", "gpt-5.5")]["total_tokens"], 6)
            self.assertEqual(len(data["recent"]), 1)  # type: ignore[arg-type]
            self.assertEqual(data["recent"][0]["purpose"], "response")  # type: ignore[index]
        finally:
            storage.connection.close()

    def test_dashboard_api_exposes_llm_usage(self) -> None:
        FastAPI, TestClient, register_dashboard_routes = dashboard_test_tools()
        storage = InMemoryBotStorage()
        try:
            storage.setup()
            storage.record_llm_usage(
                LLMUsageRecord(
                    purpose="batch_observation",
                    model="test-model",
                    prompt_chars=800,
                    completion_chars=120,
                    prompt_tokens=80,
                    completion_tokens=12,
                    total_tokens=92,
                    created_at=1_800_000_000,
                )
            )
            app = FastAPI()
            register_dashboard_routes(FakeDashboardDriver(app), storage, test_config(Path("unused.sqlite3")))

            with TestClient(app) as client:
                response = client.get("/api/dashboard/llm-usage", params={"hours": 2160, "limit": 10})

            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertEqual(data["summary"]["calls"], 1)
            self.assertEqual(data["summary"]["total_tokens"], 92)
            self.assertEqual(data["by_purpose"][0]["purpose"], "batch_observation")
            self.assertEqual(data["recent"][0]["prompt_chars"], 800)
        finally:
            storage.connection.close()

    def test_dashboard_stickers_include_delete_command(self) -> None:
        with project_temp_directory() as tmp:
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

    def test_dashboard_user_cognition_includes_relationship_and_profile(self) -> None:
        with project_temp_directory() as tmp:
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

    def test_dashboard_user_cognition_groups_by_qq_id_across_groups(self) -> None:
        with project_temp_directory() as tmp:
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
            self.assertEqual(user_items[0]["group_ids"], ["200"])
            self.assertEqual(user_items[0]["relationship"]["trust"], 2)  # type: ignore[index]
            self.assertEqual(user_items[0]["relationship"]["familiarity"], 4)  # type: ignore[index]
            self.assertEqual(user_items[0]["profile"]["summary"], "用户42喜欢海边。")  # type: ignore[index]
            self.assertEqual(user_items[0]["facts"][0]["claim_text"], "用户42喜欢海边")  # type: ignore[index]

            filtered_items = storage.list_dashboard_user_cognition(group_id="100")
            filtered_user_items = [item for item in filtered_items if item["user_id"] == "42"]

            self.assertEqual(len(filtered_user_items), 1)
            self.assertEqual(filtered_user_items[0]["group_ids"], ["200"])
            self.assertEqual(filtered_user_items[0]["relationship"]["familiarity"], 4)  # type: ignore[index]
            self.assertEqual(filtered_user_items[0]["profile"]["summary"], "用户42喜欢海边。")  # type: ignore[index]
            self.assertEqual(filtered_user_items[0]["facts"][0]["claim_text"], "用户42喜欢海边")  # type: ignore[index]

    def test_dashboard_user_cognition_uses_latest_qq_nickname(self) -> None:
        with project_temp_directory() as tmp:
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
        with project_temp_directory() as tmp:
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

    def test_dashboard_messages_include_mentions(self) -> None:
        with project_temp_directory() as tmp:
            config = test_config(Path(tmp) / "bot.sqlite3")
            storage = BotStorage.from_config(config)
            storage.setup()
            storage.record_message(
                MessageContext(
                    group_id="100",
                    user_id="42",
                    message_id="m-at",
                    plain_text="这个人 @Alice(QQ:123) 叫小明",
                    raw_message="[CQ:at,qq=123,name=Alice] 叫小明",
                    timestamp=20,
                    mentions=[
                        MessageMention(
                            user_id="123",
                            display_name="Alice",
                            raw_data='{"qq":"123","name":"Alice"}',
                        )
                    ],
                )
            )
            storage.record_message(
                MessageContext(
                    group_id="100",
                    user_id="42",
                    message_id="m-plain",
                    plain_text="普通消息",
                    raw_message="普通消息",
                    timestamp=10,
                )
            )

            messages = storage.list_dashboard_messages(group_id="100")
            by_message_id = {str(item["message_id"]): item for item in messages}

            self.assertEqual(by_message_id["m-at"]["mentions"][0]["user_id"], "123")  # type: ignore[index]
            self.assertEqual(by_message_id["m-at"]["mentions"][0]["display_name"], "Alice")  # type: ignore[index]
            self.assertEqual(by_message_id["m-plain"]["mentions"], [])

    def test_dashboard_messages_include_image_attachments_and_summary(self) -> None:
        with project_temp_directory() as tmp:
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

    def test_storage_archives_final_qa_blocks_for_dashboard(self) -> None:
        with project_temp_directory() as tmp:
            config = test_config(Path(tmp) / "bot.sqlite3")
            storage = BotStorage.from_config(config)
            storage.setup()
            context = MessageContext(
                group_id="100",
                user_id="42",
                message_id="m-qa",
                plain_text="可可你怎么看",
                raw_message="可可你怎么看",
                sender_name="alice",
                is_direct=True,
                bot_mentioned=True,
                timestamp=1_725_235_200,
            )
            decision = ParticipationDecision(
                "observe",
                "message is directed to the bot; final QA blocked reply: 涉及政治立场",
                "passive",
                0.49,
                "answer",
                1.0,
            )
            snapshot = ConversationSnapshot(
                recent_messages=["alice: 可可你怎么看", "bob: 这个新闻很复杂"],
                speaker_recent_messages=["alice: 可可你怎么看"],
                other_recent_messages=["bob: 这个新闻很复杂"],
                recent_image_descriptions=["一张新闻截图"],
            )

            storage.record_final_qa_block(
                context,
                decision,
                snapshot,
                candidate_reply="我支持这个立场。",
                qa_reason="涉及政治立场",
                qa_categories=("political_stance",),
                qa_confidence=0.94,
            )

            items = storage.list_dashboard_final_qa_blocks(group_id="100", user_id="42")

            self.assertEqual(len(items), 1)
            self.assertEqual(items[0]["candidate_reply"], "我支持这个立场。")
            self.assertEqual(items[0]["qa_reason"], "涉及政治立场")
            self.assertEqual(items[0]["qa_categories"], ["political_stance"])
            self.assertEqual(items[0]["recent_messages"], ["alice: 可可你怎么看", "bob: 这个新闻很复杂"])
            self.assertEqual(items[0]["speaker_recent_messages"], ["alice: 可可你怎么看"])
            self.assertTrue(items[0]["is_direct"])

    def test_dashboard_api_exposes_final_qa_blocks(self) -> None:
        FastAPI, TestClient, register_dashboard_routes = dashboard_test_tools()
        with project_temp_directory() as tmp:
            config = test_config(Path(tmp) / "bot.sqlite3")
            storage = BotStorage.from_config(config)
            storage.setup()
            storage.record_final_qa_block(
                MessageContext(
                    group_id="100",
                    user_id="42",
                    message_id="m-qa-api",
                    plain_text="可可讲个笑话",
                    raw_message="可可讲个笑话",
                    sender_name="alice",
                    is_direct=True,
                    timestamp=1_725_235_200,
                ),
                ParticipationDecision("observe", "final QA blocked reply: 政治话题", "passive", 0.49),
                ConversationSnapshot(recent_messages=["alice: 可可讲个笑话"]),
                candidate_reply="某政治人物笑话",
                qa_reason="政治话题",
                qa_categories=("political_stance",),
                qa_confidence=0.9,
            )
            app = FastAPI()
            register_dashboard_routes(FakeDashboardDriver(app), storage, config)

            with TestClient(app) as client:
                response = client.get("/api/dashboard/qa-blocks", params={"group_id": "100", "limit": 10})

            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertEqual(data["items"][0]["message_id"], "m-qa-api")
            self.assertEqual(data["items"][0]["candidate_reply"], "某政治人物笑话")
            self.assertEqual(data["items"][0]["qa_categories"], ["political_stance"])

    def test_dashboard_pending_generates_group_commands(self) -> None:
        with project_temp_directory() as tmp:
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

    def test_dashboard_api_can_manage_pending_fact(self) -> None:
        FastAPI, TestClient, register_dashboard_routes = dashboard_test_tools()
        with project_temp_directory() as tmp:
            config = test_config(Path(tmp) / "bot.sqlite3")
            storage = BotStorage.from_config(config)
            storage.setup()
            changed_users: list[str] = []

            async def on_fact_changed(user_ids: list[str]) -> None:
                changed_users.extend(user_ids)

            app = FastAPI()
            register_dashboard_routes(
                FakeDashboardDriver(app),
                storage,
                config,
                on_fact_changed=on_fact_changed,
            )
            write = storage.record_fact_candidates(
                [
                    FactCandidate(
                        subject_user_id="77",
                        fact_type="preference",
                        claim_text="user 77 likes seaside",
                        topic="seaside",
                        stance="positive",
                        confidence=0.86,
                        evidence_message_id="m1",
                        evidence_text="77 likes seaside",
                        source_user_id="42",
                        source_group_id="100",
                        claim_scope="third_party",
                    )
                ]
            )
            fact_id = write.pending[0].id

            with TestClient(app) as client:
                response = client.post(f"/api/dashboard/facts/{fact_id}/approve")
                self.assertEqual(response.status_code, 200)
                self.assertTrue(response.json()["ok"])
                self.assertEqual(storage.get_fact_record(fact_id).status, "accepted")  # type: ignore[union-attr]

                response = client.delete(f"/api/dashboard/facts/{fact_id}")
                self.assertEqual(response.status_code, 200)
                self.assertEqual(storage.get_fact_record(fact_id).status, "forgotten")  # type: ignore[union-attr]

            self.assertEqual(changed_users, ["77", "77"])

    def test_dashboard_api_can_reject_pending_memory_and_fact(self) -> None:
        FastAPI, TestClient, register_dashboard_routes = dashboard_test_tools()
        with project_temp_directory() as tmp:
            config = test_config(Path(tmp) / "bot.sqlite3")
            storage = BotStorage.from_config(config)
            storage.setup()
            app = FastAPI()
            register_dashboard_routes(FakeDashboardDriver(app), storage, config)
            fact_write = storage.record_fact_candidates(
                [
                    FactCandidate(
                        subject_user_id="88",
                        fact_type="preference",
                        claim_text="user 88 likes shrimp",
                        topic="shrimp",
                        stance="positive",
                        confidence=0.86,
                        evidence_message_id="m2",
                        evidence_text="88 likes shrimp",
                        source_user_id="42",
                        source_group_id="100",
                        claim_scope="third_party",
                    )
                ]
            )
            memory = MemoryCandidate(
                owner_type="user",
                owner_id="name:alice",
                kind="preference",
                content="likes fish",
                confidence=0.86,
                importance=0.5,
                evidence_message_id="m3",
                source_user_id="42",
                source_group_id="100",
                subject_user_id="name:alice",
                claim_scope="third_party",
            )
            storage.record_memory_candidates([memory])
            memory_id = storage.list_memories(
                "user",
                "name:alice",
                status="pending_confirmation",
            )[0].id

            with TestClient(app) as client:
                response = client.post(f"/api/dashboard/facts/{fact_write.pending[0].id}/reject")
                self.assertEqual(response.status_code, 200)
                self.assertEqual(
                    storage.get_fact_record(fact_write.pending[0].id).status,  # type: ignore[union-attr]
                    "rejected",
                )

                response = client.post(f"/api/dashboard/memories/{memory_id}/reject")
                self.assertEqual(response.status_code, 200)
                self.assertEqual(
                    storage.list_memories("user", "name:alice", status="pending_confirmation"),
                    [],
                )

    def test_dashboard_api_can_bulk_manage_pending_items(self) -> None:
        FastAPI, TestClient, register_dashboard_routes = dashboard_test_tools()
        with project_temp_directory() as tmp:
            config = test_config(Path(tmp) / "bot.sqlite3")
            storage = BotStorage.from_config(config)
            storage.setup()
            changed_users: list[str] = []

            async def on_fact_changed(user_ids: list[str]) -> None:
                changed_users.extend(user_ids)

            app = FastAPI()
            register_dashboard_routes(
                FakeDashboardDriver(app),
                storage,
                config,
                on_fact_changed=on_fact_changed,
            )
            fact_write = storage.record_fact_candidates(
                [
                    FactCandidate(
                        subject_user_id="77",
                        fact_type="preference",
                        claim_text="user 77 likes seaside",
                        topic="seaside",
                        stance="positive",
                        confidence=0.86,
                        evidence_message_id="m1",
                        evidence_text="77 likes seaside",
                        source_user_id="42",
                        source_group_id="100",
                        claim_scope="third_party",
                    ),
                    FactCandidate(
                        subject_user_id="88",
                        fact_type="preference",
                        claim_text="user 88 likes shrimp",
                        topic="shrimp",
                        stance="positive",
                        confidence=0.86,
                        evidence_message_id="m2",
                        evidence_text="88 likes shrimp",
                        source_user_id="42",
                        source_group_id="100",
                        claim_scope="third_party",
                    ),
                ]
            )
            storage.record_memory_candidates(
                [
                    MemoryCandidate(
                        owner_type="user",
                        owner_id="name:alice",
                        kind="preference",
                        content="likes fish",
                        confidence=0.86,
                        importance=0.5,
                        evidence_message_id="m3",
                        source_user_id="42",
                        source_group_id="100",
                        subject_user_id="name:alice",
                        claim_scope="third_party",
                    )
                ]
            )
            memory_id = storage.list_memories(
                "user",
                "name:alice",
                status="pending_confirmation",
            )[0].id

            with TestClient(app) as client:
                response = client.post(
                    "/api/dashboard/pending/bulk",
                    json={
                        "action": "approve",
                        "items": [
                            {"item_type": "fact", "id": fact_write.pending[0].id},
                            {"item_type": "memory", "id": memory_id},
                        ],
                    },
                )
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json()["count"], 2)
                self.assertEqual(
                    storage.get_fact_record(fact_write.pending[0].id).status,  # type: ignore[union-attr]
                    "accepted",
                )
                self.assertEqual(
                    storage.list_memories("user", "name:alice", status="active")[0].id,
                    memory_id,
                )

                response = client.post(
                    "/api/dashboard/pending/bulk",
                    json={
                        "action": "reject",
                        "items": [{"item_type": "fact", "id": fact_write.pending[1].id}],
                    },
                )
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json()["count"], 1)
                self.assertEqual(
                    storage.get_fact_record(fact_write.pending[1].id).status,  # type: ignore[union-attr]
                    "rejected",
                )

            self.assertEqual(changed_users, ["77"])

    def test_dashboard_api_deletes_sticker_asset_and_file(self) -> None:
        FastAPI, TestClient, register_dashboard_routes = dashboard_test_tools()
        with project_temp_directory() as tmp:
            config = test_config(Path(tmp) / "bot.sqlite3")
            storage = BotStorage.from_config(config)
            storage.setup()
            sticker_dir = config.resolve_path(config.stickers.storage_dir) / "100"
            sticker_dir.mkdir(parents=True)
            sticker_file = sticker_dir / "meme.png"
            sticker_file.write_bytes(b"fake image")
            context = MessageContext(
                group_id="100",
                user_id="42",
                message_id="m-sticker",
                plain_text="",
                raw_message="[CQ:image]",
            )
            asset = storage.upsert_sticker_asset(
                context,
                StickerCandidate(
                    url="https://example.test/meme.png",
                    file="meme.png",
                    description="meme",
                    confidence=0.86,
                ),
                local_path=str(sticker_file),
                sha256="abc",
            )
            app = FastAPI()
            register_dashboard_routes(FakeDashboardDriver(app), storage, config)

            with TestClient(app) as client:
                response = client.delete(f"/api/dashboard/stickers/{asset.id}")  # type: ignore[union-attr]

            self.assertEqual(response.status_code, 200)
            self.assertTrue(response.json()["deleted_file"])
            self.assertIsNone(storage.get_sticker_asset(asset.id if asset else 0))
            self.assertFalse(sticker_file.exists())


if __name__ == "__main__":
    unittest.main()


