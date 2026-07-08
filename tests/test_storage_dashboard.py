from __future__ import annotations

import ast
import unittest
from dataclasses import replace
from pathlib import Path


from qq_llm_bot.cognitive_storage import BotStorage
from qq_llm_bot.models import (
    FactCandidate,
    MemoryCandidate,
    MessageAttachment,
    MessageContext,
    MessageMention,
    RelationDelta,
    ParticipationDecision,
    UserProfileDraft,
)
from tests.helpers import (
    first_attribute_call_line,
    first_name_call_line,
    project_temp_directory,
    test_config,
)


class MemoryStorageTests(unittest.TestCase):
    def test_global_ignore_list_is_seeded_and_updated(self) -> None:
        with project_temp_directory() as tmp:
            base_config = test_config(Path(tmp) / "bot.sqlite3")
            config = replace(
                base_config,
                bot=replace(base_config.bot, ignored_user_ids=["QQ:42", "7"]),
            )
            storage = BotStorage.from_config(config)
            storage.setup()

            self.assertTrue(storage.is_user_ignored("42"))
            self.assertTrue(storage.is_user_ignored("QQ:42"))
            self.assertTrue(storage.is_user_ignored("7"))
            self.assertCountEqual(storage.list_ignored_users(), ["42", "7"])

            storage.remove_ignored_user("42")
            self.assertFalse(storage.is_user_ignored("42"))
            self.assertEqual(storage.list_ignored_users(), ["7"])

            storage.add_ignored_user("QQ:123")
            self.assertTrue(storage.is_user_ignored("123"))
            self.assertCountEqual(storage.list_ignored_users(), ["123", "7"])

    def test_image_generation_usage_is_counted_by_user_and_date(self) -> None:
        with project_temp_directory() as tmp:
            config = test_config(Path(tmp) / "bot.sqlite3")
            storage = BotStorage.from_config(config)
            storage.setup()

            storage.record_image_generation_usage(
                "100",
                "42",
                "2026-07-05",
                "cat",
                "data/generated_images/100/cat.png",
                created_at=1,
            )
            storage.record_image_generation_usage(
                "100",
                "42",
                "2026-07-05",
                "dog",
                "data/generated_images/100/dog.png",
                created_at=2,
            )
            storage.record_image_generation_usage(
                "100",
                "42",
                "2026-07-06",
                "bird",
                "data/generated_images/100/bird.png",
                created_at=3,
            )

            self.assertEqual(storage.count_image_generation_usage("42", "2026-07-05"), 2)
            self.assertEqual(storage.count_image_generation_usage("42", "2026-07-06"), 1)
            self.assertEqual(storage.count_image_generation_usage("7", "2026-07-05"), 0)

    def test_group_handler_records_ignored_user_messages_before_skip(self) -> None:
        plugin_dir = (
            Path(__file__).resolve().parents[1] / "plugins" / "llm_group_bot" / "__init__.py"
        ).parent
        source = (plugin_dir / "__init__.py").read_text(encoding="utf-8")
        deferred_vision_source = (plugin_dir / "deferred_vision.py").read_text(encoding="utf-8")
        observation_source = (plugin_dir / "observation_batch.py").read_text(encoding="utf-8")
        module = ast.parse(source)
        deferred_vision_module = ast.parse(deferred_vision_source)
        observation_module = ast.parse(observation_source)
        observation_class = next(
            node
            for node in observation_module.body
            if isinstance(node, ast.ClassDef) and node.name == "ObservationBatchCoordinator"
        )

        def observation_method(name: str) -> ast.FunctionDef | ast.AsyncFunctionDef:
            return next(
                node
                for node in observation_class.body
                if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and node.name == name
            )

        handler = next(
            node
            for node in module.body
            if isinstance(node, ast.AsyncFunctionDef) and node.name == "_handle_group_message"
        )
        process_handler = next(
            node
            for node in module.body
            if isinstance(node, ast.AsyncFunctionDef) and node.name == "_process_group_context"
        )
        defer_handler = observation_method("defer_observation")
        pending_deferred_handler = next(
            node
            for node in deferred_vision_module.body
            if isinstance(node, ast.AsyncFunctionDef) and node.name == "_run_pending_deferred_vision"
        )
        deferred_vision_handler = observation_method("record_deferred_vision")

        record_line = first_attribute_call_line(handler, "record_message")
        ignore_line = first_attribute_call_line(handler, "is_user_ignored")
        register_pending_line = first_attribute_call_line(handler, "register_pending_vision")
        realtime_enqueue_line = first_attribute_call_line(handler, "maybe_enqueue_realtime_reply")
        defer_check_line = first_attribute_call_line(handler, "should_defer_realtime_pipeline")
        defer_line = first_attribute_call_line(handler, "defer_observation")
        process_line = first_name_call_line(handler, "_process_group_context")
        pending_wait_line = first_attribute_call_line(process_handler, "wait_for_relevant_pending_vision")
        flush_line = first_attribute_call_line(process_handler, "flush_observation_batch")
        pipeline_line = first_attribute_call_line(process_handler, "run")
        ensure_deferred_task_line = first_attribute_call_line(defer_handler, "ensure_deferred_vision_task")
        buffer_append_line = first_attribute_call_line(defer_handler, "append")
        deferred_vision_line = first_name_call_line(
            pending_deferred_handler,
            "_record_deferred_vision",
        )
        pending_finish_line = first_name_call_line(
            pending_deferred_handler,
            "_finish_pending_vision",
        )
        observe_vision_line = first_attribute_call_line(deferred_vision_handler, "observe_vision")
        image_summary_line = first_attribute_call_line(deferred_vision_handler, "update_image_descriptions")

        self.assertLess(record_line, ignore_line)
        self.assertLess(ignore_line, register_pending_line)
        self.assertLess(register_pending_line, realtime_enqueue_line)
        self.assertLess(realtime_enqueue_line, defer_check_line)
        self.assertLess(defer_check_line, defer_line)
        self.assertLess(defer_line, process_line)
        self.assertLess(pending_wait_line, flush_line)
        self.assertLess(flush_line, pipeline_line)
        self.assertLess(ignore_line, process_line)
        self.assertLess(ensure_deferred_task_line, buffer_append_line)
        self.assertLess(deferred_vision_line, pending_finish_line)
        self.assertLess(observe_vision_line, image_summary_line)

    def test_realtime_merge_source_keeps_cancel_and_commit_contract(self) -> None:
        plugin_source = (
            Path(__file__).resolve().parents[1] / "plugins" / "llm_group_bot" / "__init__.py"
        ).read_text(encoding="utf-8")
        observation_source = (
            Path(__file__).resolve().parents[1]
            / "plugins"
            / "llm_group_bot"
            / "observation_batch.py"
        ).read_text(encoding="utf-8")
        source = (
            Path(__file__).resolve().parents[1]
            / "plugins"
            / "llm_group_bot"
            / "realtime_reply.py"
        ).read_text(encoding="utf-8")

        self.assertIn("pending.generation += 1", source)
        self.assertIn("_cancel_realtime_task(pending)", source)
        self.assertIn("pending.task.cancel()", source)
        self.assertIn("_claim_pending_realtime_reply", source)
        self.assertIn("pending.committing = True", source)
        self.assertIn("merge_realtime_contexts(contexts)", source)
        self.assertIn("observation_batch.should_defer_realtime_pipeline", plugin_source)
        self.assertIn("if context.bot_mentioned:", observation_source)

    def test_group_handler_schedules_slow_maintenance_off_send_path(self) -> None:
        root = Path(__file__).resolve().parents[1]
        plugin_source = (root / "plugins" / "llm_group_bot" / "__init__.py").read_text(
            encoding="utf-8"
        )
        observation_source = (
            root / "plugins" / "llm_group_bot" / "observation_batch.py"
        ).read_text(encoding="utf-8")
        module = ast.parse(plugin_source)
        process_handler = next(
            node
            for node in module.body
            if isinstance(node, ast.AsyncFunctionDef) and node.name == "_process_group_context"
        )
        process_source = ast.get_source_segment(plugin_source, process_handler) or ""

        self.assertNotIn("await _maintenance.update_profiles", process_source)
        self.assertNotIn("await _maintenance.reflect_group", process_source)
        self.assertIn("_schedule_post_pipeline_maintenance", process_source)
        self.assertIn("force=False", plugin_source)
        self.assertNotIn("force=bool(fact_write.accepted)", plugin_source)
        self.assertIn("force=False", observation_source)
        self.assertNotIn("force=bool(fact_write.accepted)", observation_source)

        send_line = first_attribute_call_line(process_handler, "send_group_reply")
        maintenance_schedule_lines = [
            node.lineno
            for node in ast.walk(process_handler)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_schedule_post_pipeline_maintenance"
        ]
        self.assertTrue(any(line > send_line for line in maintenance_schedule_lines))

    def test_sticker_selection_runs_after_text_reply_probability_gate(self) -> None:
        root = Path(__file__).resolve().parents[1]
        plugin_source = (root / "plugins" / "llm_group_bot" / "__init__.py").read_text(
            encoding="utf-8"
        )
        pipeline_source = (root / "qq_llm_bot" / "cognitive_agents.py").read_text(
            encoding="utf-8"
        )
        config_source = (root / "qq_llm_bot" / "config_media_sections.py").read_text(
            encoding="utf-8"
        )
        plugin_module = ast.parse(plugin_source)
        pipeline_module = ast.parse(pipeline_source)
        process_handler = next(
            node
            for node in plugin_module.body
            if isinstance(node, ast.AsyncFunctionDef) and node.name == "_process_group_context"
        )
        agent_class = next(
            node
            for node in pipeline_module.body
            if isinstance(node, ast.ClassDef) and node.name == "AgentPipeline"
        )
        run_method = next(
            node
            for node in agent_class.body
            if isinstance(node, ast.AsyncFunctionDef) and node.name == "run"
        )
        run_source = ast.get_source_segment(pipeline_source, run_method) or ""

        self.assertNotIn("await self.stickers.select", run_source)
        self.assertIn("async def select_sticker", pipeline_source)
        self.assertIn("_schedule_post_reply_sticker", plugin_source)
        self.assertIn("send_was_queued=send_result.queued", plugin_source)
        self.assertIn("random.random() < probability", plugin_source)
        self.assertIn('raw.get("send_probability", 0.10)', config_source)
        self.assertNotIn("result.selected_sticker", plugin_source)

        send_line = first_attribute_call_line(process_handler, "send_group_reply")
        sticker_schedule_lines = [
            node.lineno
            for node in ast.walk(process_handler)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_schedule_post_reply_sticker"
        ]
        self.assertTrue(any(line > send_line for line in sticker_schedule_lines))

    def test_snapshot_groups_current_speaker_context_for_llm(self) -> None:
        with project_temp_directory() as tmp:
            config = test_config(Path(tmp) / "bot.sqlite3")
            storage = BotStorage.from_config(config)
            storage.setup()
            storage.record_message(
                MessageContext(
                    group_id="100",
                    user_id="7",
                    message_id="m-bob-1",
                    plain_text="我准备直接抽",
                    raw_message="我准备直接抽",
                    sender_name="bob",
                    timestamp=10,
                )
            )
            storage.record_message(
                MessageContext(
                    group_id="100",
                    user_id="42",
                    message_id="m-alice-1",
                    plain_text="我预算不多",
                    raw_message="我预算不多",
                    sender_name="alice",
                    timestamp=20,
                )
            )
            storage.record_message(
                MessageContext(
                    group_id="100",
                    user_id="8",
                    message_id="m-carol-1",
                    plain_text="等复刻也行",
                    raw_message="等复刻也行",
                    sender_name="carol",
                    timestamp=30,
                )
            )
            context = MessageContext(
                group_id="100",
                user_id="42",
                message_id="m-alice-2",
                plain_text="那我是不是先别抽",
                raw_message="那我是不是先别抽",
                sender_name="alice",
                timestamp=40,
            )
            storage.record_message(context)

            snapshot = storage.build_snapshot(context)

            self.assertEqual(
                snapshot.speaker_recent_messages,
                ["alice: 我预算不多", "alice: 那我是不是先别抽"],
            )
            self.assertEqual(
                snapshot.other_recent_messages,
                ["bob: 我准备直接抽", "carol: 等复刻也行"],
            )
            self.assertEqual(
                snapshot.recent_messages,
                [
                    "bob: 我准备直接抽",
                    "alice: 我预算不多",
                    "carol: 等复刻也行",
                    "alice: 那我是不是先别抽",
                ],
            )

    def test_snapshot_includes_recent_bot_reply_to_same_user(self) -> None:
        with project_temp_directory() as tmp:
            config = test_config(Path(tmp) / "bot.sqlite3")
            storage = BotStorage.from_config(config)
            storage.setup()
            original_context = MessageContext(
                group_id="100",
                user_id="42",
                message_id="m-direct",
                plain_text="可可我预算不多还抽吗",
                raw_message="可可我预算不多还抽吗",
                is_direct=True,
            )
            storage.record_decision(
                original_context,
                ParticipationDecision("reply", "message is directed to the bot", "passive", 1.0),
                "预算不多的话可以先等。",
            )
            followup_context = MessageContext(
                group_id="100",
                user_id="42",
                message_id="m-followup",
                plain_text="那我是不是先别抽",
                raw_message="那我是不是先别抽",
            )

            snapshot = storage.build_snapshot(followup_context)

            self.assertEqual(snapshot.recent_bot_reply_to_user, "预算不多的话可以先等。")
            self.assertGreaterEqual(snapshot.recent_bot_reply_to_user_seconds, 0)

    def test_bot_sourced_lexicon_group_fact_is_accepted(self) -> None:
        with project_temp_directory() as tmp:
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
        with project_temp_directory() as tmp:
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
        with project_temp_directory() as tmp:
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
        with project_temp_directory() as tmp:
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
        with project_temp_directory() as tmp:
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
        with project_temp_directory() as tmp:
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

    def test_member_aliases_allow_multiple_names_and_resolve_targets(self) -> None:
        with project_temp_directory() as tmp:
            config = test_config(Path(tmp) / "bot.sqlite3")
            storage = BotStorage.from_config(config)
            storage.setup()

            facts = [
                FactCandidate(
                    subject_user_id="123",
                    fact_type="identity",
                    claim_text="QQ:123 的称呼是牛宝宝。",
                    topic="称呼",
                    stance="neutral",
                    confidence=0.95,
                    evidence_message_id="m-alias-1",
                    evidence_text="我叫牛宝宝",
                    source_user_id="123",
                    source_group_id="100",
                    claim_scope="self_report",
                    importance=0.9,
                ),
                FactCandidate(
                    subject_user_id="123",
                    fact_type="identity",
                    claim_text="QQ:123 的昵称是牛牛。",
                    topic="昵称",
                    stance="neutral",
                    confidence=0.94,
                    evidence_message_id="m-alias-2",
                    evidence_text="昵称是牛牛",
                    source_user_id="123",
                    source_group_id="100",
                    claim_scope="self_report",
                    importance=0.9,
                ),
            ]

            storage.record_fact_candidates(facts)
            by_first = storage.build_snapshot(
                MessageContext("100", "42", "q1", "谁是牛宝宝", "谁是牛宝宝")
            )
            by_second = storage.build_snapshot(
                MessageContext("100", "42", "q2", "牛牛是谁", "牛牛是谁")
            )

            self.assertEqual(by_first.target_users[0].user_id, "123")
            self.assertEqual(by_second.target_users[0].user_id, "123")
            self.assertIn("牛宝宝", by_first.target_users[0].aliases)
            self.assertIn("牛牛", by_first.target_users[0].aliases)

    def test_display_name_snapshot_guesses_target_when_alias_missing(self) -> None:
        with project_temp_directory() as tmp:
            config = test_config(Path(tmp) / "bot.sqlite3")
            storage = BotStorage.from_config(config)
            storage.setup()
            storage.record_message(
                MessageContext(
                    group_id="100",
                    user_id="123",
                    message_id="m-name",
                    plain_text="我来了",
                    raw_message="我来了",
                    sender_name="牛宝宝",
                    sender_nickname="牛牛",
                    timestamp=10,
                )
            )

            snapshot = storage.build_snapshot(
                MessageContext("100", "42", "q-name", "谁是牛宝", "谁是牛宝")
            )

            self.assertEqual(snapshot.unknown_name_refs, [])
            self.assertEqual(snapshot.target_users[0].user_id, "123")
            self.assertEqual(snapshot.target_users[0].resolution_status, "guessed")
            self.assertIn("display_name_guess:牛宝->牛宝宝", snapshot.target_users[0].match_reason)

    def test_latest_profile_nickname_guesses_target_when_group_snapshot_missing(self) -> None:
        with project_temp_directory() as tmp:
            config = test_config(Path(tmp) / "bot.sqlite3")
            storage = BotStorage.from_config(config)
            storage.setup()
            storage.record_message(
                MessageContext(
                    group_id="200",
                    user_id="456",
                    message_id="m-profile-name",
                    plain_text="跨群更新昵称",
                    raw_message="跨群更新昵称",
                    sender_name="旧名片",
                    sender_nickname="阿牛",
                    timestamp=10,
                )
            )

            snapshot = storage.build_snapshot(
                MessageContext("100", "42", "q-profile-name", "阿牛是谁", "阿牛是谁")
            )

            self.assertEqual(snapshot.target_users[0].user_id, "456")
            self.assertEqual(snapshot.target_users[0].resolution_status, "guessed")
            self.assertIn("nickname_guess:阿牛->阿牛", snapshot.target_users[0].match_reason)

    def test_display_name_guess_marks_ambiguous_close_matches(self) -> None:
        with project_temp_directory() as tmp:
            config = test_config(Path(tmp) / "bot.sqlite3")
            storage = BotStorage.from_config(config)
            storage.setup()
            storage.record_message(
                MessageContext("100", "123", "m-a", "a", "a", sender_name="阿牛", timestamp=10)
            )
            storage.record_message(
                MessageContext("100", "456", "m-b", "b", "b", sender_name="阿牛", timestamp=11)
            )

            snapshot = storage.build_snapshot(
                MessageContext("100", "42", "q-ambiguous", "阿牛是谁", "阿牛是谁")
            )

            self.assertEqual(snapshot.target_users, [])
            self.assertEqual(set(snapshot.ambiguous_name_refs["阿牛"]), {"123", "456"})

    def test_relationship_titles_are_rejected_as_member_aliases(self) -> None:
        with project_temp_directory() as tmp:
            config = test_config(Path(tmp) / "bot.sqlite3")
            storage = BotStorage.from_config(config)
            storage.setup()
            candidates = [
                FactCandidate(
                    subject_user_id="123",
                    fact_type="identity",
                    claim_text="QQ:123 的称呼是主人。",
                    topic="称呼",
                    stance="neutral",
                    confidence=0.98,
                    evidence_message_id="m-title-1",
                    evidence_text="我叫主人",
                    source_user_id="123",
                    source_group_id="100",
                    claim_scope="self_report",
                    importance=0.9,
                ),
                FactCandidate(
                    subject_user_id="456",
                    fact_type="identity",
                    claim_text="QQ:456 的昵称是老板。",
                    topic="昵称",
                    stance="neutral",
                    confidence=0.98,
                    evidence_message_id="m-title-2",
                    evidence_text="@QQ:456 叫老板",
                    source_user_id="42",
                    source_group_id="100",
                    claim_scope="third_party",
                    importance=0.9,
                ),
                FactCandidate(
                    subject_user_id="789",
                    fact_type="identity",
                    claim_text="QQ:789 的称呼是管理员。",
                    topic="称呼",
                    stance="neutral",
                    confidence=0.98,
                    evidence_message_id="m-title-3",
                    evidence_text="@QQ:789 叫管理员",
                    source_user_id="42",
                    source_group_id="100",
                    claim_scope="third_party",
                    importance=0.9,
                ),
                FactCandidate(
                    subject_user_id="999",
                    fact_type="identity",
                    claim_text="QQ:999 的称呼是爸爸。",
                    topic="称呼",
                    stance="neutral",
                    confidence=0.98,
                    evidence_message_id="m-title-4",
                    evidence_text="我叫爸爸",
                    source_user_id="999",
                    source_group_id="100",
                    claim_scope="self_report",
                    importance=0.9,
                ),
            ]

            write = storage.record_fact_candidates(candidates)
            snapshot = storage.build_snapshot(
                MessageContext("100", "42", "q-title", "谁是主人", "谁是主人")
            )

            self.assertEqual(len(write.rejected), len(candidates))
            self.assertEqual(write.accepted, [])
            self.assertEqual(write.pending, [])
            self.assertEqual(storage.list_user_facts("123", include_faded=True), [])
            self.assertEqual(snapshot.target_users, [])
            self.assertEqual(snapshot.unknown_name_refs, ["主人"])

    def test_mixed_relationship_title_claim_keeps_reasonable_alias_only(self) -> None:
        with project_temp_directory() as tmp:
            config = test_config(Path(tmp) / "bot.sqlite3")
            storage = BotStorage.from_config(config)
            storage.setup()

            write = storage.record_fact_candidates(
                [
                    FactCandidate(
                        subject_user_id="123",
                        fact_type="identity",
                        claim_text="QQ:123 的昵称是牛宝宝。",
                        topic="称呼",
                        stance="neutral",
                        confidence=0.98,
                        evidence_message_id="m-title-mixed",
                        evidence_text="别叫我主人，叫我牛宝宝",
                        source_user_id="123",
                        source_group_id="100",
                        claim_scope="self_report",
                        importance=0.9,
                    )
                ]
            )
            by_good_name = storage.build_snapshot(
                MessageContext("100", "42", "q-good", "谁是牛宝宝", "谁是牛宝宝")
            )
            by_bad_title = storage.build_snapshot(
                MessageContext("100", "42", "q-bad", "谁是主人", "谁是主人")
            )

            self.assertEqual(len(write.accepted), 1)
            self.assertEqual(by_good_name.target_users[0].user_id, "123")
            self.assertIn("牛宝宝", by_good_name.target_users[0].aliases)
            self.assertNotIn("主人", by_good_name.target_users[0].aliases)
            self.assertEqual(by_bad_title.target_users, [])
            self.assertEqual(by_bad_title.unknown_name_refs, ["主人"])

    def test_mention_target_injects_member_facts_for_non_speaker(self) -> None:
        with project_temp_directory() as tmp:
            config = test_config(Path(tmp) / "bot.sqlite3")
            storage = BotStorage.from_config(config)
            storage.setup()
            storage.record_fact_candidates(
                [
                    FactCandidate(
                        subject_user_id="1657222326",
                        fact_type="identity",
                        claim_text="QQ:1657222326 的称呼是牛宝宝。",
                        topic="称呼",
                        stance="neutral",
                        confidence=0.95,
                        evidence_message_id="m-target",
                        evidence_text="我叫牛宝宝",
                        source_user_id="1657222326",
                        source_group_id="100",
                        claim_scope="self_report",
                        importance=0.9,
                    )
                ]
            )

            snapshot = storage.build_snapshot(
                MessageContext(
                    group_id="100",
                    user_id="42",
                    message_id="q-target",
                    plain_text="你要怎么称呼 @QQ:1657222326？",
                    raw_message="",
                    mentions=[MessageMention(user_id="1657222326")],
                )
            )

            self.assertEqual(snapshot.user_facts, [])
            self.assertEqual(snapshot.target_users[0].user_id, "1657222326")
            self.assertIn("牛宝宝", snapshot.target_users[0].aliases)
            self.assertIn("牛宝宝", snapshot.target_users[0].facts[0].claim_text)

    def test_alias_correction_supersedes_only_denied_alias(self) -> None:
        with project_temp_directory() as tmp:
            config = test_config(Path(tmp) / "bot.sqlite3")
            storage = BotStorage.from_config(config)
            storage.setup()
            first = storage.record_fact_candidates(
                [
                    FactCandidate(
                        subject_user_id="123",
                        fact_type="identity",
                        claim_text="QQ:123 的称呼是牛宝宝。",
                        topic="称呼",
                        stance="neutral",
                        confidence=0.95,
                        evidence_message_id="m-old",
                        evidence_text="我叫牛宝宝",
                        source_user_id="123",
                        source_group_id="100",
                        claim_scope="self_report",
                        importance=0.9,
                    )
                ]
            ).accepted[0]

            storage.record_fact_candidates(
                [
                    FactCandidate(
                        subject_user_id="123",
                        fact_type="identity",
                        claim_text="用户表示不是牛宝宝，是牛牛。",
                        topic="称呼",
                        stance="neutral",
                        confidence=0.95,
                        evidence_message_id="m-new",
                        evidence_text="我不是牛宝宝，是牛牛",
                        source_user_id="123",
                        source_group_id="100",
                        claim_scope="self_report",
                        importance=0.9,
                    )
                ]
            )

            old_record = storage.get_fact_record(first.id)
            old_lookup = storage.build_snapshot(
                MessageContext("100", "42", "q-old", "谁是牛宝宝", "谁是牛宝宝")
            )
            new_lookup = storage.build_snapshot(
                MessageContext("100", "42", "q-new", "谁是牛牛", "谁是牛牛")
            )

            self.assertEqual(old_record.status if old_record else None, "superseded")
            self.assertEqual(old_lookup.target_users, [])
            self.assertEqual(old_lookup.unknown_name_refs, ["牛宝宝"])
            self.assertEqual(new_lookup.target_users[0].user_id, "123")
            self.assertIn("牛牛", new_lookup.target_users[0].aliases)

    def test_forgotten_and_stale_low_importance_facts_do_not_enter_context(self) -> None:
        with project_temp_directory() as tmp:
            config = test_config(Path(tmp) / "bot.sqlite3")
            storage = BotStorage.from_config(config)
            storage.setup()
            low = storage.record_fact_candidates(
                [
                    FactCandidate(
                        subject_user_id="42",
                        fact_type="opinion",
                        claim_text="用户42觉得一次截图里的按钮很亮。",
                        topic="截图按钮",
                        stance="neutral",
                        confidence=0.8,
                        evidence_message_id="m-low-importance",
                        evidence_text="image_index=0 description=button",
                        source_user_id="42",
                        source_group_id="100",
                        claim_scope="self_report",
                        importance=0.2,
                    )
                ]
            ).accepted[0]

            with storage._connect() as conn:
                conn.execute(
                    "UPDATE member_facts SET last_seen_at = 1, updated_at = 1 WHERE id = ?",
                    (low.id,),
                )

            self.assertEqual(storage.list_user_facts("42"), [])
            self.assertEqual(storage.list_user_facts("42", include_faded=True)[0].id, low.id)
            storage.forget_fact(low.id)

            self.assertEqual(storage.list_user_facts("42"), [])
            self.assertEqual(storage.get_fact_record(low.id).status, "forgotten")

    def test_low_value_fact_is_rejected(self) -> None:
        with project_temp_directory() as tmp:
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
        with project_temp_directory() as tmp:
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

    def test_user_can_manage_only_own_pending_facts(self) -> None:
        with project_temp_directory() as tmp:
            config = test_config(Path(tmp) / "bot.sqlite3")
            storage = BotStorage.from_config(config)
            storage.setup()

            write = storage.record_fact_candidates(
                [
                    FactCandidate(
                        subject_user_id="77",
                        fact_type="preference",
                        claim_text="用户77喜欢喝茶",
                        topic="喝茶",
                        stance="positive",
                        confidence=0.9,
                        evidence_message_id="m1",
                        evidence_text="77喜欢喝茶",
                        source_user_id="42",
                        source_group_id="100",
                        claim_scope="third_party",
                    ),
                    FactCandidate(
                        subject_user_id="88",
                        fact_type="preference",
                        claim_text="用户88喜欢咖啡",
                        topic="咖啡",
                        stance="positive",
                        confidence=0.9,
                        evidence_message_id="m2",
                        evidence_text="88喜欢咖啡",
                        source_user_id="42",
                        source_group_id="100",
                        claim_scope="third_party",
                    ),
                ]
            )
            pending_by_subject = {fact.subject_user_id: fact for fact in write.pending}
            fact_77 = pending_by_subject["77"]
            fact_88 = pending_by_subject["88"]

            self.assertIsNone(storage.approve_user_pending_fact("77", fact_88.id))
            record_88 = storage.get_fact_record(fact_88.id)
            self.assertIsNotNone(record_88)
            self.assertEqual(record_88.status, "pending_confirmation")  # type: ignore[union-attr]

            approved = storage.approve_user_pending_fact("QQ:77", fact_77.id)
            self.assertIsNotNone(approved)
            self.assertEqual(approved.status, "accepted")  # type: ignore[union-attr]

            self.assertFalse(storage.reject_user_pending_fact("77", fact_88.id))
            record_88 = storage.get_fact_record(fact_88.id)
            self.assertIsNotNone(record_88)
            self.assertEqual(record_88.status, "pending_confirmation")  # type: ignore[union-attr]

            self.assertTrue(storage.reject_user_pending_fact("88", fact_88.id))
            record_88 = storage.get_fact_record(fact_88.id)
            self.assertIsNotNone(record_88)
            self.assertEqual(record_88.status, "rejected")  # type: ignore[union-attr]

    def test_high_trust_third_party_fact_is_accepted(self) -> None:
        with project_temp_directory() as tmp:
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
        with project_temp_directory() as tmp:
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
        with project_temp_directory() as tmp:
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
        with project_temp_directory() as tmp:
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
        with project_temp_directory() as tmp:
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

    def test_relationship_summary_filters_low_value_message_log_items(self) -> None:
        with project_temp_directory() as tmp:
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

    def test_relationship_ranking_formats_top_five_by_closeness_and_familiarity(self) -> None:
        with project_temp_directory() as tmp:
            config = test_config(Path(tmp) / "bot.sqlite3")
            storage = BotStorage.from_config(config)
            storage.setup()
            storage.record_message(
                MessageContext(
                    group_id="100",
                    user_id="2",
                    message_id="m-name",
                    plain_text="晚上好",
                    raw_message="晚上好",
                    sender_name="Alice",
                    sender_nickname="Ali",
                    timestamp=20,
                )
            )
            for user_id, closeness, familiarity, trust in (
                ("1", 10, 1, 0),
                ("2", 4, 10, 2),
                ("3", 6, 6, 0),
                ("4", 5, 5, 0),
                ("5", 1, 8, 0),
                ("6", 8, 0, 0),
            ):
                storage.apply_relationship_delta(
                    "100",
                    user_id,
                    RelationDelta(
                        closeness=closeness,
                        familiarity=familiarity,
                        trust=trust,
                        summary_patch=f"和 {user_id} 聊过",
                    ),
                )

            ranking = storage.format_relationship_ranking("100")

            self.assertIn("本群亲密/了解程度 TOP 5", ranking)
            self.assertIn("Alice(QQ:2)", ranking)
            self.assertIn("综合=14", ranking)
            self.assertLess(ranking.index("Alice(QQ:2)"), ranking.index("QQ:3"))
            self.assertLess(ranking.index("QQ:3"), ranking.index("QQ:1"))
            self.assertNotIn("聊过", ranking)
            self.assertNotIn("|", ranking)
            self.assertNotIn("QQ:6", ranking)
            self.assertEqual(storage.format_relationship_ranking("404"), "本群暂无关系记录。")

    def test_image_description_update_keeps_unsampled_images_blank(self) -> None:
        with project_temp_directory() as tmp:
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

    def test_last_decision_includes_proactive_value_metadata(self) -> None:
        with project_temp_directory() as tmp:
            config = test_config(Path(tmp) / "bot.sqlite3")
            storage = BotStorage.from_config(config)
            storage.setup()
            context = MessageContext(
                group_id="100",
                user_id="42",
                message_id="m-decision",
                plain_text="聊聊这个活动",
                raw_message="聊聊这个活动",
            )
            storage.record_decision(
                context,
                ParticipationDecision(
                    "observe",
                    "proactive value gate rejected",
                    "active",
                    0.4,
                    "agreement",
                    0.2,
                    "busy",
                ),
                None,
            )

            text = storage.get_last_decision("100")

            self.assertIn("value=agreement:0.20", text)
            self.assertIn("traffic=busy", text)


if __name__ == "__main__":
    unittest.main()


