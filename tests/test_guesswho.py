from __future__ import annotations

import json
import unittest
from pathlib import Path

from qq_llm_bot.cognitive_storage import BotStorage
from qq_llm_bot.guesswho import (
    GUESSWHO_ACTIVE_REPLY,
    active_guesswho_game,
    clear_guesswho_games,
    finish_guesswho_game,
    format_guesswho_ranking,
    list_guesswho_candidates,
    request_guesswho_hint,
    reveal_guesswho_answer,
    start_guesswho_game,
    submit_guesswho_guess,
)
from qq_llm_bot.models import (
    FactRecord,
    MemoryRecord,
    RelationDelta,
    RelationshipState,
    UserProfileRecord,
)
from qq_llm_bot.knowledge_models import GuessWhoScoreRecord
from tests.helpers import FakeLLM, project_temp_directory, test_config


class FakeGuesswhoStorage:
    def __init__(
        self,
        *,
        familiar_user_ids: list[str],
        facts: dict[str, list[FactRecord]] | None = None,
        memories: dict[str, list[MemoryRecord]] | None = None,
        profile: UserProfileRecord | None = None,
    ) -> None:
        self.familiar_user_ids = familiar_user_ids
        self.facts = facts or {}
        self.memories = memories or {}
        self.profile = profile

    def list_familiar_user_ids(self, min_familiarity: int = 100) -> list[str]:
        return self.familiar_user_ids if min_familiarity == 100 else []

    def list_user_facts(
        self,
        user_id: str,
        limit: int = 20,
        status: str = "accepted",
        group_id: str = "",
        include_faded: bool = False,
    ) -> list[FactRecord]:
        return self.facts.get(user_id, [])[:limit]

    def get_user_profile(self, user_id: str) -> UserProfileRecord | None:
        return self.profile

    def list_memories(
        self,
        owner_type: str,
        owner_id: str,
        limit: int = 8,
        status: str = "active",
    ) -> list[MemoryRecord]:
        return self.memories.get(owner_id, [])[:limit]

    def get_relationship(self, group_id: str, user_id: str) -> RelationshipState:
        return RelationshipState(group_id=group_id, user_id=user_id, familiarity=100, trust=8)


class GuesswhoTests(unittest.IsolatedAsyncioTestCase):
    def tearDown(self) -> None:
        clear_guesswho_games()

    def test_candidates_intersect_familiar_users_with_current_group_members(self) -> None:
        storage = FakeGuesswhoStorage(familiar_user_ids=["42", "77", "99"])

        candidates = list_guesswho_candidates(storage, ["QQ:77", "88", "99"])

        self.assertEqual(candidates, ["77", "99"])

    def test_storage_lists_canonical_familiar_user_ids(self) -> None:
        with project_temp_directory() as tmp:
            config = test_config(Path(tmp) / "bot.sqlite3")
            storage = BotStorage.from_config(config)
            storage.setup()
            storage.apply_relationship_delta("100", "QQ:42", RelationDelta(familiarity=100))
            storage.apply_relationship_delta("100", "77", RelationDelta(familiarity=99))

            self.assertEqual(storage.list_familiar_user_ids(), ["42"])

    def test_guesswho_scores_are_persistent_and_group_scoped(self) -> None:
        with project_temp_directory() as tmp:
            config = test_config(Path(tmp) / "bot.sqlite3")
            storage = BotStorage.from_config(config)
            storage.setup()

            storage.record_guesswho_result("100", "QQ:2", correct=True, updated_at=1)
            storage.record_guesswho_result("100", "2", correct=True, updated_at=2)
            storage.record_guesswho_result("100", "2", correct=False, updated_at=3)
            storage.record_guesswho_result("100", "3", correct=True, updated_at=4)
            storage.record_guesswho_result("100", "3", correct=False, updated_at=5)
            storage.record_guesswho_result("100", "3", correct=False, updated_at=6)
            storage.record_guesswho_result("200", "3", correct=False, updated_at=7)

            correct_scores = storage.list_guesswho_scores("100")
            wrong_scores = storage.list_guesswho_scores("100", wrong=True)

            self.assertEqual(
                [(score.user_id, score.correct_count) for score in correct_scores],
                [("2", 2), ("3", 1)],
            )
            self.assertEqual(
                [(score.user_id, score.wrong_count) for score in wrong_scores],
                [("3", 2), ("2", 1)],
            )
            self.assertEqual(storage.list_guesswho_scores("200", wrong=True)[0].wrong_count, 1)

    def test_guesswho_ranking_formats_nickname_qq_and_count(self) -> None:
        scores = [
            GuessWhoScoreRecord("100", "2", correct_count=5, wrong_count=1, nickname="旧昵称"),
            GuessWhoScoreRecord("100", "3", correct_count=2, wrong_count=4),
        ]

        correct = format_guesswho_ranking(scores, member_names={"2": "新群名片"})
        wrong = format_guesswho_ranking(scores, wrong=True, member_names={"3": "小三"})

        self.assertEqual(
            correct,
            "猜对排行榜 TOP 10：\n"
            "1. 新群名片（QQ:2）— 5 次\n"
            "2. QQ用户（QQ:3）— 2 次",
        )
        self.assertIn("1. 小三（QQ:3）— 4 次", wrong)
        self.assertIn("2. 旧昵称（QQ:2）— 1 次", wrong)

    async def test_guesswho_locks_until_game_finishes(self) -> None:
        storage = FakeGuesswhoStorage(
            familiar_user_ids=["42"],
            facts={"42": [_fact(1), _fact(2), _fact(3)]},
        )
        llm = FakeLLM(
            [
                _clue_reply("爱研究工具，做事谨慎。"),
                _clue_reply("第二题。"),
            ]
        )

        reply = await start_guesswho_game(
            storage,
            llm,
            "100",
            ["42"],
            chooser=lambda candidates: candidates[0],
        )
        locked = await start_guesswho_game(
            storage,
            llm,
            "100",
            ["42"],
            chooser=lambda candidates: candidates[0],
        )
        game = finish_guesswho_game("100")
        unlocked = await start_guesswho_game(
            storage,
            llm,
            "100",
            ["42"],
            chooser=lambda candidates: candidates[0],
        )

        self.assertIn("猜猜这是谁", reply)
        self.assertIn("喜欢咖啡", reply)
        self.assertIn("#guess hint", reply)
        self.assertIn("#guess @成员", reply)
        self.assertIn("#guess answer", reply)
        self.assertIn("#guess rank", reply)
        self.assertNotIn("#guess end", reply)
        self.assertEqual(locked, GUESSWHO_ACTIVE_REPLY)
        self.assertEqual(game.answer_user_id, "42")  # type: ignore[union-attr]
        self.assertEqual(active_guesswho_game("100").answer_user_id, "42")  # type: ignore[union-attr]
        self.assertIn("第二题", unlocked)
        self.assertEqual(llm.text_call_purposes, ["guesswho", "guesswho"])
        self.assertIn("40-80字介绍", llm.text_calls[0][1])
        self.assertIn("intro 控制在 40-80 字", llm.text_calls[0][1])
        self.assertIn("9 条互不重复", llm.text_calls[0][1])

    async def test_hint_cooldowns_and_two_hint_limit(self) -> None:
        storage = FakeGuesswhoStorage(
            familiar_user_ids=["42"],
            facts={"42": [_fact(index) for index in range(1, 10)]},
        )
        llm = FakeLLM([_clue_reply("这是本轮的概括。")])
        await start_guesswho_game(
            storage,
            llm,
            "100",
            ["42"],
            chooser=lambda candidates: candidates[0],
            now=1_000,
        )

        early = request_guesswho_hint("100", now=1_119)
        first = request_guesswho_hint("100", now=1_120)
        second_early = request_guesswho_hint("100", now=1_239)
        second = request_guesswho_hint("100", now=1_240)
        exhausted = request_guesswho_hint("100", now=2_000)

        self.assertEqual((early.status, early.remaining_seconds), ("cooldown", 1))
        self.assertEqual(first.status, "ok")
        self.assertEqual(first.hint_number, 1)
        self.assertEqual(first.facts, ("提示事实4", "提示事实5", "提示事实6"))
        self.assertEqual((second_early.status, second_early.remaining_seconds), ("cooldown", 1))
        self.assertEqual(second.status, "ok")
        self.assertEqual(second.hint_number, 2)
        self.assertEqual(second.facts, ("提示事实7", "提示事实8", "提示事实9"))
        self.assertEqual(exhausted.status, "exhausted")

    async def test_answer_is_locked_for_five_minutes(self) -> None:
        storage = FakeGuesswhoStorage(familiar_user_ids=["42"])
        llm = FakeLLM([_clue_reply("这是本轮的概括。")])
        await start_guesswho_game(
            storage,
            llm,
            "100",
            ["42"],
            chooser=lambda candidates: candidates[0],
            now=2_000,
        )

        early = reveal_guesswho_answer("100", now=2_299)
        self.assertEqual((early.status, early.remaining_seconds), ("cooldown", 1))
        self.assertIsNotNone(active_guesswho_game("100"))

        revealed = reveal_guesswho_answer("100", now=2_300)
        self.assertEqual(revealed.status, "revealed")
        self.assertEqual(revealed.game.answer_user_id, "42")  # type: ignore[union-attr]
        self.assertIsNone(active_guesswho_game("100"))

    async def test_wrong_guess_keeps_lock_and_correct_guess_ends_game(self) -> None:
        storage = FakeGuesswhoStorage(familiar_user_ids=["42"])
        llm = FakeLLM([_clue_reply("这是本轮的概括。")])
        await start_guesswho_game(
            storage,
            llm,
            "100",
            ["42"],
            chooser=lambda candidates: candidates[0],
            now=3_000,
        )

        wrong = submit_guesswho_guess("100", "77")
        self.assertEqual(wrong.status, "incorrect")
        self.assertIsNotNone(active_guesswho_game("100"))

        correct = submit_guesswho_guess("100", "QQ:42")
        self.assertEqual(correct.status, "correct")
        self.assertEqual(correct.game.answer_user_id, "42")  # type: ignore[union-attr]
        self.assertIsNone(active_guesswho_game("100"))

    async def test_guesswho_prompt_excludes_identity_material_and_redacts_user_id(self) -> None:
        storage = FakeGuesswhoStorage(
            familiar_user_ids=["42"],
            facts={
                "42": [
                    _fact(1, fact_type="identity", claim_text="用户42叫小明", topic="小明"),
                    _fact(2, claim_text="用户42喜欢咖啡", topic="咖啡"),
                    _fact(3, claim_text="QQ:42常聊 Python", topic="Python"),
                ]
            },
            memories={
                "42": [
                    MemoryRecord(
                        id=1,
                        owner_type="user",
                        owner_id="42",
                        kind="alias",
                        content="小明",
                        confidence=0.9,
                        importance=0.7,
                        status="active",
                        updated_at=1,
                    )
                ]
            },
        )
        llm = FakeLLM(
            [
                json.dumps(
                    {
                        "intro": "爱喝咖啡，也常聊技术。",
                        "facts": [
                            "用户42喜欢咖啡",
                            "QQ:42常聊 Python",
                            "会研究工具",
                            "会整理经验",
                            "偏好清晰的表达",
                            "经常讨论软件",
                            "做事比较谨慎",
                            "愿意分享心得",
                            "关注实用工具",
                        ],
                    },
                    ensure_ascii=False,
                )
            ]
        )

        reply = await start_guesswho_game(
            storage,
            llm,
            "100",
            ["42"],
            chooser=lambda candidates: candidates[0],
        )
        prompt = llm.text_calls[0][1]

        self.assertNotIn("小明", prompt)
        self.assertNotIn("用户42喜欢咖啡", prompt)
        self.assertIn("这个人喜欢咖啡", prompt)
        self.assertNotIn("用户42", reply)
        self.assertNotIn("QQ:42", reply)


def _fact(
    index: int,
    *,
    fact_type: str = "preference",
    claim_text: str | None = None,
    topic: str | None = None,
) -> FactRecord:
    return FactRecord(
        id=index,
        subject_user_id="42",
        fact_type=fact_type,
        claim_text=claim_text or f"用户42稳定偏好话题{index}",
        topic=topic or f"话题{index}",
        stance="positive",
        confidence=0.9,
        status="accepted",
        claim_scope="self_report",
        source_user_id="42",
        source_group_id="100",
        evidence_message_id=f"m{index}",
        evidence_text=f"证据{index}",
        created_at=index,
        updated_at=index,
        importance=0.7,
        last_seen_at=index,
    )


def _clue_reply(intro: str) -> str:
    return json.dumps(
        {
            "intro": intro,
            "facts": ["喜欢咖啡", "常聊工程工具", "会整理经验"]
            + [f"提示事实{index}" for index in range(4, 10)],
        },
        ensure_ascii=False,
    )


if __name__ == "__main__":
    unittest.main()
