from __future__ import annotations

import unittest
from pathlib import Path

from qq_llm_bot.cognitive_storage import BotStorage
from qq_llm_bot.guesswho import (
    GUESSWHO_ACTIVE_REPLY,
    active_guesswho_game,
    clear_guesswho_games,
    finish_guesswho_game,
    list_guesswho_candidates,
    start_guesswho_game,
)
from qq_llm_bot.models import (
    FactRecord,
    MemoryRecord,
    RelationDelta,
    RelationshipState,
    UserProfileRecord,
)
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

    async def test_guesswho_locks_until_tellmewho_finishes_game(self) -> None:
        storage = FakeGuesswhoStorage(
            familiar_user_ids=["42"],
            facts={"42": [_fact(1), _fact(2), _fact(3)]},
        )
        llm = FakeLLM(
            [
                '{"intro":"爱研究工具，做事谨慎。","facts":["喜欢咖啡","常聊工程工具","会整理经验"]}',
                '{"intro":"第二题。","facts":["事实A","事实B","事实C"]}',
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
        self.assertEqual(locked, GUESSWHO_ACTIVE_REPLY)
        self.assertEqual(game.answer_user_id, "42")  # type: ignore[union-attr]
        self.assertEqual(active_guesswho_game("100").answer_user_id, "42")  # type: ignore[union-attr]
        self.assertIn("第二题", unlocked)
        self.assertEqual(llm.text_call_purposes, ["guesswho", "guesswho"])

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
            ['{"intro":"爱喝咖啡，也常聊技术。","facts":["用户42喜欢咖啡","QQ:42常聊 Python","会研究工具"]}']
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


if __name__ == "__main__":
    unittest.main()
