from __future__ import annotations

import unittest

from qq_llm_bot.models import FactRecord, MemoryRecord, RelationshipState, UserProfileRecord
from qq_llm_bot.whoami import WHOAMI_NOT_ENOUGH_FACTS_REPLY, build_whoami_reply
from tests.helpers import FakeLLM


class FakeWhoamiStorage:
    def __init__(
        self,
        facts: list[FactRecord],
        *,
        memories: list[MemoryRecord] | None = None,
        profile: UserProfileRecord | None = None,
        relationship: RelationshipState | None = None,
    ) -> None:
        self.facts = facts
        self.memories = memories or []
        self.profile = profile
        self.relationship = relationship
        self.fact_calls: list[dict[str, object]] = []

    def list_user_facts(
        self,
        user_id: str,
        limit: int = 20,
        status: str = "accepted",
        group_id: str = "",
        include_faded: bool = False,
    ) -> list[FactRecord]:
        self.fact_calls.append(
            {
                "user_id": user_id,
                "limit": limit,
                "status": status,
                "group_id": group_id,
                "include_faded": include_faded,
            }
        )
        return self.facts[:limit]

    def get_user_profile(self, user_id: str) -> UserProfileRecord | None:
        return self.profile

    def list_memories(
        self,
        owner_type: str,
        owner_id: str,
        limit: int = 8,
        status: str = "active",
    ) -> list[MemoryRecord]:
        return self.memories[:limit]

    def get_relationship(self, group_id: str, user_id: str) -> RelationshipState | None:
        return self.relationship


class WhoamiTests(unittest.IsolatedAsyncioTestCase):
    async def test_whoami_replies_not_familiar_before_five_facts(self) -> None:
        storage = FakeWhoamiStorage([_fact(index) for index in range(4)])
        llm = FakeLLM(["不会被使用"])

        reply = await build_whoami_reply(storage, llm, "42", group_id="100")

        self.assertEqual(reply, WHOAMI_NOT_ENOUGH_FACTS_REPLY)
        self.assertEqual(llm.text_calls, [])
        self.assertEqual(storage.fact_calls[0]["include_faded"], True)

    async def test_whoami_summarizes_with_llm_after_five_facts(self) -> None:
        expected = "你喜欢咖啡，也常聊技术；做事谨慎，愿意把新工具试清楚再安利。"
        storage = FakeWhoamiStorage(
            [
                _fact(1, claim_text="用户42喜欢咖啡", topic="咖啡"),
                _fact(2, claim_text="用户42常聊 Python 工程问题", topic="Python"),
                _fact(3, claim_text="用户42做决定前会先比较方案", topic="决策习惯"),
                _fact(4, claim_text="用户42愿意尝试新工具", topic="工具"),
                _fact(5, claim_text="用户42喜欢把经验整理成笔记", topic="笔记"),
            ],
            memories=[
                MemoryRecord(
                    id=1,
                    owner_type="user",
                    owner_id="42",
                    kind="habit",
                    content="常整理项目笔记",
                    confidence=0.9,
                    importance=0.7,
                    status="active",
                    updated_at=1,
                )
            ],
            profile=UserProfileRecord(
                user_id="42",
                summary="偏好技术讨论，习惯先验证再判断。",
                traits={"interests": ["工程工具"]},
                fact_count=5,
                version=1,
                updated_at=1,
            ),
            relationship=RelationshipState(
                group_id="100",
                user_id="42",
                familiarity=6,
                trust=3,
                summary="经常和可可讨论技术选择。",
            ),
        )
        llm = FakeLLM([expected])

        reply = await build_whoami_reply(
            storage,
            llm,
            "42",
            group_id="100",
            display_name="Alice",
        )

        self.assertEqual(reply, expected)
        self.assertEqual(llm.text_call_purposes, ["whoami"])
        self.assertIn("不超过 80 字", llm.text_calls[0][1])
        self.assertIn("Alice(QQ:42)", llm.text_calls[0][1])
        self.assertIn("用户42喜欢咖啡", llm.text_calls[0][1])
        self.assertIn("常整理项目笔记", llm.text_calls[0][1])


def _fact(
    index: int,
    *,
    claim_text: str | None = None,
    topic: str | None = None,
) -> FactRecord:
    return FactRecord(
        id=index,
        subject_user_id="42",
        fact_type="preference",
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
