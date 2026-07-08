from __future__ import annotations

from typing import Literal

DecisionAction = Literal["observe", "reply", "proactive_reply"]
ParticipationValueType = Literal[
    "none",
    "direct_reply",
    "answer",
    "synthesis",
    "missing_angle",
    "useful_context",
    "clarifying_question",
    "humor",
    "agreement",
    "empathy",
    "rephrase",
]
MemoryStatus = Literal["active", "candidate", "pending_confirmation", "conflict", "rejected", "forgotten"]
ClaimScope = Literal["self_report", "third_party", "bot_directed", "group_fact"]
VerificationStatus = Literal["accepted", "pending_confirmation", "conflict", "rejected"]
FactStatus = Literal["candidate", "accepted", "pending_confirmation", "rejected", "superseded", "forgotten"]
