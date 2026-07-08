from __future__ import annotations


CONFLICT_SENSITIVE_KINDS = {
    "alias",
    "identity",
    "location",
    "preference",
    "dislike",
    "experience",
    "self_experience",
    "self_preference",
    "self_boundary",
    "persona_fact",
}
SENSITIVE_CONFIRMATION_KINDS = {"identity", "location"}
TRUSTED_THIRD_PARTY_THRESHOLD = 70
FACT_INACTIVE_STATUSES = {"rejected", "superseded", "forgotten"}
PROTECTED_FACT_TYPES = {"identity", "boundary", "skill", "habit"}
