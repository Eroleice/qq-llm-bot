from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class FinalQAResult:
    allowed: bool
    reason: str = ""
    categories: tuple[str, ...] = ()
    confidence: float = 0.0


FINAL_QA_CATEGORIES = {
    "context_mismatch",
    "political_stance",
    "inappropriate",
    "privacy",
    "unsafe_self_claim",
    "system_leak",
    "low_value",
    "other",
}


def safe_final_qa_categories(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    categories: list[str] = []
    for item in value:
        category = str(item).strip()
        if category in FINAL_QA_CATEGORIES and category not in categories:
            categories.append(category)
        if len(categories) >= 5:
            break
    return tuple(categories)


def final_qa_category_for_reason(reason: str) -> str:
    if reason in FINAL_QA_CATEGORIES:
        return reason
    if "political" in reason:
        return "political_stance"
    if "privacy" in reason:
        return "privacy"
    if "system" in reason:
        return "system_leak"
    if "self" in reason:
        return "unsafe_self_claim"
    return "other"
