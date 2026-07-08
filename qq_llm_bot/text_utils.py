from __future__ import annotations


def safe_choice(value: str, allowed: set[str], fallback: str) -> str:
    return value if value in allowed else fallback


def normalize_lexicon_term(term: str) -> str:
    return " ".join(str(term).strip().lower().split())


def lexicon_subject(term: str) -> str:
    return f"term:{normalize_lexicon_term(term)}"


def safe_path_part(value: str, *, limit: int = 80, fallback: str = "unknown") -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(value))
    return cleaned[: max(1, int(limit))] or fallback
