from __future__ import annotations

import qq_llm_bot.storage_guesswho as _storage_guesswho
from qq_llm_bot.knowledge_models import GuessWhoScoreRecord


class StorageGuessWhoFacadeMixin:
    def record_guesswho_result(
        self,
        group_id: str,
        user_id: str,
        *,
        correct: bool,
        updated_at: int | None = None,
    ) -> None:
        _storage_guesswho.record_guesswho_result(
            self,
            group_id,
            user_id,
            correct=correct,
            updated_at=updated_at,
        )

    def list_guesswho_scores(
        self,
        group_id: str,
        *,
        wrong: bool = False,
        limit: int = 10,
    ) -> list[GuessWhoScoreRecord]:
        return _storage_guesswho.list_guesswho_scores(
            self,
            group_id,
            wrong=wrong,
            limit=limit,
        )
