from __future__ import annotations

import qq_llm_bot.storage_profiles as _storage_profiles
from qq_llm_bot.models import FactRecord, UserProfileDraft, UserProfileRecord


class StorageCognitionProfileFacadeMixin:
    def get_user_profile(self, user_id: str) -> UserProfileRecord | None:
        return _storage_profiles.get_user_profile(self, user_id)

    def should_update_user_profile(self, user_id: str, threshold: int | None = None) -> bool:
        return _storage_profiles.should_update_user_profile(self, user_id, threshold)

    def maybe_update_user_profile(
        self,
        user_id: str,
        draft: UserProfileDraft,
        facts: list[FactRecord],
        force: bool = False,
    ) -> UserProfileRecord | None:
        return _storage_profiles.maybe_update_user_profile(self, user_id, draft, facts, force)

    def clear_user_profile(self, user_id: str) -> bool:
        return _storage_profiles.clear_user_profile(self, user_id)

    def format_user_profile(self, user_id: str) -> str:
        return _storage_profiles.format_user_profile(self, user_id)
