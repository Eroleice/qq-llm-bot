from __future__ import annotations

import qq_llm_bot.storage_access as _storage_access
from qq_llm_bot.config import ParticipationMode


class StorageAccessFacadeMixin:
    def is_group_enabled(self, group_id: str) -> bool:
        return _storage_access.is_group_enabled(self, group_id)

    def set_group_enabled(self, group_id: str, enabled: bool) -> None:
        _storage_access.set_group_enabled(self, group_id, enabled)

    def list_enabled_groups(self) -> list[str]:
        return _storage_access.list_enabled_groups(self)

    def get_group_mode(self, group_id: str, default_mode: ParticipationMode) -> ParticipationMode:
        return _storage_access.get_group_mode(self, group_id, default_mode)

    def set_group_mode(self, group_id: str, mode: ParticipationMode) -> None:
        _storage_access.set_group_mode(self, group_id, mode)

    def is_admin(self, user_id: str) -> bool:
        return _storage_access.is_admin(self, user_id)

    def add_admin(self, user_id: str) -> None:
        _storage_access.add_admin(self, user_id)

    def remove_admin(self, user_id: str) -> None:
        _storage_access.remove_admin(self, user_id)

    def list_admins(self) -> list[str]:
        return _storage_access.list_admins(self)

    def is_user_ignored(self, user_id: str) -> bool:
        return _storage_access.is_user_ignored(self, user_id)

    def add_ignored_user(self, user_id: str) -> None:
        _storage_access.add_ignored_user(self, user_id)

    def remove_ignored_user(self, user_id: str) -> None:
        _storage_access.remove_ignored_user(self, user_id)

    def list_ignored_users(self) -> list[str]:
        return _storage_access.list_ignored_users(self)
