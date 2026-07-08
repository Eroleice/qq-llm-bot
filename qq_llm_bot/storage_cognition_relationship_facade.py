from __future__ import annotations

import qq_llm_bot.storage_relationships as _storage_relationships
from qq_llm_bot.models import RelationDelta, RelationshipState


class StorageCognitionRelationshipFacadeMixin:
    def get_relationship(self, group_id: str, user_id: str) -> RelationshipState:
        return _storage_relationships.get_relationship(self, group_id, user_id)

    def apply_relationship_delta(self, group_id: str, user_id: str, delta: RelationDelta) -> RelationshipState:
        return _storage_relationships.apply_relationship_delta(self, group_id, user_id, delta)

    def touch_relationship(self, group_id: str, user_id: str, familiarity_delta: int = 1) -> None:
        _storage_relationships.touch_relationship(self, group_id, user_id, familiarity_delta)

    def format_relationship(self, group_id: str, user_id: str) -> str:
        return _storage_relationships.format_relationship(self, group_id, user_id)

    def format_relationship_ranking(self, group_id: str, limit: int = 5) -> str:
        return _storage_relationships.format_relationship_ranking(self, group_id, limit)
