from __future__ import annotations

from qq_llm_bot.storage_cognition_audit_facade import StorageCognitionAuditFacadeMixin
from qq_llm_bot.storage_cognition_fact_facade import StorageCognitionFactFacadeMixin
from qq_llm_bot.storage_cognition_memory_facade import StorageCognitionMemoryFacadeMixin
from qq_llm_bot.storage_cognition_persona_facade import StorageCognitionPersonaFacadeMixin
from qq_llm_bot.storage_cognition_profile_facade import StorageCognitionProfileFacadeMixin
from qq_llm_bot.storage_cognition_relationship_facade import (
    StorageCognitionRelationshipFacadeMixin,
)


class StorageCognitionFacadeMixin(
    StorageCognitionMemoryFacadeMixin,
    StorageCognitionFactFacadeMixin,
    StorageCognitionProfileFacadeMixin,
    StorageCognitionRelationshipFacadeMixin,
    StorageCognitionAuditFacadeMixin,
    StorageCognitionPersonaFacadeMixin,
):
    pass
