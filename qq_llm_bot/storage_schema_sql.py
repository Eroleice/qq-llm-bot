from __future__ import annotations

from qq_llm_bot.storage_schema_access_sql import ACCESS_SCHEMA_SQL
from qq_llm_bot.storage_schema_audit_sql import AUDIT_SCHEMA_SQL
from qq_llm_bot.storage_schema_cognition_sql import COGNITION_SCHEMA_SQL
from qq_llm_bot.storage_schema_game_sql import GAME_SCHEMA_SQL
from qq_llm_bot.storage_schema_media_sql import MEDIA_SCHEMA_SQL
from qq_llm_bot.storage_schema_message_sql import MESSAGE_SCHEMA_SQL


SCHEMA_SQL = "\n\n".join(
    (
        MESSAGE_SCHEMA_SQL,
        MEDIA_SCHEMA_SQL,
        ACCESS_SCHEMA_SQL,
        COGNITION_SCHEMA_SQL,
        AUDIT_SCHEMA_SQL,
        GAME_SCHEMA_SQL,
    )
)
