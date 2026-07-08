from __future__ import annotations

import sqlite3

from qq_llm_bot.storage_migrations import migrate_storage_schema
from qq_llm_bot.storage_schema_sql import SCHEMA_SQL
from qq_llm_bot.storage_seed import seed_storage_config


__all__ = [
    "SCHEMA_SQL",
    "create_storage_schema",
    "migrate_storage_schema",
    "seed_storage_config",
]


def create_storage_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA_SQL)
