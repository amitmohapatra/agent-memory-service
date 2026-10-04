"""Alembic environment. Uses the service settings for the URL; sync psycopg3 driver."""

from __future__ import annotations

import os
from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool, text

from memory_service.adapters.db.orm import Base
from memory_service.config.settings import get_settings

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def include_object(obj, name, type_, reflected, compare_to):
    """Ignore tables owned by other tools (Procrastinate manages its own schema)."""
    return not (type_ == "table" and name.startswith("procrastinate_"))


#: Online migrations fail fast instead of queueing (docs/deploy/database.md). An ALTER
#: waiting on a lock a long transaction holds blocks every query that arrives after it, so a
#: migration that cannot get its lock within this fails, to be retried, rather than stalling
#: the live service. The statement bound stops a runaway backfill. Both are PostgreSQL
#: interval strings and can be raised for a maintenance window.
LOCK_TIMEOUT = os.environ.get("MEMORY_MIGRATION_LOCK_TIMEOUT", "5s")
STATEMENT_TIMEOUT = os.environ.get("MEMORY_MIGRATION_STATEMENT_TIMEOUT", "15min")


def _bound(connection) -> None:  # type: ignore[no-untyped-def]
    """Session-level, so they also hold inside ``autocommit_block`` (CONCURRENTLY)."""
    connection.execute(text("SELECT set_config('lock_timeout', :v, false)"), {"v": LOCK_TIMEOUT})
    connection.execute(
        text("SELECT set_config('statement_timeout', :v, false)"), {"v": STATEMENT_TIMEOUT}
    )
    connection.commit()


def _url() -> str:
    override = config.get_main_option("sqlalchemy.url")
    return override or get_settings().database.sync_url


def run_migrations_offline() -> None:
    context.configure(
        url=_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        include_object=include_object,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    section = config.get_section(config.config_ini_section) or {}
    section["sqlalchemy.url"] = _url()
    connectable = engine_from_config(section, prefix="sqlalchemy.", poolclass=pool.NullPool)
    with connectable.connect() as connection:
        _bound(connection)
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
            include_object=include_object,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
