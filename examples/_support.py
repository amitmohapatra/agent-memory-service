"""What every example shares: a memory service to talk to, with nothing to start first.

By default an example runs **offline**: the service is built in this process (``create_app``)
and the SDK reaches it through ``httpx.ASGITransport``, so there is no server, no port, no
network and no model. The stand-ins are the ones the test suite runs on (``Overrides``):
in-process search, cache, authorization and blob store, jobs run inline, the hash embedding
and the lexical NLI. Retrieval quality under them is meaningless; the calls, the scope rules
and the durability are the real ones.

The one thing it needs is PostgreSQL, the service's source of truth, which has no stand-in:
``postgresql+psycopg://memory:memory@localhost:5432/memory_examples`` unless
``MEMORY_EXAMPLES_DATABASE_URL`` says otherwise (``make dev-up`` or any local PostgreSQL 16).
The database is created and migrated on first use and emptied at the start of every run, so
it must be one of its own: a URL whose database name does not end in ``examples`` is refused.

Set ``EXAMPLES_LIVE=1`` to run an example against a running service instead
(``MEMORY_URL``, ``TRELLIS_API_KEY``; ``make dev-up`` or ``uv run python examples/_serve.py``).
The two examples that onboard tenants also need ``TRELLIS_BOOTSTRAP_KEY`` live, and say so.
"""

from __future__ import annotations

import os
import secrets
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

import httpx

from trellis.memory import MemoryClient

REPO = Path(__file__).resolve().parents[1]
FIXTURES = REPO / "tests" / "fixtures"
DATABASE_URL = os.environ.get(
    "MEMORY_EXAMPLES_DATABASE_URL",
    "postgresql+psycopg://memory:memory@localhost:5432/memory_examples",
)
LIVE = os.environ.get("EXAMPLES_LIVE") == "1"
#: the development key of the offline service (and of the local compose stack)
DEV_KEY = "dev-key"


@dataclass
class Service:
    """A memory service to make clients for: in this process, or the one at ``MEMORY_URL``."""

    url: str
    transport: httpx.AsyncBaseTransport | None
    #: the key an example uses when it does not onboard a tenant of its own
    api_key: str
    #: the platform operator's key; ``None`` when the service has none
    bootstrap_key: str | None

    def client(self, api_key: str | None = None) -> MemoryClient:
        """An SDK client with ``api_key`` (default: the service's own key)."""
        key = api_key or self.api_key
        if self.transport is None:
            return MemoryClient(self.url, api_key=key)
        http = httpx.AsyncClient(transport=self.transport, base_url=self.url)
        return MemoryClient(self.url, api_key=key, http_client=http)


def _database_name(url: str) -> str:
    return urlsplit(url).path.rsplit("/", 1)[-1]


def _prepare_database(url: str) -> None:
    """Create the examples database if it is missing, migrate it, and empty it."""
    import psycopg
    from alembic import command
    from alembic.config import Config

    name = _database_name(url)
    if not name.endswith("examples"):
        sys.exit(
            f"refusing to use database {name!r}: the examples empty their database on every "
            "run, so MEMORY_EXAMPLES_DATABASE_URL must name one of its own (ending in 'examples')"
        )
    plain = url.replace("postgresql+psycopg://", "postgresql://", 1)
    admin = plain.rsplit("/", 1)[0] + "/postgres"
    try:
        with psycopg.connect(admin, autocommit=True, connect_timeout=5) as conn:
            exists = conn.execute(
                "SELECT 1 FROM pg_database WHERE datname = %s", (name,)
            ).fetchone()
            if not exists:
                conn.execute(f'CREATE DATABASE "{name}"')
    except psycopg.OperationalError as exc:
        sys.exit(
            f"PostgreSQL is not reachable for the examples ({exc.__class__.__name__}). Start one "
            "(`make dev-up`, or any local PostgreSQL 16 with user/password memory) or set "
            "MEMORY_EXAMPLES_DATABASE_URL."
        )

    # No ini file: alembic's file logging would print every migration on every run.
    cfg = Config()
    cfg.set_main_option("script_location", str(REPO / "migrations"))
    cfg.set_main_option("sqlalchemy.url", url)
    command.upgrade(cfg, "head")

    with psycopg.connect(plain, autocommit=True) as conn:
        tables = [
            row[0]
            for row in conn.execute(
                "SELECT tablename FROM pg_tables WHERE schemaname = 'public' "
                "AND tablename <> 'alembic_version' AND tablename NOT LIKE 'procrastinate%'"
            )
        ]
        if tables:
            conn.execute(
                "TRUNCATE " + ", ".join(f'"{t}"' for t in tables) + " RESTART IDENTITY CASCADE"
            )


@asynccontextmanager
async def service(*, onboarding: bool = False) -> AsyncIterator[Service]:
    """The service an example talks to.

    ``onboarding=False``: development mode, the way a laptop runs it. The key ``dev-key``
    trusts the scope headers it is sent and acts in the tenant ``default`` unless a call names
    another. ``onboarding=True``: ``api_key`` mode, the way a deployment runs it. Only the
    platform operator's bootstrap key exists at first; it onboards tenants, and every other
    caller presents a key the service issued.
    """
    if LIVE:
        bootstrap = os.environ.get("TRELLIS_BOOTSTRAP_KEY")
        if onboarding and not bootstrap:
            print("skipped: live onboarding needs TRELLIS_BOOTSTRAP_KEY (the operator's key)")
            raise SystemExit(0)
        yield Service(
            url=os.environ.get("MEMORY_URL", "http://localhost:8080"),
            transport=None,
            api_key=os.environ.get("TRELLIS_API_KEY", DEV_KEY),
            bootstrap_key=bootstrap,
        )
        return

    # The service's own configuration is cleared first: an example is the same program on
    # every machine, whatever the shell or a .env file says.
    for name in [k for k in os.environ if k.startswith("MEMORY__")]:
        del os.environ[name]
    for name in ("BIFROST_URL", "BIFROST_VIRTUAL_KEY", "OTEL_EXPORTER_OTLP_ENDPOINT"):
        os.environ.pop(name, None)

    _prepare_database(DATABASE_URL)

    from memory_service.api.app import create_app
    from memory_service.application.container import Overrides
    from memory_service.config.settings import Settings

    # generated per run and held only in this process; it is never printed
    bootstrap = secrets.token_urlsafe(32) if onboarding else None
    authentication: dict[str, object] = (
        {"bootstrap_admin_key": bootstrap} if onboarding else {"trusted_dev_api_keys": [DEV_KEY]}
    )
    settings = Settings(
        _env_file=None,  # type: ignore[call-arg]
        service={"environment": "dev", "log_json": False, "log_level": "ERROR"},
        database={"url": DATABASE_URL},
        authentication=authentication,
    )
    overrides = Overrides(
        cache="memory",
        search="memory",
        tasks="inline",
        authorization="memory",
        blob="memory",
        embedding="hash",
        nli="lexical",
        document_parser="builtin",
    )
    app = create_app(settings, overrides=overrides)
    async with app.router.lifespan_context(app):
        yield Service(
            url="http://memory.example",
            transport=httpx.ASGITransport(app=app),
            api_key=DEV_KEY,
            bootstrap_key=bootstrap,
        )


def run_id() -> str:
    """A short id that keeps one run's threads apart from another's on a live service."""
    return secrets.token_hex(4)
