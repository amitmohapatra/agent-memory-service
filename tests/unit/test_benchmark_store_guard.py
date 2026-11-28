"""A harness resets whole stores, so it may only ever point at a store it owns.

``reset_store`` deletes a tenant's rows from 25 tables and its vectors. The benchmark default
database URL names the shared ``memory`` database that a deployed API reads, so the name of
the target is the only thing between a measurement and someone else's data. These are the
refusals; they are asserted here rather than in each harness because the guard lives at the
TRUNCATE, which every harness passes through.
"""

from __future__ import annotations

import pytest
from benchmark.common import (
    DEDICATED_DATABASES,
    ISOLATED_QDRANT_PORT,
    dedicated_database,
    isolated_qdrant,
)

from tests.conftest import DB_URL

pytestmark = pytest.mark.unit

#: the suite's own server, with the database name swapped: only ``conftest.DB_URL`` may spell
#: a DSN, and what is under test here is the database NAME, not the host
HOST = DB_URL.rsplit("/", 1)[0]


@pytest.mark.parametrize(
    "database",
    [
        "memory_bench_conv",
        "memory_bench_docs",
        "memory_hi_locomo",
        "p7_locomo",
        "p7_retrieval",
        "p9_locomo",
        "p9_integration",
    ],
)
def test_a_dedicated_benchmark_database_is_accepted(database: str) -> None:
    assert dedicated_database(f"{HOST}/{database}") == database


@pytest.mark.parametrize("database", ["memory", "harness_live", "postgres", "memory_prod", ""])
def test_a_shared_or_deployed_database_is_refused(database: str) -> None:
    """``memory`` is the benchmark default and a deployed store at the same time: the single
    most likely way to lose a deployment to a forgotten environment variable."""
    with pytest.raises(ValueError, match="not a benchmark database"):
        dedicated_database(f"{HOST}/{database}")


def test_the_refusal_names_what_would_be_accepted() -> None:
    with pytest.raises(ValueError) as caught:
        dedicated_database(f"{HOST}/memory")
    message = str(caught.value)
    assert all(prefix in message for prefix in DEDICATED_DATABASES)
    # the refusal names the database, never the DSN it came from (credentials included)
    assert HOST not in message


def test_only_the_isolated_qdrant_is_accepted() -> None:
    url = f"http://host.docker.internal:{ISOLATED_QDRANT_PORT}"
    assert isolated_qdrant(url) == url
    with pytest.raises(ValueError, match="not the isolated Qdrant"):
        isolated_qdrant("http://host.docker.internal:6333")


async def test_a_reset_clears_its_own_tenant_and_no_other() -> None:
    """Two corpora share one benchmark database under their own tenants
    (``BENCH_CORPUS_TENANT_PREFIX``): resetting one deletes its rows and vectors and leaves
    the other's. It TRUNCATEd every table, and a second corpus wiped the first one's rows
    while its vectors stayed behind."""
    from types import SimpleNamespace

    from benchmark.common import reset_store
    from benchmark.retrieval import TABLES

    said: list[tuple[str, dict]] = []

    class Result:
        def scalars(self):
            return ["memories", "observations", "messages"]

    class Conn:
        async def execute(self, statement, params=None):
            said.append((str(statement), dict(params or {})))
            return Result()

    class Begin:
        async def __aenter__(self):
            return Conn()

        async def __aexit__(self, *exc):
            return False

    deleted: list[tuple[str, str]] = []

    async def delete_by_filter(name, flt):
        deleted.append((name, flt.tenant_id))
        return 1

    url = SimpleNamespace(render_as_string=lambda hide_password: f"{HOST}/p7_locomo")
    container = SimpleNamespace(
        database=SimpleNamespace(engine=SimpleNamespace(url=url, begin=Begin)),
        services={"indexer": SimpleNamespace(collection=lambda base: f"c_{base}")},
        search=SimpleNamespace(delete_by_filter=delete_by_filter),
    )
    await reset_store(container, "llm_conv_0")
    sql = [s for s, _ in said]
    assert not any("TRUNCATE" in s for s in sql)
    deletes = [(s, p) for s, p in said if s.startswith("DELETE")]
    assert [s.split()[2] for s, _ in deletes] == [
        t
        for t in (x.strip() for x in TABLES.split(","))
        if t in ("memories", "observations", "messages")
    ]
    assert all(
        p == {"tenant": "llm_conv_0"} and "WHERE tenant_id = :tenant" in s for s, p in deletes
    )
    assert {t for _, t in deleted} == {"llm_conv_0"}
