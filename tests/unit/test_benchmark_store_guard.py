"""A harness resets whole stores, so it may only ever point at a store it owns.

``reset_store`` TRUNCATEs 25 tables and deletes the tenant's vectors. The benchmark default
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
    ["memory_bench_conv", "memory_bench_docs", "memory_hi_locomo", "p7_locomo", "p7_retrieval"],
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
