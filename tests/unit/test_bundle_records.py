"""A built bundle's record (``modules.context.handles``): the handles its prompt cites, the
evidence it carried, and how a handle resolves back to the item it named - for the scope that
was shown it only, and never at the cost of a failed call when the cache is down."""

from __future__ import annotations

import pytest

from memory_service.adapters.cache.memory_cache import MemoryCache
from memory_service.domain.context import MemoryExecutionContext
from memory_service.domain.context_bundle import (
    ContextBundle,
    ContextItem,
    ConversationWindow,
    EvidenceReport,
    UnusedEvidence,
)
from memory_service.domain.enums import EvidenceStatus, QueryType, Representation
from memory_service.domain.errors import NotFound
from memory_service.modules.context.handles import (
    RECORD_TTL_SECONDS,
    BundleRecord,
    BundleRecords,
    RecordedEvidence,
    is_handle,
    record_of,
)

pytestmark = pytest.mark.unit

ALICE = MemoryExecutionContext(tenant_id="acme", user_id="alice", workspace_id="ws1")
ALICE_RUN = MemoryExecutionContext(
    tenant_id="acme", user_id="alice", workspace_id="ws1", agent_id="buyer", agent_run_id="run_1"
)
MALLORY = MemoryExecutionContext(tenant_id="acme", user_id="mallory", workspace_id="ws1")


def _item(item_id: str, rep: Representation, text: str, **attributes: str) -> ContextItem:
    return ContextItem(
        item_id=item_id,
        representation=rep,
        text=text,
        citation=f"{rep.value}:{item_id}",
        attributes=dict(attributes),
    )


def _bundle(bundle_id: str = "bnd_1") -> ContextBundle:
    return ContextBundle(
        bundle_id=bundle_id,
        query="what is the timezone?",
        query_type=QueryType.GENERAL_SEMANTIC,
        conversation=ConversationWindow(),
        memories=[
            _item("mem_1", Representation.MEMORY, "My timezone is CET."),
            _item("mem_2", Representation.MEMORY, "Omar moved to Lisbon.", provider="llm"),
        ],
        knowledge=[_item("chk_1", Representation.CHUNK, "Section 3: EBITDA rose 4%.")],
        graph_facts=[_item("rel_1", Representation.RELATION, "EBITDA driven_by savings")],
        summaries=[_item("sum_1", Representation.SUMMARY, "Summary of section 3.")],
        evidence=EvidenceReport(
            status=EvidenceStatus.COMPLETE,
            unused=[UnusedEvidence(item_id="chk_9", kind="chunk", text="Unused chunk.")],
        ),
        token_budget=100,
        token_estimate=10,
    )


class _CountingCache(MemoryCache):
    """The in-process cache, recording the TTL of every write."""

    def __init__(self) -> None:
        super().__init__()
        self.ttls: dict[str, int | None] = {}

    async def set(self, key: str, value: bytes, *, ttl_seconds: int | None = None) -> None:
        self.ttls[key] = ttl_seconds
        await super().set(key, value, ttl_seconds=ttl_seconds)


# --------------------------------------------------------------------------- record_of


def test_a_record_numbers_handles_and_evidence_in_prompt_order() -> None:
    record = record_of(_bundle())
    assert record.bundle_id == "bnd_1"
    assert record.handles == {
        "m1": "mem_1",
        "m2": "mem_2",
        "f1": "rel_1",
        "s1": "sum_1",
        "d1": "chk_1",
    }
    assert [(e.citation, e.item_id, e.kind) for e in record.evidence] == [
        ("m1", "mem_1", "MEMORY"),
        ("m2", "mem_2", "MEMORY"),
        ("f1", "rel_1", "RELATION"),
        ("s1", "sum_1", "SUMMARY"),
        ("d1", "chk_1", "CHUNK"),
    ]
    assert record.unused == [RecordedEvidence(item_id="chk_9", kind="chunk", text="Unused chunk.")]


def test_a_model_extracted_item_keeps_its_handle_but_carries_no_text() -> None:
    by_id = {e.item_id: e for e in record_of(_bundle()).evidence}
    assert by_id["mem_2"].text == "" and by_id["mem_2"].citation == "m2"
    assert by_id["mem_1"].text == "My timezone is CET."


def test_an_empty_bundle_records_nothing_to_resolve() -> None:
    empty = ContextBundle(
        bundle_id="bnd_0",
        query="q",
        query_type=QueryType.GENERAL_SEMANTIC,
        conversation=ConversationWindow(),
        evidence=EvidenceReport(status=EvidenceStatus.COMPLETE),
        token_budget=10,
        token_estimate=0,
    )
    assert record_of(empty) == BundleRecord(bundle_id="bnd_0")


# --------------------------------------------------------------------------- is_handle


@pytest.mark.parametrize("ref", ["m1", "f2", "s10", "d9999"])
def test_a_short_prefixed_ordinal_is_a_handle(ref: str) -> None:
    assert is_handle(ref)


@pytest.mark.parametrize("ref", ["m0", "m01", "d10000", "x1", "M1", "mem_1", "m1 ", "", "m"])
def test_anything_else_is_a_record_id(ref: str) -> None:
    assert not is_handle(ref)


# --------------------------------------------------------------------------- storing / loading


async def test_a_stored_record_loads_back_for_its_scope_for_thirty_minutes() -> None:
    cache = _CountingCache()
    records = BundleRecords(cache)
    record = record_of(_bundle())
    await records.store(ALICE, record)
    assert await records.load(ALICE, "bnd_1") == record
    assert list(cache.ttls.values()) == [RECORD_TTL_SECONDS]


async def test_another_principal_in_the_tenant_cannot_load_the_record() -> None:
    records = BundleRecords(MemoryCache())
    await records.store(ALICE, record_of(_bundle()))
    assert await records.load(MALLORY, "bnd_1") is None
    assert await records.load(ALICE_RUN, "bnd_1") is None, "agent lineage is part of the scope"


async def test_an_unknown_or_empty_bundle_id_loads_nothing() -> None:
    records = BundleRecords(MemoryCache())
    assert await records.load(ALICE, "bnd_missing") is None
    assert await records.load(ALICE, "") is None


async def test_the_bundle_a_run_was_given_last_is_remembered_per_run() -> None:
    records = BundleRecords(MemoryCache())
    assert await records.latest(ALICE_RUN) is None
    await records.given(ALICE_RUN, "bnd_1")
    await records.given(ALICE_RUN, "bnd_2")
    assert await records.latest(ALICE_RUN) == "bnd_2"
    other_run = ALICE_RUN.model_copy(update={"agent_run_id": "run_2"})
    assert await records.latest(other_run) is None


async def test_a_caller_outside_a_run_has_no_latest_bundle() -> None:
    cache = _CountingCache()
    records = BundleRecords(cache)
    await records.given(ALICE, "bnd_1")
    assert cache.ttls == {}, "nothing is written without a run"
    assert await records.latest(ALICE) is None


async def test_without_a_cache_nothing_is_recorded_and_nothing_resolves() -> None:
    records = BundleRecords(None)
    await records.store(ALICE, record_of(_bundle()))
    await records.given(ALICE_RUN, "bnd_1")
    assert await records.load(ALICE, "bnd_1") is None
    assert await records.latest(ALICE_RUN) is None
    assert await records.resolve(ALICE, "mem_1") == "mem_1"
    with pytest.raises(NotFound):
        await records.resolve(ALICE_RUN, "m1")


async def test_a_cache_outage_degrades_to_nothing_recorded_never_an_error() -> None:
    cache = MemoryCache()
    records = BundleRecords(cache)
    cache.available = False
    await records.store(ALICE, record_of(_bundle()))
    await records.given(ALICE_RUN, "bnd_1")
    assert await records.load(ALICE, "bnd_1") is None
    assert await records.latest(ALICE_RUN) is None
    cache.available = True
    assert await records.load(ALICE, "bnd_1") is None, "the write during the outage was dropped"


async def test_a_record_written_before_an_outage_is_unreadable_during_it() -> None:
    cache = MemoryCache()
    records = BundleRecords(cache)
    await records.store(ALICE_RUN, record_of(_bundle()))
    await records.given(ALICE_RUN, "bnd_1")
    cache.available = False
    assert await records.load(ALICE_RUN, "bnd_1") is None
    assert await records.latest(ALICE_RUN) is None
    cache.available = True
    assert await records.latest(ALICE_RUN) == "bnd_1"


# --------------------------------------------------------------------------- resolve


async def test_a_record_id_resolves_to_itself_without_a_lookup() -> None:
    cache = MemoryCache()
    records = BundleRecords(cache)
    assert await records.resolve(ALICE, "mem_42") == "mem_42"
    assert cache.ops == 0


async def test_a_handle_resolves_through_the_named_bundle() -> None:
    records = BundleRecords(MemoryCache())
    await records.store(ALICE, record_of(_bundle("bnd_1")))
    assert await records.resolve(ALICE, "m1", bundle_id="bnd_1") == "mem_1"
    assert await records.resolve(ALICE, "d1", bundle_id="bnd_1") == "chk_1"


async def test_a_handle_resolves_through_the_runs_latest_bundle_by_default() -> None:
    records = BundleRecords(MemoryCache())
    older = _bundle("bnd_old").model_copy(
        update={"memories": [_item("mem_old", Representation.MEMORY, "Old.")]}
    )
    await records.store(ALICE_RUN, record_of(older))
    await records.store(ALICE_RUN, record_of(_bundle("bnd_new")))
    await records.given(ALICE_RUN, "bnd_old")
    assert await records.resolve(ALICE_RUN, "m1") == "mem_old"
    await records.given(ALICE_RUN, "bnd_new")
    assert await records.resolve(ALICE_RUN, "m1") == "mem_1"
    # an explicit bundle wins over the latest one
    assert await records.resolve(ALICE_RUN, "m1", bundle_id="bnd_old") == "mem_old"


async def test_a_handle_the_bundle_does_not_name_is_not_found() -> None:
    records = BundleRecords(MemoryCache())
    await records.store(ALICE, record_of(_bundle()))
    with pytest.raises(NotFound, match="handle m9 does not resolve"):
        await records.resolve(ALICE, "m9", bundle_id="bnd_1")


async def test_a_handle_of_an_unknown_bundle_is_not_found() -> None:
    records = BundleRecords(MemoryCache())
    with pytest.raises(NotFound, match="handle m1 does not resolve"):
        await records.resolve(ALICE, "m1", bundle_id="bnd_missing")


async def test_a_handle_with_no_bundle_and_no_run_is_not_found() -> None:
    records = BundleRecords(MemoryCache())
    await records.store(ALICE, record_of(_bundle()))
    with pytest.raises(NotFound):
        await records.resolve(ALICE, "m1")


async def test_another_scope_cannot_resolve_a_handle_it_was_not_shown() -> None:
    records = BundleRecords(MemoryCache())
    await records.store(ALICE, record_of(_bundle()))
    with pytest.raises(NotFound):
        await records.resolve(MALLORY, "m1", bundle_id="bnd_1")
