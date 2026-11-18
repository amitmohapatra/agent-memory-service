"""Tool search reads what an input schema says the arguments are, and a tool record embedded
from an earlier build's text is re-indexed without anyone touching the catalog."""

from __future__ import annotations

from typing import Any

import pytest

from memory_service.domain.context import MemoryExecutionContext
from memory_service.domain.ids import content_hash
from memory_service.domain.tools import ToolDescriptor
from memory_service.modules.tools.index import record_hash
from memory_service.ports.search import SearchFilter

pytestmark = pytest.mark.integration

#: A word that only the refund tool's nested parameter description carries.
TASK = "chargeback"


def _ctx() -> MemoryExecutionContext:
    return MemoryExecutionContext(tenant_id="acme", user_id="u1", workspace_id="ws1")


def _catalog() -> list[ToolDescriptor]:
    refund = ToolDescriptor(
        tenant_id="",
        name="billing-adjust",
        description="Adjust a billing record.",
        input_schema={
            "type": "object",
            "properties": {
                "record": {"type": "string", "description": "The billing record id"},
                "reason": {
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": "Why: a chargeback the card issuer opened",
                        }
                    },
                },
            },
        },
    )
    other = ToolDescriptor(
        tenant_id="",
        name="billing-export",
        description="Export billing records as a spreadsheet.",
        input_schema={
            "type": "object",
            "properties": {"record": {"type": "string", "description": "The billing record id"}},
        },
    )
    return [refund, other]


async def _publish(container, uow_factory) -> None:
    async with uow_factory() as uow:
        await container.services["tool_memory"].put_catalog(
            uow, _ctx(), [(e, frozenset(ToolDescriptor.model_fields)) for e in _catalog()]
        )
        await uow.commit()
    await container.tasks.drain()  # tools.index


async def _search(container) -> list[tuple[str, float]]:
    """``(name, relevance)`` of each tool found: the stand-in encoder has no calibrated
    ``tool_cosine``, so every hit matches."""
    found = await container.services["tool_index"].search("acme", "ws1", TASK)
    assert all(m.matches for m in found)
    return [(m.name, m.relevance) for m in found]


def _old_index_text(self: ToolDescriptor) -> str:
    """What an earlier build embedded: name, description and argument names."""
    fields = sorted(((self.input_schema or {}).get("properties") or {}).keys())
    return " ".join(filter(None, (self.name, self.description, " ".join(fields))))


async def test_a_tool_is_found_by_its_parameter_description(container, uow_factory) -> None:
    refund = _catalog()[0]
    assert TASK not in _old_index_text(refund).casefold(), "only the parameter says it"
    assert TASK in refund.index_text()
    await _publish(container, uow_factory)
    found = await _search(container)
    assert found[0] == ("billing-adjust", 1.0)
    assert dict(found)["billing-export"] < 1.0


async def test_records_an_earlier_build_wrote_are_reindexed_on_the_periodic_refresh(
    container, uow_factory, monkeypatch
) -> None:
    index = container.services["tool_index"]
    # what an existing deployment holds: records embedded from the earlier text, no text_hash
    with monkeypatch.context() as patched:
        patched.setattr(ToolDescriptor, "index_text", _old_index_text)
        await _publish(container, uow_factory)
        async with uow_factory() as uow:
            entries = await uow.tools.catalog("acme", workspace_id="ws1", limit=10)
        records = await index._records(entries)
        for record in records:
            record.payload.pop("text_hash")
        await container.search.upsert(records)
    # an entry a recorded call created: never indexed by a job of its own
    async with uow_factory() as uow:
        bare = await uow.tools.ensure("acme", "crm-bare-entry")
        await uow.commit()
    ids = [e.tool_id for e in entries]

    async def stored() -> dict[str, Any]:
        return {
            r.record_id: r.payload.get("text_hash")
            for r in await container.search.get(index.collection, [*ids, bare.tool_id])
        }

    assert set((await stored()).values()) == {None}

    await container.tasks.run_periodic("periodic.tools_refresh")

    assert await stored() == {e.tool_id: record_hash(e.index_text()) for e in [*entries, bare]}, (
        "the stale records re-embedded, the missing one indexed"
    )
    assert (await _search(container))[0] == ("billing-adjust", 1.0)
    # once per process: the next tick reads nothing; forced, nothing is stale any more
    assert await index.refresh() == 0
    assert await index.refresh(force=True) == 0


async def test_a_refresh_after_a_text_change_reindexes_only_what_changed(
    container, uow_factory, monkeypatch
) -> None:
    await _publish(container, uow_factory)
    index = container.services["tool_index"]
    assert await index.refresh(force=True) == 0, "records the current build wrote are current"

    original = ToolDescriptor.index_text

    def next_build(self: ToolDescriptor) -> str:
        text = original(self)
        return text + " refunds" if self.name == "billing-adjust" else text

    monkeypatch.setattr(ToolDescriptor, "index_text", next_build)
    assert await index.refresh(force=True) == 1
    assert await index.refresh(force=True) == 0


async def test_a_tool_first_seen_in_a_recorded_call_becomes_searchable(
    container, uow_factory
) -> None:
    """``record`` creates a bare catalog entry for a tool nobody published; the refresh
    indexes it like any entry without a record, so tool search can offer it by name."""
    index = container.services["tool_index"]
    ctx = MemoryExecutionContext(tenant_id="acme", user_id="u1", workspace_id="ws1")
    async with uow_factory() as uow:
        await container.services["tool_memory"].record(
            uow, ctx, tool="ledger-chargeback_open", args={"case": "C-1"}
        )
        await uow.commit()
    await container.tasks.drain()
    assert await _search(container) == []

    assert await index.refresh(force=True) == 1
    assert [name for name, _ in await _search(container)] == ["ledger-chargeback_open"]
    assert await index.refresh(force=True) == 0


async def test_the_reindex_tool_rebuilds_tool_search_after_a_model_change(
    container, uow_factory
) -> None:
    """A model change is a new, empty collection: ``make reindex`` fills it from the
    catalog, every tenant's or one tenant's only."""
    from memory_service.tools.reindex import rebuild_search_index

    await _publish(container, uow_factory)
    globex = MemoryExecutionContext(tenant_id="globex", user_id="u9")
    async with uow_factory() as uow:
        await container.services["tool_memory"].put_catalog(
            uow,
            globex,
            [(ToolDescriptor(tenant_id="", name="globex-tool"), frozenset({"name"}))],
        )
        await uow.commit()
    await container.tasks.drain()
    index = container.services["tool_index"]

    async def counts() -> dict[str, int]:
        return {
            t: len(await container.search.record_ids(index.collection, SearchFilter(tenant_id=t)))
            for t in ("acme", "globex")
        }

    assert await counts() == {"acme": 2, "globex": 1}

    await container.search.drop_collection(index.collection)  # the new model's: empty
    index.forget_collections()
    assert await _search(container) == []

    acme = await rebuild_search_index(container, tenant_id="acme")
    assert acme.ok and acme.tools == 2
    assert await counts() == {"acme": 2, "globex": 0}, "never another tenant's"
    assert (await _search(container))[0] == ("billing-adjust", 1.0)

    everyone = await rebuild_search_index(container)
    assert everyone.ok and everyone.tools == 3
    assert await counts() == {"acme": 2, "globex": 1}


# --- the run's available tools, filtered on an indexed keyword field -------------------------


async def _available(container, available, workspace_id: str | None = "ws1") -> list[str]:
    found = await container.services["tool_index"].search(
        "acme", workspace_id, TASK, available=available
    )
    return [m.name for m in found]


async def _as_an_earlier_release_wrote_them(container, uow_factory) -> list[ToolDescriptor]:
    """Rewrite the catalog's records as the release before ``tool_name`` wrote them: the same
    vectors, a bare ``text_hash`` and no keyword field to filter on."""
    index = container.services["tool_index"]
    async with uow_factory() as uow:
        entries = await uow.tools.catalog("acme", workspace_id="ws1", limit=10)
    records = await index._records(entries)
    for record, entry in zip(records, entries, strict=True):
        record.payload.pop("tool_name", None)
        record.payload["text_hash"] = content_hash(entry.index_text())
    await container.search.upsert(records)
    return entries


async def test_records_an_earlier_layout_wrote_are_moved_to_the_new_one_without_embedding(
    container, uow_factory, monkeypatch
) -> None:
    """The window after a deploy: the old records have no ``tool_name``, so a search that
    names the run's tools finds none of them. The periodic refresh moves them to the new
    layout - payload only, the vectors are of the same text - and they are found again."""
    await _publish(container, uow_factory)
    await _as_an_earlier_release_wrote_them(container, uow_factory)
    index = container.services["tool_index"]
    assert await _available(container, ["billing-adjust"]) == [], "not yet: no tool_name"

    async def no_embedding(texts, hashes):
        raise AssertionError("an unchanged text is not embedded again")

    monkeypatch.setattr(index.indexer, "embed_cached", no_embedding)
    await container.tasks.run_periodic("periodic.tools_refresh")
    assert await _available(container, ["billing-adjust"]) == ["billing-adjust"]
    assert await index.refresh(force=True) == 0


async def test_asking_whether_old_records_are_indexed_moves_them_there_and_then(
    container, uow_factory
) -> None:
    """A harness waits (seconds) for ``GET /v1/tools`` to say its tools are indexed before its
    first hints; the refresh runs on the worker's five-minute tick. So the question itself
    moves a current-text record of the old layout, and the wait ends at once."""
    await _publish(container, uow_factory)
    entries = await _as_an_earlier_release_wrote_them(container, uow_factory)
    index = container.services["tool_index"]
    assert await index.indexed(entries) == {e.tool_id for e in entries}
    assert await _available(container, ["billing-adjust", "billing-export"]) == [
        "billing-adjust",
        "billing-export",
    ]


async def test_the_available_set_narrows_the_search_whatever_its_size_or_repeats(
    container, uow_factory
) -> None:
    await _publish(container, uow_factory)
    assert await _available(container, []) == []
    assert await _available(container, ["nobody-has-this"]) == []
    assert await _available(container, ["billing-export", "billing-export"]) == ["billing-export"]
    many = [f"tool-{n:04d}" for n in range(2000)]
    assert await _available(container, [*many, "billing-export"]) == ["billing-export"]
    assert await _available(container, None) == ["billing-adjust", "billing-export"]


async def test_a_workspaces_own_entry_and_the_tenants_of_one_name_are_one_tool(
    container, uow_factory
) -> None:
    """A workspace's entry shadows the tenant's of the same name: both records match the name,
    the search answers it once. Another workspace sees the tenant's alone, and never a name
    only a workspace not its own publishes."""
    tenant_wide = MemoryExecutionContext(tenant_id="acme", user_id="u1")
    async with uow_factory() as uow:
        service = container.services["tool_memory"]
        await service.put_catalog(
            uow,
            tenant_wide,
            [(e, frozenset(ToolDescriptor.model_fields)) for e in _catalog()[:1]],
        )
        await uow.commit()
    await _publish(container, uow_factory)  # ws1's own billing-adjust and billing-export
    names = ["billing-adjust", "billing-export"]
    assert await _available(container, names) == names
    assert await _available(container, names, workspace_id="ws2") == ["billing-adjust"]
    assert await _available(container, names, workspace_id=None) == ["billing-adjust"]
