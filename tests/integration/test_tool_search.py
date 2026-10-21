"""Tool search reads what an input schema says the arguments are, and a tool record embedded
from an earlier build's text is re-indexed without anyone touching the catalog."""

from __future__ import annotations

from typing import Any

import pytest

from memory_service.domain.context import MemoryExecutionContext
from memory_service.domain.ids import content_hash
from memory_service.domain.tools import ToolDescriptor

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
    return await container.services["tool_index"].search("acme", "ws1", TASK)


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
    # an entry a recorded call created, never indexed: left to its own job
    async with uow_factory() as uow:
        bare = await uow.tools.ensure("acme", "chargeback-bare")
        await uow.commit()
    ids = [e.tool_id for e in entries]

    async def stored() -> dict[str, Any]:
        return {
            r.record_id: r.payload.get("text_hash")
            for r in await container.search.get(index.collection, [*ids, bare.tool_id])
        }

    assert set((await stored()).values()) == {None}

    await container.tasks.run_periodic("periodic.tools_refresh")

    assert await stored() == {e.tool_id: content_hash(e.index_text()) for e in entries}
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
