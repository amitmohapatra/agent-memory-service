"""What a catalog entry is searched by (``ToolDescriptor.index_text``): its name, description and
argument names, then what the input schema says each argument is, nested ones included, each
distinct text once and the whole bounded."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from memory_service.domain.ids import content_hash
from memory_service.domain.tools import (
    INDEX_ARGUMENT_MAX_CHARS,
    INDEX_SCHEMA_DEPTH,
    INDEX_TEXT_MAX_CHARS,
    ToolDescriptor,
    argument_descriptions,
    bounded_text,
)
from memory_service.modules.tools.index import ToolIndex
from memory_service.ports.search import SearchRecord


def _tool(schema: dict[str, Any] | None, description: str = "Create a record") -> ToolDescriptor:
    return ToolDescriptor(
        tenant_id="acme", name="crm-create", description=description, input_schema=schema
    )


def test_an_entry_without_argument_descriptions_reads_as_before() -> None:
    tool = _tool({"type": "object", "properties": {"b": {"type": "string"}, "a": {}}})
    assert tool.index_text() == "crm-create Create a record a b"
    assert _tool(None, description="").index_text() == "crm-create"


def test_the_arguments_descriptions_follow_the_argument_names() -> None:
    tool = _tool(
        {
            "type": "object",
            "properties": {
                "iban": {"type": "string", "description": "Bank account to wire the refund to"},
                "amount": {"type": "number", "description": "Refund amount in cents"},
            },
        }
    )
    assert tool.index_text() == (
        "crm-create Create a record amount iban Refund amount in cents "
        "Bank account to wire the refund to"
    )


def test_nested_objects_array_items_refs_and_alternatives_are_read() -> None:
    schema = {
        "type": "object",
        "properties": {
            "customer": {
                "type": "object",
                "description": "Who is billed",
                "properties": {
                    "address": {
                        "type": "object",
                        "properties": {
                            "postcode": {"type": "string", "description": "Postal code"},
                        },
                    },
                },
            },
            "lines": {
                "type": "array",
                "items": {
                    "type": "object",
                    "description": "One invoice line",
                    "properties": {"sku": {"type": "string", "description": "Stock keeping unit"}},
                },
            },
            "carrier": {"$ref": "#/$defs/Carrier"},
            "note": {"anyOf": [{"type": "string", "description": "Free text"}, {"type": "null"}]},
        },
        "$defs": {
            "Carrier": {
                "type": "object",
                "description": "Parcel service",
                "properties": {"tracking": {"type": "string", "description": "Tracking number"}},
            }
        },
    }
    described = argument_descriptions(schema)
    # level by level, by argument name within a level
    assert described == [
        "Parcel service",
        "Who is billed",
        "One invoice line",
        "Free text",
        "Tracking number",
        "Stock keeping unit",
        "Postal code",  # the customer's address's: a level deeper
    ]
    text = _tool(schema).index_text()
    assert all(d in text for d in described)


def test_descriptions_deeper_than_the_limit_are_left_out() -> None:
    schema: dict[str, Any] = {"type": "object", "properties": {}}
    node = schema
    for level in range(1, INDEX_SCHEMA_DEPTH + 3):
        child = {"type": "object", "description": f"level{level}", "properties": {}}
        node["properties"]["x"] = child
        node = child
    described = argument_descriptions(schema)
    assert described == [f"level{n}" for n in range(1, INDEX_SCHEMA_DEPTH + 1)]
    assert argument_descriptions(schema, depth=1) == ["level1"]


def test_a_self_referencing_schema_ends() -> None:
    schema = {
        "type": "object",
        "properties": {"node": {"$ref": "#/definitions/Node"}},
        "definitions": {
            "Node": {
                "type": "object",
                "description": "A tree node",
                "properties": {"child": {"$ref": "#/definitions/Node"}},
            }
        },
    }
    assert argument_descriptions(schema) == ["A tree node"] * INDEX_SCHEMA_DEPTH
    assert _tool(schema).index_text().count("A tree node") == 1


def test_the_order_does_not_depend_on_how_the_schema_was_stored() -> None:
    a = {"type": "object", "properties": {"x": {"description": "ex"}, "y": {"description": "why"}}}
    b = {"properties": {"y": {"description": "why"}, "x": {"description": "ex"}}, "type": "object"}
    assert _tool(a).index_text() == _tool(b).index_text()


def test_repeated_text_appears_once() -> None:
    tool = _tool(
        {
            "type": "object",
            "properties": {
                "from": {"description": "An  ISO\n date"},
                "to": {"description": "an iso date"},
                "same": {"description": "Create a record"},  # the tool's own description
                "blank": {"description": "   "},
            },
        }
    )
    text = tool.index_text()
    assert text.casefold().count("iso date") == 1
    assert text.count("Create a record") == 1
    assert "  " not in text and "\n" not in text


def test_the_text_is_bounded() -> None:
    many = {
        f"arg{i:03d}": {"type": "string", "description": f"distinct parameter meaning {i} " * 20}
        for i in range(200)
    }
    text = _tool({"type": "object", "properties": many}).index_text()
    assert len(text) <= INDEX_TEXT_MAX_CHARS
    assert text.startswith("crm-create Create a record arg000 arg001")
    assert "distinct parameter meaning 0" in text

    long = "x" * (INDEX_ARGUMENT_MAX_CHARS * 3)
    one = _tool({"type": "object", "properties": {"a": {"description": long}}}).index_text()
    assert one.endswith(" " + "x" * INDEX_ARGUMENT_MAX_CHARS)

    huge = _tool(None, description="word " * INDEX_TEXT_MAX_CHARS).index_text()
    assert len(huge) <= INDEX_TEXT_MAX_CHARS and huge.endswith("word")


def test_a_cut_falls_on_a_space_and_a_script_without_spaces_is_cut_at_the_limit() -> None:
    assert bounded_text(["alpha beta gamma"], limit=12) == "alpha beta"
    assert bounded_text(["請求書の支払期限を延長する"], limit=5) == "請求書の支"
    assert bounded_text(["a", "b", "c"], limit=3) == "a b"
    assert bounded_text(["abc", "ABC", ""], limit=100) == "abc"


def test_multilingual_descriptions_are_kept_as_written() -> None:
    tool = _tool(
        {
            "type": "object",
            "properties": {
                "frist": {"type": "string", "description": "Zahlungsfrist der Rechnung"},
                "期限": {"type": "string", "description": "請求書の支払期限"},
            },
        }
    )
    text = tool.index_text()
    assert "Zahlungsfrist der Rechnung" in text and "請求書の支払期限" in text
    assert "期限" in text.split()


def test_odd_schemas_are_tolerated() -> None:
    odd: dict[str, Any] = {
        "properties": {
            "a": "not a schema",
            "b": {"description": 7},
            "c": {"items": [{"description": "first of a tuple"}, "x"]},
            "d": {"$ref": "https://example.com/elsewhere"},
            "e": {"$ref": "#/properties/a"},
        }
    }
    assert argument_descriptions(odd) == ["first of a tuple"]
    assert argument_descriptions(None) == []
    assert "first of a tuple" in _tool(odd).index_text()


# --- whether the index holds an entry as it is now ------------------------------------------


class _Store:
    def __init__(self, records: list[SearchRecord]) -> None:
        self.records = {r.record_id: r for r in records}

    async def ensure_collection(self, spec) -> None:
        return None

    async def get(self, collection, ids):
        return [self.records[i] for i in ids if i in self.records]


class _Indexer:
    spaces = SimpleNamespace(dimensions={})
    sparse = SimpleNamespace(server_side_idf=True)

    def collection(self, name: str) -> str:
        return name


def _record(entry: ToolDescriptor, text: str) -> SearchRecord:
    return SearchRecord(
        record_id=entry.tool_id,
        collection="tools",
        tenant_id="acme",
        payload={"text_hash": content_hash(text)},
    )


def _current(entry: ToolDescriptor) -> SearchRecord:
    """The record the current release writes for ``entry``."""
    from memory_service.modules.tools.index import record_hash

    return SearchRecord(
        record_id=entry.tool_id,
        collection="tools",
        tenant_id="acme",
        payload={"text_hash": record_hash(entry.index_text()), "tool_name": entry.name},
    )


async def test_an_entry_is_indexed_once_its_record_holds_its_current_text() -> None:
    """B13: the first run asked for hints before its just-published tools were indexed, and
    every hint was a zero. ``GET /v1/tools`` says per entry whether search finds it now."""
    fresh = ToolDescriptor(tenant_id="acme", tool_id="tol_a", name="store_demand")
    changed = ToolDescriptor(tenant_id="acme", tool_id="tol_b", name="excess_stock")
    absent = ToolDescriptor(tenant_id="acme", tool_id="tol_c", name="lost_sales")
    store = _Store([_current(fresh), _record(changed, "an older text")])
    index = ToolIndex(None, _Indexer(), store)  # type: ignore[arg-type]
    assert await index.indexed([fresh, changed, absent]) == {"tol_a"}
    assert await index.indexed([]) == set()


# --- the record layout a search filters on ---------------------------------------------------


class _SearchStore(_Store):
    """A store that records the filter a search sends and the payloads patched in place."""

    def __init__(self, records: list[SearchRecord]) -> None:
        super().__init__(records)
        self.filters: list[Any] = []
        self.patched: dict[str, dict[str, Any]] = {}

    async def search_hybrid(self, collection, *, flt, **kwargs):
        self.filters.append(flt)
        return []

    async def set_payloads(self, collection, payloads) -> None:
        for record_id, fields in payloads.items():
            self.patched[record_id] = dict(fields)
            self.records[record_id].payload.update(fields)


class _NoEncoderIndexer(_Indexer):
    """No encoder at all: a test fails if anything is embedded."""

    spaces = SimpleNamespace(dimensions={}, primary=SimpleNamespace(tool_cosine=0.0))

    async def embed_cached(self, texts, hashes):  # pragma: no cover - the failure itself
        raise AssertionError("a record of the current text is not embedded again")


def _vectors():
    from memory_service.domain.script import Script
    from memory_service.modules.retrieval.engine import QueryVectors
    from memory_service.ports.search import SparseVector, VectorName

    return QueryVectors(
        dense={VectorName.DENSE_ML: [0.1] * 4},
        sparse=SparseVector(indices=[1], values=[1.0]),
        script=Script.LATIN,
    )


def _admits(flt: Any, payload: dict[str, Any]) -> bool:
    """Whether a store applying ``flt`` keeps a record of ``payload`` (Qdrant's reading:
    every ``must`` and ``must_any`` entry, and one of the ``should_any`` ones)."""
    if payload.get("tenant_id", flt.tenant_id) != flt.tenant_id:
        return False
    must = all(payload.get(k) == v for k, v in flt.must.items())

    def any_of(key: str, values: list[str]) -> bool:
        held = payload.get(key)
        return any(h in values for h in held) if isinstance(held, list) else held in values

    must_any = all(any_of(k, v) for k, v in flt.must_any.items())
    should = not flt.should_any or any(any_of(k, v) for k, v in flt.should_any.items())
    return must and must_any and should


async def test_the_available_tools_are_filtered_on_an_indexed_keyword_field() -> None:
    """``available`` narrows the search in the store, on ``tool_name``, a keyword payload
    index, besides the tenant and the audience, which stay required."""
    from memory_service.adapters.search.qdrant_store import _PAYLOAD_INDEXES

    store = _SearchStore([])
    index = ToolIndex(None, _NoEncoderIndexer(), store)  # type: ignore[arg-type]
    await index.search("acme", "ws1", "t", vectors=_vectors(), available=["b", "a", "b"])
    (flt,) = store.filters
    assert flt.should_any["tool_name"] == ["a", "b"], "each name once"
    assert "tool_name" not in flt.must_any and "text" not in flt.must_any
    assert {*flt.must, *flt.must_any} <= set(_PAYLOAD_INDEXES)
    assert _PAYLOAD_INDEXES["tool_name"] == "keyword"
    assert flt.must == {"kind": "tool"}
    assert flt.must_any["visibility_keys"] == ["tenant:acme", "workspace:acme/ws1"]

    assert await index.search("acme", "ws1", "t", vectors=_vectors(), available=[]) == []
    assert len(store.filters) == 1, "an empty set searches nothing"


async def test_a_record_an_older_release_wrote_is_found_by_a_filtered_search() -> None:
    """While releases are mixed, a worker of the earlier release writes layout-1 records
    (no ``tool_name``) back over the moved ones on each refresh tick, and a record is not
    moved until this release's next refresh or ``GET /v1/tools``. A search filtered on
    ``tool_name`` alone found none of them and a run's hints came back empty. The search
    matches the name in either field, ``tool_name`` or layout 1's ``text``, still inside the
    tenant, the kind and the audience."""
    store = _SearchStore([])
    index = ToolIndex(None, _NoEncoderIndexer(), store)  # type: ignore[arg-type]
    await index.search("acme", "ws1", "t", vectors=_vectors(), available=["crm-create"])
    (flt,) = store.filters
    audience = {"kind": "tool", "tenant_id": "acme", "visibility_keys": ["tenant:acme"]}
    v1 = {**audience, "text": "crm-create", "text_hash": "abc"}
    v2 = {**v1, "tool_name": "crm-create", "text_hash": "v2:abc"}
    assert _admits(flt, v2)
    assert _admits(flt, v1), "a layout-1 record is still found"
    assert not _admits(flt, {**v1, "text": "other"}), "another tool is not"
    assert not _admits(flt, {**v2, "tenant_id": "other"}), "nor another tenant's"
    assert not _admits(flt, {**v2, "visibility_keys": ["workspace:acme/ws2"]}), "nor out of scope"


async def test_a_record_carries_its_tool_name_and_its_layouts_hash() -> None:
    from memory_service.modules.tools.index import RECORD_FORMAT, record_hash

    class _Embedding(_Indexer):
        sparse = SimpleNamespace(
            server_side_idf=True,
            encode_documents=lambda texts: [None for _ in texts],
        )

        async def embed_cached(self, texts, hashes):
            return {}

    entry = ToolDescriptor(tenant_id="acme", tool_id="tol_a", name="crm-create")
    index = ToolIndex(None, _Embedding(), _Store([]))  # type: ignore[arg-type]
    (record,) = await index._records([entry])
    assert record.payload["tool_name"] == "crm-create"
    assert record.payload["text_hash"] == record_hash(entry.index_text())
    assert record.payload["text_hash"] != content_hash(entry.index_text())
    assert str(RECORD_FORMAT) in record.payload["text_hash"]


async def test_a_record_an_older_layout_wrote_is_rewritten_in_place_when_asked_about() -> None:
    """A record of the entry's current text that an earlier release wrote (no ``tool_name``,
    a bare ``text_hash``) is not found by a filtered search. ``indexed`` - what a harness
    waits on before its first hints - moves it to the current layout there and then, payload
    only (its vectors are of the same text), so the wait does not run out; a record of other
    text is still not indexed until its job or the refresh embeds it."""
    old = ToolDescriptor(tenant_id="acme", tool_id="tol_a", name="store_demand")
    changed = ToolDescriptor(tenant_id="acme", tool_id="tol_b", name="excess_stock")
    store = _SearchStore([_record(old, old.index_text()), _record(changed, "an older text")])
    index = ToolIndex(None, _NoEncoderIndexer(), store)  # type: ignore[arg-type]
    assert await index.indexed([old, changed]) == {"tol_a"}
    assert set(store.patched) == {"tol_a"}
    assert store.patched["tol_a"]["tool_name"] == "store_demand"
    assert await index.indexed([old, changed]) == {"tol_a"}, "and current from then on"


async def test_a_failed_rewrite_reads_as_not_indexed_yet() -> None:
    class _Failing(_SearchStore):
        async def set_payloads(self, collection, payloads) -> None:
            raise RuntimeError("store away")

    old = ToolDescriptor(tenant_id="acme", tool_id="tol_a", name="store_demand")
    index = ToolIndex(None, _NoEncoderIndexer(), _Failing([_record(old, old.index_text())]))  # type: ignore[arg-type]
    assert await index.indexed([old]) == set()


class _Catalog:
    """A unit of work whose catalog pages ``entries`` by ``tool_id``."""

    def __init__(self, entries: list[ToolDescriptor]) -> None:
        self.entries = sorted(entries, key=lambda e: e.tool_id)
        self.tools = self

    async def __aenter__(self) -> _Catalog:
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None

    async def catalog_page(self, *, tenant_id, after, limit):
        return [e for e in self.entries if e.tool_id > after][:limit]


async def test_a_failed_rewrite_of_one_page_does_not_stop_the_refresh(monkeypatch) -> None:
    """``refresh`` moves each page's old-layout records in place. One failed write (the
    store briefly away) aborted the whole sweep: the pages after it, and the same page's
    records that needed embedding, waited for the next tick. Like ``indexed``, the sweep now
    logs the failure and goes on, and the refresh is not trusted as done, so the failed page's
    records are moved on the next tick."""
    from memory_service.modules.tools import index as index_module

    monkeypatch.setattr(index_module, "REFRESH_BATCH", 2)
    first = ToolDescriptor(tenant_id="acme", tool_id="tol_a", name="store_demand")
    changed = ToolDescriptor(tenant_id="acme", tool_id="tol_b", name="excess_stock")
    later = ToolDescriptor(tenant_id="acme", tool_id="tol_c", name="lost_sales")

    class _FlakyStore(_SearchStore):
        failures = 1
        upserted: list[str] = []

        async def set_payloads(self, collection, payloads) -> None:
            if self.failures:
                self.failures -= 1
                raise RuntimeError("store away")
            await super().set_payloads(collection, payloads)

        async def upsert(self, records) -> None:
            self.upserted += [r.record_id for r in records]
            self.records.update({r.record_id: r for r in records})

    class _Embedding(_Indexer):
        sparse = SimpleNamespace(
            server_side_idf=True, encode_documents=lambda texts: [None for _ in texts]
        )

        async def embed_cached(self, texts, hashes):
            return {}

    store = _FlakyStore(
        [
            _record(first, first.index_text()),
            _record(changed, "an older text"),
            _record(later, later.index_text()),
        ]
    )
    catalog = _Catalog([first, changed, later])
    index = ToolIndex(lambda: catalog, _Embedding(), store)  # type: ignore[arg-type]
    assert await index.refresh() == 2, "tol_b embedded, tol_c moved"
    assert store.upserted == ["tol_b"]
    assert set(store.patched) == {"tol_c"}
    # not trusted as done: the next tick compares again instead of waiting six hours
    assert await index.refresh() == 1, "the failed page's record, a tick later"
    assert set(store.patched) == {"tol_a", "tol_c"}
    assert await index.refresh() == 0, "and done, trusted from then on"
