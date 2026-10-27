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


async def test_an_entry_is_indexed_once_its_record_holds_its_current_text() -> None:
    """B13: the first run asked for hints before its just-published tools were indexed, and
    every hint was a zero. ``GET /v1/tools`` says per entry whether search finds it now."""
    fresh = ToolDescriptor(tenant_id="acme", tool_id="tol_a", name="store_demand")
    changed = ToolDescriptor(tenant_id="acme", tool_id="tol_b", name="excess_stock")
    absent = ToolDescriptor(tenant_id="acme", tool_id="tol_c", name="lost_sales")
    store = _Store([_record(fresh, fresh.index_text()), _record(changed, "an older text")])
    index = ToolIndex(None, _Indexer(), store)  # type: ignore[arg-type]
    assert await index.indexed([fresh, changed, absent]) == {"tol_a"}
    assert await index.indexed([]) == set()
