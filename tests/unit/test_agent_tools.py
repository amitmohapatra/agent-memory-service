"""The memory agent tools without a service: the fixed set, their schemas, pull bookkeeping."""

from __future__ import annotations

from datetime import UTC, datetime

from memory_service.modules.agent_tools.service import TOOLS, AgentTools, result_ids
from memory_service.modules.retrieval.engine import Candidate, _observed_within
from memory_service.modules.retrieval.search import candidate_item

FINAL_SET = {
    "memory_search",
    "memory_remember",
    "memory_update",
    "memory_forget",
    "profile_edit",
    "tool_search",
}


def test_the_set_is_final_and_every_schema_is_self_contained() -> None:
    specs = AgentTools.specs()
    assert {s["name"] for s in specs} == FINAL_SET == {t.name for t in TOOLS}
    for spec in specs:
        schema = spec["input_schema"]
        assert schema["type"] == "object" and schema.get("additionalProperties") is False
        assert "$defs" not in schema and "$ref" not in str(schema), spec["name"]
        assert spec["description"] and spec["description"][0].isupper()
    by_name = {s["name"]: s["input_schema"] for s in specs}
    search = by_name["memory_search"]
    assert search["required"] == ["query"] and search["properties"]["k"]["maximum"] == 20
    assert "message" in str(search["properties"]["kinds"]), "the history is a kind of search"
    assert by_name["memory_update"]["required"] == ["id", "content"]
    assert by_name["memory_forget"]["required"] == ["id"]
    assert by_name["profile_edit"]["required"] == ["block", "new"]
    assert set(by_name["tool_search"]["properties"]) == {"task"}, (
        "the toolbox is the caller's, not an argument the model fills"
    )
    # what the model reads every call: no title repeating a name, no null branch or default
    for spec in specs:
        for name, prop in spec["input_schema"]["properties"].items():
            assert "title" not in prop and prop.get("default", 0) is not None, (spec["name"], name)
            assert {"type": "null"} not in prop.get("anyOf", []), (spec["name"], name)
    assert search["properties"]["time_from"] == {
        "description": "observed since",
        "format": "date-time",
        "type": "string",
    }


def test_only_named_items_are_counted_as_what_a_pull_returned() -> None:
    assert result_ids([{"id": "mem_1"}, {"text": "no id"}, {"id": "chk_2"}]) == ["mem_1", "chk_2"]
    assert result_ids({"id": "mem_3"}) == ["mem_3"]
    assert result_ids({"candidates": []}) == []


def test_a_time_window_keeps_only_what_was_observed_inside_it() -> None:
    start, end = datetime(2026, 1, 1, tzinfo=UTC), datetime(2026, 2, 1, tzinfo=UTC)

    def at(observed: str | None) -> Candidate:
        payload = {"observed_at": observed} if observed else {}
        return Candidate(record_id="mem_1", kind="memory", text="t", score=1.0, payload=payload)

    assert _observed_within(at("2026-01-15T00:00:00+00:00"), (start, end))
    assert not _observed_within(at("2026-03-01T00:00:00+00:00"), (start, end))
    assert _observed_within(at("2026-03-01T00:00:00+00:00"), (start, None))
    assert not _observed_within(at(None), (start, None)), "undated evidence has no place in it"


def test_an_item_is_what_a_caller_uses_and_cites() -> None:
    c = Candidate(
        record_id="chk_1",
        kind="chunk",
        text="x" * 1200,
        score=0.3,
        payload={"document_id": "doc_1", "page": 4, "observed_at": "2026-01-15T10:00:00+00:00"},
    )
    item = candidate_item(c, debug=False, text_chars=1000)
    assert item.model_dump(exclude_none=True) == {
        "id": "chk_1",
        "kind": "chunk",
        "text": "x" * 1000 + "…",
        "observed_on": "2026-01-15",
        "document_id": "doc_1",
        "page": 4,
    }
    assert candidate_item(c, debug=True).debug["score"] == 0.3  # type: ignore[index]
