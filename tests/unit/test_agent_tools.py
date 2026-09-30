"""The memory agent tools without a service: the fixed set, their schemas, pull bookkeeping."""

from __future__ import annotations

from datetime import UTC, datetime

from memory_service.modules.agent_tools.service import TOOLS, AgentTools, _within, result_ids

FINAL_SET = {
    "memory_search",
    "memory_remember",
    "memory_update",
    "memory_forget",
    "history_search",
    "profile_edit",
    "procedures_search",
    "tool_search",
    "record_outcome",
}


def test_the_set_is_final_and_every_schema_is_self_contained() -> None:
    specs = AgentTools.specs()
    assert {s["name"] for s in specs} == FINAL_SET == {t.name for t in TOOLS}
    for spec in specs:
        schema = spec["input_schema"]
        assert schema["type"] == "object" and schema.get("additionalProperties") is False
        assert "$defs" not in schema and "$ref" not in str(schema), spec["name"]
        assert spec["description"] and spec["description"][0].isupper()
    search = next(s for s in specs if s["name"] == "memory_search")["input_schema"]
    assert search["required"] == ["query"] and search["properties"]["k"]["maximum"] == 20


def test_only_named_items_are_counted_as_what_a_pull_returned() -> None:
    assert result_ids([{"id": "mem_1"}, {"text": "no id"}, {"id": "chk_2"}]) == ["mem_1", "chk_2"]
    assert result_ids({"id": "mem_3"}) == ["mem_3"]
    assert result_ids({"candidates": []}) == []


def test_a_time_window_keeps_only_what_was_observed_inside_it() -> None:
    start, end = datetime(2026, 1, 1, tzinfo=UTC), datetime(2026, 2, 1, tzinfo=UTC)
    assert _within(None, None, None)
    assert _within("2026-01-15T00:00:00+00:00", start, end)
    assert not _within("2026-03-01T00:00:00+00:00", start, end)
    assert not _within(None, start, None), "undated evidence cannot be placed in a window"
