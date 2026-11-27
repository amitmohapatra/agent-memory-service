"""A context for a caller that searches on demand (ADR 0051): what stands is carried, what
retrieval ranks for the request is left to be asked for, the evidence status still says
whether memory holds something for it, and everything left out is one search away."""

from __future__ import annotations

import json

import pytest

from memory_service.modules.context.check import NOTE as CHECK_NOTE
from memory_service.modules.context.demand import NOTE, RANKED
from tests.integration.test_context_check import (
    HEADERS,
    QUESTION,
    SHORT,
    TABLE,
    Judge,
    _read,
    _texts,
    _told,
)
from tests.integration.test_statements_follow_the_user import _chat

pytestmark = pytest.mark.integration

NOTHING_HELD = "What is the boiling point of tungsten in kelvin?"


async def test_what_stands_is_carried_and_what_is_ranked_is_left(container, uow_factory) -> None:
    await _told(container, uow_factory)
    pushed = await _read(container, None)
    assert "Coach tote" in _texts(pushed, "memories")

    asked = await _read(container, None, recall=False)
    # the rule that applies to the request stands; nothing ranked is carried
    assert "table" in _texts(asked, "profile")
    for section in RANKED:
        assert not asked.get(section), section
    assert not asked.get("earlier_chats")
    # and the report says what memory holds for the request, as it does when it is carried
    assert asked["evidence_status"] == pushed["evidence_status"] != "INSUFFICIENT"
    note = asked["diagnostics"][NOTE]
    assert note["carried"] is False and set(note["left"]) <= {*RANKED, "earlier_chats"}
    assert note["left"]["memories"] == len(pushed["memories"]) >= 1
    assert asked["token_estimate"] < pushed["token_estimate"]
    assert asked["bundle_id"] != pushed["bundle_id"], "another context, another cache entry"


async def test_the_prompt_form_has_no_memories_section(container, uow_factory) -> None:
    await _told(container, uow_factory)
    builder = container.services["context_builder"]
    pushed = json.loads(await builder.build_api(_chat(), QUESTION, output="prompt"))
    asked = json.loads(await builder.build_api(_chat(), QUESTION, output="prompt", recall=False))
    assert "Coach tote is short" in pushed["rendered"]
    assert "Coach tote is short" not in asked["rendered"] and "dentist" not in asked["rendered"]
    assert "table" in asked["rendered"]
    assert asked["evidence_status"] == pushed["evidence_status"]


async def test_nothing_held_for_the_request_says_so_either_way(container, uow_factory) -> None:
    await _told(container, uow_factory)
    asked = await _read(container, None, NOTHING_HELD, recall=False)
    pushed = await _read(container, None, NOTHING_HELD)
    assert asked["evidence_status"] == pushed["evidence_status"]


async def test_what_was_left_is_still_what_an_answer_is_checked_against(
    container, uow_factory
) -> None:
    await _told(container, uow_factory)
    builder = container.services["context_builder"]
    bundle = await builder.build(_chat(), QUESTION, recall=False)
    assert bundle.memories == []
    assert any("Coach tote" in u.text for u in bundle.evidence.unused)


async def test_a_context_built_one_way_never_answers_for_the_other(container, uow_factory) -> None:
    await _told(container, uow_factory)
    builder = container.services["context_builder"]
    ctx = _chat()
    first = json.loads(await builder.build_api(ctx, QUESTION, output="full", recall=False))
    await builder.drain()  # the cache write is off the request path
    pushed = json.loads(await builder.build_api(ctx, QUESTION, output="full"))
    assert "memories" not in first and "Coach tote" in _texts(pushed, "memories")
    await builder.drain()
    again = json.loads(await builder.build_api(ctx, QUESTION, output="full", recall=False))
    assert "memories" not in again and _texts(again, "profile") == _texts(first, "profile")


async def test_the_check_reads_only_what_is_carried(container, uow_factory) -> None:
    """Where a model checks a context (ADR 0048), it is shown the standing lines alone:
    nothing ranked is carried, so nothing ranked is asked about."""
    await _told(container, uow_factory)
    judge = Judge("holiday")
    body = await _read(container, judge, recall=False)
    kinds = {item["kind"] for item in judge.sent[0]["ITEMS"]}
    assert "said in an earlier conversation" not in kinds and "a rule the person gave" in kinds
    assert "holiday" not in _texts(body, "profile") and "table" in _texts(body, "profile")
    assert body["diagnostics"][CHECK_NOTE]["listed"] == len(judge.sent[0]["ITEMS"])


async def test_everything_left_out_is_one_search_away(container, uow_factory) -> None:
    await _told(container, uow_factory)
    found = await container.services.search.search(_chat(), QUESTION, kinds=["memory"], limit=10)
    assert any(SHORT in item.text for item in found.items)


def test_the_route_takes_the_choice_and_refuses_what_is_not_one(client) -> None:
    container = client.app.state.container
    said = client.post(
        "/v1/messages",
        headers=HEADERS,
        json={
            "scope": {"thread_id": "thread_told"},
            "messages": [{"role": "USER", "content": text} for text in (TABLE, SHORT)],
        },
    )
    assert said.status_code in (200, 201, 202), said.text
    client.portal.call(container.tasks.drain)
    client.portal.call(container.tasks.drain)
    pushed = client.post("/v1/context", headers=HEADERS, json={"query": QUESTION})
    asked = client.post("/v1/context", headers=HEADERS, json={"query": QUESTION, "recall": False})
    assert pushed.status_code == asked.status_code == 200, asked.text
    assert "Coach tote is short" in pushed.json()["rendered"]
    assert "Coach tote is short" not in asked.json()["rendered"]
    assert asked.json()["evidence_status"] == pushed.json()["evidence_status"]
    refused = client.post("/v1/context", headers=HEADERS, json={"query": QUESTION, "recall": "x"})
    assert refused.status_code == 422
