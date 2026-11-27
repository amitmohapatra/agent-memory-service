"""The context check through a real build (ADR 0048): what a person said is stored and
retrieved as always, a model is asked once what of it helps with the request, and what it
names does not reach the context. The model is scripted (it drops by a word in the item):
these prove the build around the call - what is sent, cached and reported - not the model's
judgement, which needs a paid run to measure."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from memory_service.modules.context.check import CHECKED, LATE, NOTE, USE
from tests.integration.test_statements_follow_the_user import _chat, _say
from tests.support_llm import mocked_gateway

pytestmark = pytest.mark.integration

TABLE = "Always answer stock questions as a table with columns store, item, count."
HOLIDAY = "Never schedule a delivery to the stock room on a public holiday."
LAPTOPS = "Our whole team runs Linux laptops now."
BOOKS = "Our stock accounts are kept in GBP."
SHORT = "The Coach tote is short of stock at stores 12 and 40."
DENTIST = "My dentist moved the stock take reminder to Tuesday."
QUESTION = "Which stores are short of stock of the Coach tote?"


class Judge:
    """The model as the check reaches it: it names every item whose text has one of
    ``words``, and records what it was sent. ``hangs``: it never answers."""

    def __init__(self, *words: str, hangs: bool = False, allowed: bool = True) -> None:
        self.words, self.hangs, self.allowed = words, hangs, allowed
        self.sent: list[dict[str, Any]] = []
        self.available = False  # no beliefs, no pages: the builder's other reads are as without

    def wants(self, use: str) -> bool:
        return self.allowed and use == USE

    async def structured(self, use: str, *, user: str, **_: Any) -> dict[str, Any] | None:
        body = json.loads(user)
        self.sent.append(body)
        if self.hangs:
            await asyncio.sleep(60)
        return {
            "drop": [
                item["n"]
                for item in body["ITEMS"]
                if any(word.casefold() in item["text"].casefold() for word in self.words)
            ]
        }


async def _told(container, uow_factory) -> None:
    """What the person said in an earlier conversation: two rules, two things about their
    work, the answer to the question and something else."""
    earlier = _chat()
    for said in (TABLE, HOLIDAY, LAPTOPS, BOOKS, SHORT, DENTIST):
        await _say(container, uow_factory, earlier, said)


async def _read(container, judge: Judge | None, query: str = QUESTION, **more: Any) -> dict:
    builder = container.services["context_builder"]
    if judge is not None:
        builder.check.assist = judge
    body = await builder.build_api(_chat(), query, output="full", debug=True, **more)
    return json.loads(body)


def _texts(body: dict, section: str) -> str:
    if section == "profile":
        return "\n".join(b["text"] for b in body.get("profile") or [])
    return "\n".join(str(i.get("text", "")) for i in body.get(section) or [])


async def test_what_the_model_names_does_not_reach_the_context(container, uow_factory) -> None:
    await _told(container, uow_factory)
    plain = await _read(container, Judge(allowed=False))
    # as built: both rules share the question's word "stock" (and the user block lists
    # them again), the nearest things said of the person's work come whatever is asked,
    # and the other memory is near enough
    assert "holiday" in _texts(plain, "profile") and "Linux" in _texts(plain, "profile")
    assert "dentist" in _texts(plain, "memories")
    assert NOTE not in plain["diagnostics"]

    judge = Judge("holiday", "Linux", "GBP", "dentist")
    checked = await _read(container, judge, QUESTION + " ")  # another request: no cache hit
    assert len(judge.sent) == 1, "one call for the whole context"
    profile, memories = _texts(checked, "profile"), _texts(checked, "memories")
    assert "holiday" not in profile and "Linux" not in profile and "GBP" not in profile
    assert "dentist" not in memories
    # what it did not name is as built
    assert "table" in profile and "Coach tote" in memories
    note = checked["diagnostics"][NOTE]
    assert note["outcome"] == CHECKED
    assert note["listed"] == len(judge.sent[0]["ITEMS"])
    assert {d["section"] for d in note["dropped"]} >= {"standing_rules", "memories"}
    assert checked["token_estimate"] < plain["token_estimate"]


async def test_the_model_is_sent_the_request_and_the_items_not_the_whole_context(
    container, uow_factory
) -> None:
    await _told(container, uow_factory)
    judge = Judge()
    await _read(container, judge)
    sent = judge.sent[0]
    assert sent["REQUEST"] == QUESTION and len(sent["TODAY"]) == 10
    kinds = {item["kind"] for item in sent["ITEMS"]}
    assert "a rule the person gave" in kinds and "said in an earlier conversation" in kinds
    assert [item["n"] for item in sent["ITEMS"]] == list(range(1, len(sent["ITEMS"]) + 1))


async def test_the_model_may_remove_all_it_was_shown_and_the_context_says_so(
    container, uow_factory
) -> None:
    """Nothing memory attached is kept for its rank: a request none of it helps with gets
    none of it, and the evidence status then says memory holds nothing for it."""
    await _told(container, uow_factory)
    everything = Judge("a", "e", "i", "o", "u")
    body = await _read(container, everything)
    assert "memories" not in body and _texts(body, "profile") == ""
    assert body["evidence_status"] == "INSUFFICIENT"
    note = body["diagnostics"][NOTE]
    assert note["listed"] == len(everything.sent[0]["ITEMS"]) == len(note["dropped"])


async def test_a_pasted_skill_goes_whole_with_the_rule_the_person_gave_left_in_place(
    container, uow_factory
) -> None:
    """Stored before a model read messages: a skill's text pasted into a chat is the
    person's standing rule, its sentences are lines of their ``user`` block, and it ranks
    first for the request it blocks. Removed once, as the turn, it is gone from all three."""
    earlier = _chat()
    pasted = (
        "save this skill:\n# Read-Only Mode\nUse this skill when the user wants to look at "
        "allocations without changing them.\nNever create, update, or delete allocations. "
        "Always tell the user which plan you read."
    )
    for said in ("Never create an allocation when ATA is 0.", pasted):
        await _say(container, uow_factory, earlier, said)
    await container.services["profile"].refresh_user("acme", "u1")
    ask = "Create an allocation for the tote at store 12."
    plain = await _read(container, Judge(allowed=False), ask)
    assert "Read-Only Mode" in _texts(plain, "profile")
    assert "Never create, update, or delete allocations." in _texts(plain, "profile")

    judge = Judge("read-only mode")  # names the pasted turn, and only it
    checked = await _read(container, judge, ask + " ")
    everywhere = _texts(checked, "profile") + _texts(checked, "memories")
    assert "Read-Only" not in everywhere and "delete allocations" not in everywhere
    assert "which plan you read" not in everywhere
    assert "ATA is 0" in everywhere
    shown = [item["text"] for item in judge.sent[0]["ITEMS"]]
    assert "Never create, update, or delete allocations." not in shown, "asked once, as the turn"


async def test_the_checked_context_is_what_the_cache_answers_with(container, uow_factory) -> None:
    await _told(container, uow_factory)
    builder = container.services["context_builder"]
    judge = Judge("dentist")
    builder.check.assist = judge
    ctx = _chat()
    first = json.loads(await builder.build_api(ctx, QUESTION, output="full"))
    await builder.drain()  # the cache write is off the request path
    again = json.loads(await builder.build_api(ctx, QUESTION, output="full"))
    assert len(judge.sent) == 1, "a cached context is not checked a second time"
    assert "dentist" not in _texts(first, "memories")
    assert _texts(again, "memories") == _texts(first, "memories")
    # and the prompt form, rendered from the same checked bundle
    prompt = json.loads(await builder.build_api(ctx, QUESTION, output="prompt"))
    assert "dentist" not in prompt["rendered"] and "Coach tote" in prompt["rendered"]
    assert len(judge.sent) == 1


async def test_a_model_that_is_late_leaves_the_context_as_built(container, uow_factory) -> None:
    await _told(container, uow_factory)
    plain = await _read(container, Judge(allowed=False))
    builder = container.services["context_builder"]
    builder.check.deadline_seconds = 0.3
    late = await _read(container, Judge("dentist", hangs=True), QUESTION + " ")
    assert late["diagnostics"][NOTE]["outcome"] == LATE
    assert _texts(late, "memories") == _texts(plain, "memories")
    assert _texts(late, "profile") == _texts(plain, "profile")


async def test_a_dropped_memory_is_still_what_an_answer_is_checked_against(
    container, uow_factory
) -> None:
    """A memory the check removed was found and is not carried: it joins the evidence the
    bundle did not use, which ``/v1/verify`` reads for contradictions."""
    await _told(container, uow_factory)
    builder = container.services["context_builder"]
    builder.check.assist = Judge("dentist")
    bundle = await builder.build(_chat(), QUESTION)
    assert not any("dentist" in m.text for m in bundle.memories)
    assert any("dentist" in u.text for u in bundle.evidence.unused)


# --------------------------------------------------------------------------- the policy


HEADERS = {"X-API-Key": "test-key", "X-Trellis-Tenant": "acme", "X-Trellis-User": "reader"}


def _policy(client, uses: list[str], *, read_assist: bool) -> None:
    container = client.app.state.container

    async def put() -> None:
        async with container.services["uow_factory"]() as uow:
            await container.services["model_policies"].set(
                uow, "acme", uses=uses, read_assist=read_assist, models={}
            )
            await uow.commit()

    client.portal.call(put)


def test_the_check_follows_the_tenants_model_policy(client) -> None:
    """On by default; a tenant takes it out by leaving ``context_check`` out of its policy's
    uses, or by turning its reads' model calls off (``read_assist``)."""
    container = client.app.state.container
    response = client.post(
        "/v1/messages",
        headers=HEADERS,
        json={
            "scope": {"thread_id": "thread_told"},
            "messages": [{"role": "USER", "content": said} for said in (TABLE, SHORT, DENTIST)],
        },
    )
    assert response.status_code in (200, 201, 202), response.text
    client.portal.call(container.tasks.drain)
    client.portal.call(container.tasks.drain)
    with mocked_gateway([{"drop": []}]) as gw:
        configured = gw.assist([USE])
        assist = container.services["llm_assist"]
        assist.provider, assist.settings = configured.provider, configured.settings
        assist.policies = container.services["model_policies"]
        try:
            asked = 0
            for uses, read_assist, calls in (
                (["summaries"], True, 0),
                ([USE], False, 0),
                ([USE, "summaries"], True, 1),
            ):
                _policy(client, uses, read_assist=read_assist)
                response = client.post(
                    "/v1/context",
                    headers=HEADERS,
                    json={"query": f"{QUESTION} ({uses}, {read_assist})"},
                )
                assert response.status_code == 200, response.text
                assert gw.route.call_count - asked == calls, (uses, read_assist)
                asked = gw.route.call_count
            sent = gw.prompts()[-1]
            assert "ITEMS" in sent["messages"][1]["content"]
        finally:
            client.portal.call(configured.provider.close)
