"""The clean-up of what was stored before a model read a person's messages (ADR 0050): the
stored messages are read again by the model, and what was made from the ones it says are a
request or shared material is forgotten - the way a person's own "forget this" is. The model
is scripted (it answers by a phrase of the message): these prove what a verdict does, not the
verdict."""

from __future__ import annotations

import json
from typing import Any

import pytest

from memory_service.modules.memory.gapfill import GapFill
from memory_service.tools.clean_up import clean_up
from tests.integration.test_memory import _memories
from tests.integration.test_statements_follow_the_user import _chat, _say

pytestmark = pytest.mark.integration

OWN_RULE = "Never create an allocation when ATA is 0."
BRAND = "My default brand is KateSpade and my channel is Retail-Store."
PASTED = (
    "save this skill:\n# Read-Only Mode\nUse this skill when the user wants to look at "
    "allocations without changing them.\nNever create, update, or delete allocations. "
    "Always tell the user which plan you read."
)
FOR_NOW = "ok for now skip the consent stage"
ASK = "Create an allocation for the tote at store 12."


class Reader:
    """The model as the pipeline reaches it: what kind of message it says each one is, by a
    phrase of the message (``None``: it does not answer); a statement otherwise. ``rules``:
    the sentences it reads as the speaker's lasting rule, where a message says one."""

    def __init__(self, *rules: str, **kinds: str | None) -> None:
        self.kinds = {phrase.replace("_", " "): kind for phrase, kind in kinds.items()}
        self.rules = rules
        self.read: list[str] = []
        self.known: dict[str, list[str]] = {}

    def wants(self, use: str) -> bool:
        return True

    async def structured(self, use: str, *, user: str, **_: Any) -> dict[str, Any] | None:
        sent = json.loads(user)
        message = sent["message"]
        self.read.append(message)
        self.known[message] = [k["text"] for k in sent["known"]]
        ops = [_rule(r) for r in self.rules if r in message]
        for phrase, kind in self.kinds.items():
            if phrase in message.casefold():
                return None if kind is None else {"kind": kind, "ops": ops}
        return {"kind": "statement", "ops": ops}


def _rule(sentence: str) -> dict[str, Any]:
    return {
        "op": "add",
        "fact": sentence,
        "quote": sentence,
        "type": "rule",
        "subject": "",
        "date": "",
        "until": "",
        "confidence": 0.9,
        "target": 0,
    }


async def _stored_without_a_model(container, uow_factory, user: str = "u1") -> None:
    chat = _chat(user)
    for said in (OWN_RULE, BRAND, PASTED, FOR_NOW):
        await _say(container, uow_factory, chat, said)


async def _texts(container, uow_factory, user: str = "u1") -> list[str]:
    return [m.content for m in await _memories(uow_factory, _chat(user), container)]


async def _rules(container, user: str = "u1") -> str:
    body = await container.services["context_builder"].build_api(_chat(user), ASK, output="full")
    blocks = json.loads(body).get("profile") or []
    return "\n".join(b["text"] for b in blocks if b["block"] in ("standing_rules", "user"))


async def test_what_was_made_from_a_message_that_is_no_statement_is_forgotten(
    container, uow_factory
) -> None:
    await _stored_without_a_model(container, uow_factory)
    before = await _texts(container, uow_factory)
    # as the patterns left it: the pasted skill's sentences are the person's rules, and their
    # own rule on the same subject is not listed (the later "never ..." superseded it)
    assert "Never create, update, or delete allocations." in before
    assert FOR_NOW in before
    listed = await _rules(container)
    assert "delete allocations" in listed and "ATA is 0" not in listed

    reader = Reader(save_this_skill="material", for_now="request")
    container.services["observation_pipeline"].gapfill = GapFill(reader)  # type: ignore[arg-type]

    # without apply: a report, and nothing changed
    dry = await clean_up(container, tenant_id="acme")
    assert (dry.applied, dry.read, dry.kept, dry.unread, dry.messages) == (False, 4, 2, 0, 2)
    assert dry.forgotten >= 3 and dry.failures == []
    assert {e["category"] for e in dry.examples} >= {"rule", "verbatim_turn"}
    assert await _texts(container, uow_factory) == before

    done = await clean_up(container, tenant_id="acme", apply=True)
    await container.tasks.drain()
    await container.tasks.drain()
    assert (done.applied, done.messages, done.forgotten) == (True, 2, dry.forgotten)
    after = await _texts(container, uow_factory)
    assert not any("delete allocations" in t or "which plan you read" in t for t in after)
    assert FOR_NOW not in after
    # what the person said of themselves is as it was, and their own rule holds again
    assert OWN_RULE in after and BRAND in after
    listed = await _rules(container)
    assert "ATA is 0" in listed and "delete allocations" not in listed

    # nothing left to forget: a second run reads only what is still stored, and keeps it
    again = await clean_up(container, tenant_id="acme", apply=True)
    assert (again.read, again.kept, again.messages, again.forgotten) == (2, 2, 0, 0)


async def test_a_rule_the_model_reads_from_a_request_stays_and_the_request_goes(
    container, uow_factory
) -> None:
    """A request that also gives a rule: the request is no memory, the rule is the person's.
    The model reads the message as it first would - not shown the rule the patterns made of
    it, or it would add nothing (a memory holds it) and the rule would go with the request."""
    rule = "From now on always answer stock questions as a table."
    quoted = "Never create, update, or delete allocations."
    asked = f"Show me the top 20 stores by sales. {rule}"
    chat = _chat()
    for said in (asked, PASTED):
        await _say(container, uow_factory, chat, said)
    before = await _texts(container, uow_factory)
    assert rule in before and asked in before and quoted in before

    # the pasted skill's sentence is a rule to the patterns; the model reads none in it
    reader = Reader(rule, show_me="request", save_this_skill="material")
    container.services["observation_pipeline"].gapfill = GapFill(reader)  # type: ignore[arg-type]
    done = await clean_up(container, tenant_id="acme", apply=True)
    await container.tasks.drain()
    await container.tasks.drain()
    assert rule not in reader.known[asked], "read as the first time: its own memories unseen"
    assert (done.messages, done.confirmed, done.failures) == (2, 1, [])
    after = await _texts(container, uow_factory)
    assert rule in after and asked not in after
    assert quoted not in after and PASTED not in after
    assert "as a table" in await _rules(container)


async def test_a_message_the_model_does_not_answer_for_is_left_as_it_is(
    container, uow_factory
) -> None:
    await _stored_without_a_model(container, uow_factory)
    before = await _texts(container, uow_factory)
    silent = Reader(save_this_skill=None, for_now="request")
    container.services["observation_pipeline"].gapfill = GapFill(silent)  # type: ignore[arg-type]
    report = await clean_up(container, tenant_id="acme", apply=True)
    await container.tasks.drain()
    assert (report.unread, report.messages) == (1, 1)
    after = await _texts(container, uow_factory)
    assert "Never create, update, or delete allocations." in after  # unread: never guessed
    assert FOR_NOW not in after and len(after) < len(before)


async def test_no_model_allowed_reads_nothing_and_forgets_nothing(container, uow_factory) -> None:
    await _stored_without_a_model(container, uow_factory)
    before = await _texts(container, uow_factory)
    report = await clean_up(container, tenant_id="acme", apply=True)
    assert (report.read, report.forgotten, report.unread) == (0, 0, 4)
    assert await _texts(container, uow_factory) == before


async def test_one_users_messages_alone_and_at_most_so_many(container, uow_factory) -> None:
    await _stored_without_a_model(container, uow_factory, "u1")
    await _stored_without_a_model(container, uow_factory, "u2")
    reader = Reader(save_this_skill="material", for_now="request")
    container.services["observation_pipeline"].gapfill = GapFill(reader)  # type: ignore[arg-type]
    theirs = await _texts(container, uow_factory, "u2")
    report = await clean_up(container, tenant_id="acme", user_id="u1", apply=True)
    await container.tasks.drain()
    assert report.messages == 2 and len(reader.read) == 4
    assert await _texts(container, uow_factory, "u2") == theirs
    assert FOR_NOW not in await _texts(container, uow_factory, "u1")
    # at most so many messages, in the order they were said
    limited = await clean_up(container, tenant_id="acme", user_id="u2", limit=1)
    assert limited.read + limited.unread == 1
    assert (await clean_up(container, tenant_id="other")).read == 0
