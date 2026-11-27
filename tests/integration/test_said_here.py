"""A rule or a preference says when it was said in the conversation the context is for: a
model asked "what did I tell you in this chat?" can tell a rule given here from one given
weeks ago elsewhere. A line from another conversation is as it always was."""

from __future__ import annotations

import json
import re

import pytest

from memory_service.modules.context.standing import HERE, listed_rules
from tests.integration.test_context_check import QUESTION, TABLE, Judge, _texts
from tests.integration.test_statements_follow_the_user import _chat, _say

pytestmark = pytest.mark.integration

MARKED = re.compile(r"as a table[^\n]*\(said \d{4}-\d{2}-\d{2}, this conversation\)")
PLAIN = re.compile(r"as a table[^\n]*\(said \d{4}-\d{2}-\d{2}\)")


async def _profile(container, ctx, query: str = QUESTION) -> str:
    body = await container.services["context_builder"].build_api(ctx, query, output="full")
    return _texts(json.loads(body), "profile")


async def test_a_rule_said_here_says_so_and_one_said_elsewhere_does_not(
    container, uow_factory
) -> None:
    here, elsewhere = _chat(), _chat()
    await _say(container, uow_factory, here, TABLE)
    assert MARKED.search(await _profile(container, here))
    plain = await _profile(container, elsewhere)
    assert PLAIN.search(plain) and HERE not in plain


async def test_one_conversations_mark_is_never_served_to_another(container, uow_factory) -> None:
    here, elsewhere = _chat(), _chat()
    await _say(container, uow_factory, here, TABLE)
    builder = container.services["context_builder"]
    first = await _profile(container, here)
    await builder.drain()  # the cache write is off the request path
    other = await _profile(container, elsewhere)
    await builder.drain()
    again = await _profile(container, here)
    assert HERE in first and HERE in again and HERE not in other


async def test_a_marked_line_is_one_statement_to_everything_that_reads_it(
    container, uow_factory
) -> None:
    """The context check lists it once and removes it from every place (ADR 0048), and the
    rules a caller reads off the section are the statements, without date or mark."""
    here = _chat()
    await _say(container, uow_factory, here, TABLE)
    builder = container.services["context_builder"]
    bundle = await builder.build(here, QUESTION)
    assert listed_rules(bundle.profile) == [TABLE]

    judge = Judge("table")
    builder.check.assist = judge
    body = json.loads(await builder.build_api(here, QUESTION + " ", output="full", debug=True))
    shown = [item["text"] for item in judge.sent[0]["ITEMS"]]
    assert shown.count(TABLE) == 1 and not any(HERE in text for text in shown)
    assert "table" not in _texts(body, "profile")
