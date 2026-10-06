"""Every audience the service offers, exercised through the HTTP API.

Six visibilities, and the point of this file is that there are exactly six and all of them
work. Four were withdrawn because they could not: GROUP was never writable (the groups a
request asserts were persisted nowhere, so the job that builds the memory had no anchor),
WORK had no grant path anywhere so only its author ever read it, WORKSPACE needed a
membership grant nothing in the service issued, and GLOBAL was TENANT wearing a name that
implies crossing tenants - which nothing can.

Each case asserts who sees it AND who does not. A visibility test that only checks the
positive half is how an audience ends up wider than anyone intended.
"""

from __future__ import annotations

import pytest

from memory_service.domain.ids import new_id
from tests.e2e.conftest import post_message

pytestmark = pytest.mark.e2e

KEY = {"X-API-Key": "test-key"}
TENANT = "search-team"


def _h(user: str | None = None) -> dict[str, str]:
    h = {**KEY, "X-Trellis-Tenant": TENANT}
    if user:
        h["X-Trellis-User"] = user
    return h


def _scope(**extra: str) -> dict[str, str]:
    return {
        "thread_id": new_id("thread"),
        "session_id": new_id("session"),
        "turn_id": new_id("turn"),
        **extra,
    }


def _write(client, headers, content, visibility, scope=None) -> bool:
    """Store ``content`` as a memory for exactly this audience."""
    body = {"scope": scope or _scope(), "content": content, "visibility": visibility}
    r = client.post("/v1/memories", headers=headers, json=body)
    return r.status_code in (200, 201)


def _read(client, headers, query, scope=None) -> set[str]:
    r = client.post(
        "/v1/recall",
        headers=headers,
        json={"scope": scope or _scope(), "query": query, "kinds": ["memory"]},
    )
    assert r.status_code == 200, r.text
    return {i["text"] for i in r.json()["items"]}


def _has(texts: set[str], needle: str) -> bool:
    return any(needle in t for t in texts)


# ---------------------------------------------------------------- USER
def test_user_memory_follows_the_person_across_chats_and_agents(client) -> None:
    """The product's core value: an agent acting for someone knows what they know."""
    assert _write(client, _h("alice"), "My timezone is Europe/Berlin.", "USER")
    q = "what is my timezone?"

    assert _has(_read(client, _h("alice"), q), "Europe/Berlin"), "alice, a different chat"
    assert _has(_read(client, _h("alice"), q, _scope(agent_id="research")), "Europe/Berlin"), (
        "her agent inherits it without any grant"
    )
    assert not _has(_read(client, _h("bob"), q), "Europe/Berlin"), "another person does not"


# -------------------------------------------------------------- THREAD
def test_thread_memory_stays_in_its_own_conversation(client) -> None:
    """The default audience, and the leak an integrator reported: a new chat used to inherit
    the previous one's turns, because the row carried its author's key and the author matched
    it from anywhere."""
    chat = _scope()
    assert _write(client, _h("alice"), "We chose ClickHouse for this task.", "THREAD", chat)
    q = "what did we choose?"

    assert _has(_read(client, _h("alice"), q, chat), "ClickHouse"), "in its own chat"
    assert not _has(_read(client, _h("alice"), q), "ClickHouse"), "not in another chat of hers"
    assert not _has(_read(client, _h("bob"), q, chat), "ClickHouse"), "and not for someone else"


# -------------------------------------------------------------- TENANT
def test_tenant_memory_reaches_the_whole_team_and_no_further(client) -> None:
    assert _write(client, _h("alice"), "The on-call rota moves to PagerDuty.", "TENANT")
    q = "where is the on-call rota?"

    assert _has(_read(client, _h("alice"), q), "PagerDuty")
    assert _has(_read(client, _h("bob"), q), "PagerDuty"), "everyone on the team"
    assert _has(_read(client, _h("bob"), q, _scope(agent_id="writer")), "PagerDuty"), (
        "and their agents"
    )

    other = {**KEY, "X-Trellis-Tenant": "data-team", "X-Trellis-User": "alice"}
    assert not _has(_read(client, other, q), "PagerDuty"), "another tenant, same user id"


# ------------------------------------------------------------- PRIVATE
def test_private_is_one_principal_and_not_even_its_user(client) -> None:
    run = _scope(agent_id="research", agent_run_id=new_id("agent_run"))
    assert _write(client, _h("alice"), "Scratch: retry the flaky query.", "PRIVATE", run)
    q = "what should be retried?"

    assert _has(_read(client, _h("alice"), q, _scope(agent_id="research")), "flaky query"), (
        "the same agent, including on a later run - PRIVATE is its durable store"
    )
    assert not _has(_read(client, _h("alice"), q), "flaky query"), "not alice herself"
    assert not _has(_read(client, _h("alice"), q, _scope(agent_id="writer")), "flaky query"), (
        "not her other agent"
    )
    assert not _has(_read(client, _h("bob"), q, _scope(agent_id="research")), "flaky query"), (
        "and not the same agent name under another user - the principal is bound"
    )


def test_private_is_the_durable_store_of_a_job_that_has_no_user(client) -> None:
    """A scheduled job has no user, so USER is not available to it. PRIVATE is, and it
    persists across executions - which is what makes stateful jobs possible at all."""
    job = _scope(agent_id="nightly", agent_run_id=new_id("agent_run"))
    assert _write(client, _h(), "Last sync processed 412 rows.", "PRIVATE", job)
    q = "what did the last sync process?"

    later = _scope(agent_id="nightly", agent_run_id=new_id("agent_run"))
    assert _has(_read(client, _h(), q, later), "412 rows"), "the same job, a later run"
    assert not _has(
        _read(client, _h(), q, _scope(agent_id="hourly", agent_run_id=new_id("agent_run"))),
        "412 rows",
    ), "a different job"


# --------------------------------------------------------- AGENT_GROUP
def test_agent_group_lets_peers_co_work_at_any_depth(client) -> None:
    """The channel for parallel agents, and it is not bounded by the run tree: peers share by
    holding the same group id, however deep they sit and whoever they run for."""
    crew = "crew-alpha"
    scope = _scope(agent_id="peer_a", agent_run_id=new_id("agent_run"), agent_group_id=crew)
    assert _write(client, _h("alice"), "Upstream deps are mapped.", "AGENT_GROUP", scope)
    q = "are the upstream deps mapped?"

    peer = _scope(agent_id="peer_b", agent_run_id=new_id("agent_run"), agent_group_id=crew)
    assert _has(_read(client, _h("alice"), q, peer), "Upstream deps"), "a peer in the crew"
    assert _has(_read(client, _h("bob"), q, peer), "Upstream deps"), (
        "even acting for another user - the crew is the audience, not the person"
    )

    off_crew = _scope(agent_id="peer_c", agent_run_id=new_id("agent_run"), agent_group_id="other")
    assert not _has(_read(client, _h("alice"), q, off_crew), "Upstream deps"), "another crew"
    assert not _has(_read(client, _h("alice"), q), "Upstream deps"), "and nobody outside one"


# ----------------------------------------------------------------- RUN
def test_run_hands_off_down_reports_up_and_never_sideways(client) -> None:
    """Hand-off inside one execution. Directional on purpose: ``run:<mine>`` flows down to the
    runs I spawn, ``runup:<parent>`` reports back to the run that spawned me, and a sibling
    carrying its parent's key still matches neither.

    RUN used to carry the author's principal key instead, which made it an IDENTITY audience
    wearing a run's name: five parallel workers on one agent_id read each other's scratch, a
    retry inherited the failed attempt's reasoning, and the supervisor that spawned them saw
    none of it. Hand-off ran backwards.
    """
    parent = new_id("agent_run")
    sup = _scope(agent_id="super", agent_run_id=parent)
    assert _write(client, _h("alice"), "Plan: split revenue from cost.", "RUN", sup)

    child = _scope(agent_id="worker", agent_run_id=new_id("agent_run"), parent_agent_run_id=parent)
    assert _write(client, _h("alice"), "Finding: schema drift in Q3.", "RUN", child)

    plan, finding = "split revenue", "schema drift"
    down = _read(client, _h("alice"), "plan or finding", child)
    assert _has(down, plan), "a child reads what its parent handed down"

    up = _read(client, _h("alice"), "plan or finding", sup)
    assert _has(up, finding), "and the supervisor reads what its child reported"

    sibling = _scope(
        agent_id="worker2", agent_run_id=new_id("agent_run"), parent_agent_run_id=parent
    )
    seen = _read(client, _h("alice"), "plan or finding", sibling)
    assert _has(seen, plan), "a sibling still receives the parent's hand-off"
    assert not _has(seen, finding), "but never its sibling's notes"

    later = _scope(agent_id="worker", agent_run_id=new_id("agent_run"))
    assert not _has(_read(client, _h("alice"), "plan or finding", later), finding), (
        "and a later run of the same agent inherits nothing - that is what USER is for"
    )

    assert not _has(_read(client, _h("alice"), "plan or finding"), finding), "the user sees none"


# ---------------------------------------------------- past conversations
def test_message_search_across_threads_reads_only_the_users_own_conversations(client) -> None:
    """``threads=all``: what the user said in any earlier chat, by message, with its thread;
    another person's chats are never searched, and the default is still this thread."""
    alice_old, bob_old = _scope(), _scope()
    said = (
        (_h("alice"), alice_old, "My locker code is 3141."),
        (_h("bob"), bob_old, "My locker code is 2718."),
    )
    for headers, scope, text in said:
        r = post_message(client, headers, {"scope": scope, "role": "USER", "content": text})
        assert r.status_code in (200, 201, 202), r.text

    def recall(headers: dict[str, str], **extra: str) -> list[dict[str, str]]:
        body = {"scope": _scope(), "query": "locker code", "kinds": ["message"], **extra}
        r = client.post("/v1/recall", headers=headers, json=body)
        assert r.status_code == 200, r.text
        return r.json()["items"]

    assert recall(_h("alice")) == [], "a new chat's own messages by default"
    mine = recall(_h("alice"), threads="all")
    assert [(i["text"], i["thread_id"]) for i in mine] == [
        ("USER: My locker code is 3141.", alice_old["thread_id"])
    ], "alice's earlier chat, and not bob's"
    assert [i["text"] for i in recall(_h("bob"), threads="all")] == [
        "USER: My locker code is 2718."
    ]
    assert recall(_h(), threads="all") == [], "no user: no one's conversations"
    r = client.post(
        "/v1/recall",
        headers=_h("alice"),
        json={"scope": _scope(), "query": "x", "kinds": ["message"], "threads": "everyone"},
    )
    assert r.status_code == 422


# ------------------------------------------------- the four that are gone
@pytest.mark.parametrize("withdrawn", ["GROUP", "WORK", "WORKSPACE", "GLOBAL"])
def test_a_withdrawn_audience_is_refused_at_the_door(client, withdrawn: str) -> None:
    """Refused by the schema, not accepted and quietly dropped. Three of these used to be
    accepted with a 202 and then produce a memory nobody could read."""
    r = client.post(
        "/v1/memories",
        headers=_h("alice"),
        json={"scope": _scope(), "content": "x", "visibility": withdrawn},
    )
    assert r.status_code == 422, f"{withdrawn} must not be accepted: {r.text}"
