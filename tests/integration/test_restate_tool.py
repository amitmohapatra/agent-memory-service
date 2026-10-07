"""Restating stored turns after the fact (``tools.restate``, ADR 0027 decision 6).

Turns stored before a tenant could use ``memory_restatement`` keep the key they were indexed
with. The tool restates each CURRENT verbatim turn through the gateway (mocked here, never
paid) exactly as ingest would, stores the restatement and its relations, and re-indexes the
turns that changed - so a question in words only the restatement uses now finds the turn.
"""

from __future__ import annotations

import runpy
import sys
import warnings
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from benchmark.common import submit_observation
from sqlalchemy import select

from memory_service.adapters.db.orm import MemoryRow
from memory_service.application.container import build_container
from memory_service.domain.context import MemoryExecutionContext
from memory_service.modules.jobs.registry import register_handlers
from memory_service.modules.memory.restatement import USE
from memory_service.tools import restate as restate_tool
from memory_service.tools.restate import RestateReport, restate_turns
from tests.integration.conftest import integration_overrides, integration_settings
from tests.support_llm import mocked_gateway

pytestmark = pytest.mark.integration

ASK = MemoryExecutionContext(tenant_id="acme", user_id="caroline", workspace_id="ws1")
SAY = MemoryExecutionContext(tenant_id="acme", user_id="melanie", workspace_id="ws1")
ELSEWHERE = MemoryExecutionContext(tenant_id="globex", user_id="melanie", workspace_id="ws1")
WHEN = datetime(2023, 5, 8, 13, 56, tzinfo=UTC)
EARLIER = "We went hiking at Mount Tam on Saturday."
TURN = "I adopted a puppy named Biscuit last weekend."
RESTATED = {
    "restatement": "Melanie adopted a dog called Biscuit on 2023-05-06.",
    "facts": ["Melanie has a dog called Biscuit."],
    "relations": [
        {"subject": "Melanie", "predicate": "adopted", "object": "a puppy named Biscuit"}
    ],
}
SILENT = {"restatement": "", "facts": [], "relations": []}


async def _say(container, uow_factory, ctx, content, *, at=WHEN) -> None:
    register_handlers(container)
    async with uow_factory() as uow:
        await submit_observation(uow, ctx, content=content, occurred_at=at)
        await uow.commit()
    await container.tasks.drain()
    await container.tasks.drain()


async def _turns(container, tenant_id: str = "acme") -> dict[str, MemoryRow]:
    """The tenant's verbatim turns, by content."""
    async with container.database.session_factory() as session:
        rows = (
            await session.scalars(
                select(MemoryRow).where(
                    MemoryRow.tenant_id == tenant_id,
                    MemoryRow.system_metadata["category"].astext == "verbatim_turn",
                )
            )
        ).all()
    return {r.content: r for r in rows}


def _use_gateway(container, gateway, uses=(USE,)) -> None:
    """Point the service's model at the mocked gateway, allowed ``uses`` only."""
    container.services["llm_assist"] = gateway.assist(list(uses))


async def _lexical_hit(container, ctx, query: str, memory_id: str) -> bool:
    """Whether BM25 over the memory's own key (not the dense stand-in) finds it."""
    result = await container.services["retrieval"].retrieve(ctx, query, kinds=("memory",))
    return any(c.record_id == memory_id and "bm25" in c.retrievers for c in result.candidates)


async def test_a_stored_turn_is_restated_stored_and_reindexed(container, uow_factory) -> None:
    await _say(container, uow_factory, ASK, EARLIER)
    await _say(container, uow_factory, SAY, TURN, at=WHEN + timedelta(minutes=1))
    before = await _turns(container)
    assert all("restatement" not in r.system_metadata for r in before.values())
    turn_id = before[TURN].memory_id
    assert not await _lexical_hit(container, SAY, "dog", turn_id), "no word of the turn says dog"

    with mocked_gateway([SILENT, RESTATED]) as gw:
        _use_gateway(container, gw)
        report = await restate_turns(container, tenant_id="acme")
        prompts = gw.prompts()

    assert report.as_dict() == {"restated": 1, "skipped": 1, "failures": []}
    # oldest first, and each turn shown with the turn before it and its speaker
    assert len(prompts) == 2
    assert EARLIER in prompts[0]["messages"][-1]["content"]
    assert TURN in prompts[1]["messages"][-1]["content"]
    assert "melanie" in prompts[1]["messages"][-1]["content"]
    assert EARLIER in prompts[1]["messages"][-1]["content"]

    after = await _turns(container)
    meta = after[TURN].system_metadata
    assert meta["restatement"] == (
        "Melanie adopted a dog called Biscuit on 2023-05-06. Melanie has a dog called Biscuit."
    )
    assert meta["restatement_relations"] == [["Melanie", "adopted", "a puppy named Biscuit"]]
    assert after[TURN].content == TURN, "the turn itself is kept verbatim"
    reindexed, first_indexed = after[TURN].indexed_at, before[TURN].indexed_at
    assert reindexed is not None and first_indexed is not None
    assert reindexed > first_indexed, "re-indexed with the new key"
    passed_on = after[EARLIER]
    assert "restatement" not in passed_on.system_metadata, "a turn the model passed on is left"
    assert await _lexical_hit(container, SAY, "dog", turn_id)


async def test_a_restated_turn_is_skipped_on_the_next_run_unless_forced(
    container, uow_factory
) -> None:
    await _say(container, uow_factory, SAY, TURN)
    with mocked_gateway([RESTATED]) as gw:
        _use_gateway(container, gw)
        assert (await restate_turns(container, tenant_id="acme")).restated == 1
        again = await restate_turns(container, tenant_id="acme")
        assert again.as_dict() == {"restated": 0, "skipped": 0, "failures": []}
        assert gw.route.call_count == 1, "nothing left to restate, nothing asked"

    relations_only = {**SILENT, "relations": RESTATED["relations"]}
    with mocked_gateway([relations_only]) as gw:
        _use_gateway(container, gw)
        forced = await restate_turns(container, tenant_id="acme", force=True)
    assert forced.restated == 1
    meta = (await _turns(container))[TURN].system_metadata
    assert "restatement" not in meta, "a forced run replaces the old restatement"
    assert meta["restatement_relations"] == [["Melanie", "adopted", "a puppy named Biscuit"]]


async def test_a_restatement_without_relations_drops_the_old_ones(container, uow_factory) -> None:
    await _say(container, uow_factory, SAY, TURN)
    with mocked_gateway([RESTATED]) as gw:
        _use_gateway(container, gw)
        await restate_turns(container, tenant_id="acme")
    text_only = {**RESTATED, "relations": []}
    with mocked_gateway([text_only]) as gw:
        _use_gateway(container, gw)
        await restate_turns(container, tenant_id="acme", force=True)
    meta = (await _turns(container))[TURN].system_metadata
    assert meta["restatement"].startswith("Melanie adopted a dog")
    assert "restatement_relations" not in meta


async def test_limit_restates_only_the_oldest_turns(container, uow_factory) -> None:
    await _say(container, uow_factory, SAY, "We visited the aquarium in Monterey.")
    await _say(container, uow_factory, SAY, TURN, at=WHEN + timedelta(hours=1))
    with mocked_gateway([RESTATED]) as gw:
        _use_gateway(container, gw)
        report = await restate_turns(container, tenant_id="acme", limit=1)
        assert gw.route.call_count == 1
    assert report.restated + report.skipped == 1
    assert "Monterey" in gw.prompts()[0]["messages"][-1]["content"]


async def test_a_turn_the_model_said_nothing_about_is_left_as_it_was(
    container, uow_factory
) -> None:
    await _say(container, uow_factory, SAY, TURN)
    before = (await _turns(container))[TURN]
    with mocked_gateway([SILENT]) as gw:
        _use_gateway(container, gw)
        report = await restate_turns(container, tenant_id="acme")
        assert gw.route.call_count == 1
    assert report.as_dict() == {"restated": 0, "skipped": 1, "failures": []}
    after = (await _turns(container))[TURN]
    assert after.system_metadata == before.system_metadata
    assert after.indexed_at == before.indexed_at, "nothing changed, nothing re-indexed"


async def test_without_the_use_in_policy_no_model_is_asked_and_nothing_changes(
    container, uow_factory
) -> None:
    await _say(container, uow_factory, SAY, TURN)
    with mocked_gateway([RESTATED]) as gw:
        _use_gateway(container, gw, uses=["query_expansion"])
        report = await restate_turns(container, tenant_id="acme")
        assert gw.route.call_count == 0
    assert report.as_dict() == {"restated": 0, "skipped": 1, "failures": []}
    assert "restatement" not in (await _turns(container))[TURN].system_metadata


async def test_only_the_named_tenant_s_turns_are_restated(container, uow_factory) -> None:
    await _say(container, uow_factory, SAY, TURN)
    await _say(container, uow_factory, ELSEWHERE, TURN)
    with mocked_gateway([RESTATED]) as gw:
        _use_gateway(container, gw)
        report = await restate_turns(container, tenant_id="acme")
    assert report.restated == 1
    assert "restatement" in (await _turns(container, "acme"))[TURN].system_metadata
    assert "restatement" not in (await _turns(container, "globex"))[TURN].system_metadata


async def test_a_turn_that_fails_is_reported_and_the_rest_carry_on(
    container, uow_factory, monkeypatch
) -> None:
    await _say(container, uow_factory, SAY, "We visited the aquarium in Monterey.")
    await _say(container, uow_factory, SAY, TURN, at=WHEN + timedelta(hours=1))
    failing = (await _turns(container))["We visited the aquarium in Monterey."].memory_id
    real = restate_tool.restate

    async def flaky(assist: Any, *, text: str, **kwargs: Any) -> Any:
        if "Monterey" in text:
            raise RuntimeError("gateway exploded")
        return await real(assist, text=text, **kwargs)

    monkeypatch.setattr(restate_tool, "restate", flaky)
    with mocked_gateway([RESTATED]) as gw:
        _use_gateway(container, gw)
        report = await restate_turns(container, tenant_id="acme")
    assert report.restated == 1 and report.skipped == 0
    assert report.failures == [f"memory acme/{failing}: gateway exploded"]
    turns = await _turns(container)
    assert "restatement" not in turns["We visited the aquarium in Monterey."].system_metadata
    assert "restatement" in turns[TURN].system_metadata


async def test_a_memory_of_another_tenant_or_none_at_all_is_never_restated(
    container, uow_factory
) -> None:
    await _say(container, uow_factory, ELSEWHERE, TURN)
    theirs = (await _turns(container, "globex"))[TURN].memory_id
    with mocked_gateway([RESTATED]) as gw:
        _use_gateway(container, gw)
        assert not await restate_tool._restate_one(container, "acme", theirs)
        assert not await restate_tool._restate_one(container, "acme", "mem_missing")
        assert gw.route.call_count == 0
    assert "restatement" not in (await _turns(container, "globex"))[TURN].system_metadata


async def test_a_tenant_with_nothing_stored_restates_nothing(container) -> None:
    report = await restate_turns(container, tenant_id="nobody")
    assert report == RestateReport()


# --------------------------------------------------------------------------- the command


def _command(monkeypatch, make_settings, tmp_path, *argv: str) -> list[Any]:
    """Run the module's ``main`` against the suite's database, recording the container."""
    built: list[Any] = []
    settings = integration_settings(make_settings, blob={"filesystem_root": str(tmp_path)})

    async def build(_settings: Any, version: str) -> Any:
        container = await build_container(
            settings, version, overrides=integration_overrides(blob=None)
        )
        built.append(container)
        return container

    monkeypatch.setattr(restate_tool, "build_container", build)
    monkeypatch.setattr(sys, "argv", ["restate", *argv])
    return built


def test_the_command_prints_its_report_and_exits_zero(
    monkeypatch, make_settings, tmp_path, capsys
) -> None:
    built = _command(monkeypatch, make_settings, tmp_path, "--tenant", "nobody-cli", "--limit", "5")
    assert restate_tool.main() == 0
    printed = capsys.readouterr().out.splitlines()
    assert printed[-1] == "{'restated': 0, 'skipped': 0, 'failures': []}"
    assert len(built) == 1 and built[0].settings.service.environment == "test"


def test_the_command_exits_one_when_a_turn_failed(
    monkeypatch, make_settings, tmp_path, capsys
) -> None:
    _command(monkeypatch, make_settings, tmp_path, "--tenant", "acme", "--force")
    seen: dict[str, Any] = {}

    async def failed(container: Any, **kwargs: Any) -> RestateReport:
        seen.update(kwargs)
        return RestateReport(restated=2, failures=["memory acme/mem_1: boom"])

    monkeypatch.setattr(restate_tool, "restate_turns", failed)
    assert restate_tool.main() == 1
    assert seen == {"tenant_id": "acme", "limit": None, "force": True}
    assert "'failures': ['memory acme/mem_1: boom']" in capsys.readouterr().out


def test_the_command_requires_a_tenant(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["restate"])
    with pytest.raises(SystemExit) as exit_:
        restate_tool.main()
    assert exit_.value.code == 2


def test_the_command_closes_its_container_even_when_the_run_raises(
    monkeypatch, make_settings, tmp_path
) -> None:
    built = _command(monkeypatch, make_settings, tmp_path, "--tenant", "acme")
    closed: list[bool] = []

    async def explode(container: Any, **kwargs: Any) -> RestateReport:
        real_close = container.close

        async def close() -> None:
            closed.append(True)
            await real_close()

        container.close = close
        raise RuntimeError("database gone")

    monkeypatch.setattr(restate_tool, "restate_turns", explode)
    with pytest.raises(RuntimeError, match="database gone"):
        restate_tool.main()
    assert closed == [True] and len(built) == 1


def test_the_module_runs_as_a_command(monkeypatch, capsys) -> None:
    monkeypatch.setattr(sys, "argv", ["restate", "--help"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # runpy re-executing an imported module
        with pytest.raises(SystemExit) as exit_:
            runpy.run_module("memory_service.tools.restate", run_name="__main__")
    assert exit_.value.code == 0
    assert "Restate stored conversation turns" in capsys.readouterr().out
