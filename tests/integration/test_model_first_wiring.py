# ruff: noqa: RUF001 - literal multilingual fixtures.
"""Model-first extraction on the write path, behind its tenant switch (ADR 0045).

``Tenant.model_first_extraction`` is off by default and for a tenant with no row. A value the
user replaces is never left current, with the model and without it."""

from __future__ import annotations

import secrets
from datetime import UTC, datetime, timedelta

import pytest
from benchmark.common import submit_observation

from memory_service.config.constants import MEMORY_INTELLIGENCE
from memory_service.domain.context import MemoryExecutionContext
from memory_service.domain.tenancy import Tenant
from memory_service.modules.llm.assist import LLMAssist
from memory_service.modules.memory.native import NativeMemoryIntelligence
from memory_service.modules.memory.pipeline import ObservationPipeline
from tests.support_llm import mocked_gateway

pytestmark = pytest.mark.integration

START = datetime(2026, 5, 4, 9, 0, tzinfo=UTC)


class Spy(NativeMemoryIntelligence):
    """The native provider, recording the switch each call was given."""

    seen: list[tuple[str, bool]]

    async def extract(self, observation, ctx, *, model_first=False):  # type: ignore[override]
        self.seen.append(("extract", model_first))
        return await super().extract(observation, ctx, model_first=model_first)

    async def consolidate(self, candidate, existing, ctx, *, model_first=False):  # type: ignore[override]
        self.seen.append(("consolidate", model_first))
        return await super().consolidate(candidate, existing, ctx, model_first=model_first)


async def _tenant(uow_factory, switch: bool | None) -> MemoryExecutionContext:
    tenant_id = f"mf-{secrets.token_hex(4)}"
    if switch is not None:
        async with uow_factory() as uow:
            await uow.tenants.add(
                Tenant(tenant_id=tenant_id, name="Model first", model_first_extraction=switch)
            )
            await uow.commit()
    return MemoryExecutionContext(tenant_id=tenant_id, user_id="u1", thread_id="t1")


async def _say(uow_factory, pipeline, ctx, texts: list[str]) -> None:
    for i, text in enumerate(texts):
        async with uow_factory() as uow:
            observed = await submit_observation(
                uow, ctx, content=text, occurred_at=START + timedelta(days=i)
            )
            await uow.commit()
        await pipeline.run({"tenant_id": ctx.tenant_id, "observation_id": observed.observation_id})


async def _current(uow_factory, ctx) -> list:
    async with uow_factory() as uow:
        return await uow.memories.list_recent(
            since=datetime(2000, 1, 1, tzinfo=UTC), tenant_id=ctx.tenant_id
        )


def _pipeline(uow_factory, provider, assist: LLMAssist) -> ObservationPipeline:
    return ObservationPipeline(uow_factory, provider, settings=MEMORY_INTELLIGENCE, assist=assist)


@pytest.mark.parametrize(
    ("switch", "expected"),
    [(True, True), (False, False), (None, False)],
    ids=["switched-on", "switched-off", "no-tenant-row"],
)
async def test_the_tenant_switch_reaches_extraction_and_consolidation(
    uow_factory, switch: bool | None, expected: bool
) -> None:
    ctx = await _tenant(uow_factory, switch)
    with mocked_gateway([{"facts": [], "times": [], "units": []}]) as gateway:
        assist = gateway.assist(uses=["contextual_extraction"])
        spy = Spy(MEMORY_INTELLIGENCE, assist=assist)
        spy.seen = []
        await _say(uow_factory, _pipeline(uow_factory, spy, assist), ctx, ["I live in Berlin."])
    assert spy.seen and {flag for _, flag in spy.seen} == {expected}


async def test_without_a_model_the_switch_is_not_even_read(uow_factory) -> None:
    """No model can be reached: the rules read every message, whatever the tenant chose."""
    ctx = await _tenant(uow_factory, True)
    spy = Spy(MEMORY_INTELLIGENCE)
    spy.seen = []
    await _say(uow_factory, _pipeline(uow_factory, spy, LLMAssist.disabled()), ctx, ["Hi there."])
    assert {flag for _, flag in spy.seen} <= {False}


async def test_a_replaced_value_is_never_current_without_a_model(uow_factory) -> None:
    ctx = await _tenant(uow_factory, True)
    native = NativeMemoryIntelligence(MEMORY_INTELLIGENCE)
    pipeline = _pipeline(uow_factory, native, LLMAssist.disabled())
    await _say(uow_factory, pipeline, ctx, ["I live in Berlin.", "I moved to Munich."])
    lives = [m.object for m in await _current(uow_factory, ctx) if m.predicate == "lives_in"]
    assert lives == ["munich"]


async def test_a_replaced_value_is_never_current_with_the_model(uow_factory) -> None:
    """The rules read "I live in Berlin"; only the model reads the German correction. Its
    lives_in replaces Berlin: Berlin is closed, München is current, and the model's paraphrase
    of the same value a day later is not a second memory."""
    ctx = await _tenant(uow_factory, True)
    replies = [
        {"facts": [], "times": []},
        {
            "facts": [
                {
                    "text": "Der Nutzer wohnt jetzt in München",
                    "kind": "profile",
                    "slot": "lives_in",
                    "value": "München",
                    "sentences": [0],
                }
            ],
            "times": [],
        },
        {
            "facts": [
                {
                    "text": "Der Nutzer lebt in München",
                    "kind": "profile",
                    "slot": "lives_in",
                    "value": "München",
                    "sentences": [0],
                }
            ],
            "times": [],
        },
    ]
    with mocked_gateway(replies) as gateway:
        assist = gateway.assist(uses=["contextual_extraction"])
        native = NativeMemoryIntelligence(MEMORY_INTELLIGENCE, assist=assist)
        await _say(
            uow_factory,
            _pipeline(uow_factory, native, assist),
            ctx,
            ["I live in Berlin.", "Ich wohne jetzt in München.", "Ich lebe in München."],
        )
        assert gateway.route.call_count == 3, "one call per message"
    current = await _current(uow_factory, ctx)
    lives = [m for m in current if m.predicate == "lives_in"]
    assert [m.object for m in lives] == ["München"]
    assert lives[0].reinforcement_count == 2, "the paraphrase reinforced it"
