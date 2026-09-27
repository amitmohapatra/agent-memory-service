"""Continuous consolidation must bridge time without crossing access boundaries."""

from datetime import timedelta

import pytest

from memory_service.domain.enums import Visibility
from memory_service.modules.memory.reflection import ReflectionService
from tests.support_llm import mocked_gateway
from tests.unit.test_llm_memory import CTX, _Memories, _sources, _UoW

pytestmark = pytest.mark.unit


async def test_new_user_fact_recovers_older_same_user_fact_across_threads():
    old = await _sources("I prefer concise answers.", age=timedelta(days=10))
    fresh = await _sources(
        "I prefer bullet points.", ctx=CTX.model_copy(update={"thread_id": "thread-two"})
    )
    uow = _UoW(_Memories(fresh + old))
    with mocked_gateway([{"insights": []}]) as gateway:
        service = ReflectionService(lambda: uow, assist=gateway.assist(uses=["reflection"]))
        await service.reflect_all()
    prompt = gateway.prompts()[0]["messages"][1]["content"]
    assert old[0].memory_id in prompt and fresh[0].memory_id in prompt


async def test_history_excludes_other_owners_and_private_sources():
    fresh = await _sources("I prefer concise answers.")
    private = await _sources("I prefer private medical reminders.", age=timedelta(days=10))
    private[0].visibility = Visibility.PRIVATE
    private[0].system_metadata["visibility_keys"] = ["principal:acme/user:u1"]
    other = await _sources("I prefer long answers.", age=timedelta(days=10))
    other[0].owner_principal = "user:other"
    uow = _UoW(_Memories(fresh + private + other))
    service = ReflectionService(lambda: uow)
    selected = await service._with_history(fresh)
    assert [m.memory_id for m in selected] == [fresh[0].memory_id]


@pytest.mark.parametrize("visibility", [Visibility.PRIVATE, Visibility.USER, Visibility.TENANT])
async def test_reflection_retains_exact_source_audience(visibility):
    sources = await _sources("I prefer concise answers.", "I prefer bullet points.")
    keys = {
        Visibility.PRIVATE: ["principal:acme/user:u1"],
        Visibility.USER: ["principal:acme/user:u1", "user:acme/u1"],
        Visibility.TENANT: ["principal:acme/user:u1", "tenant:acme"],
    }[visibility]
    for source in sources:
        source.visibility = visibility
        source.system_metadata["visibility_keys"] = keys
    service = ReflectionService(lambda: None)
    memory = service._insight(
        {
            "content": "User prefers concise answers and bullet points.",
            "memory_type": "PREFERENCE",
            "source_memory_ids": [m.memory_id for m in sources],
        },
        {m.memory_id: m for m in sources},
        CTX,
        now=sources[0].updated_at,
    )
    assert memory.visibility is visibility
    assert memory.system_metadata["visibility_keys"] == sorted(keys)


async def test_empty_success_is_not_repeated_and_bounded_backlog_advances():
    sources = await _sources(*[f"I prefer format number {i}." for i in range(45)])
    uow = _UoW(_Memories(sources))
    with mocked_gateway([{"insights": []}] * 3) as gateway:
        for _ in range(3):
            service = ReflectionService(
                lambda: uow, assist=gateway.assist(uses=["reflection"]), max_batches=1
            )
            assert await service.reflect_all() == []
        assert len(uow.memories.reflected) == 45
        assert await service.reflect_all() == []
        assert gateway.route.call_count == 3


async def test_changed_revision_is_reconsidered_after_a_successful_empty_batch():
    sources = await _sources("I prefer concise answers.", "I prefer bullet points.")
    uow = _UoW(_Memories(sources))
    with mocked_gateway([{"insights": []}] * 2) as gateway:
        service = ReflectionService(lambda: uow, assist=gateway.assist(uses=["reflection"]))
        await service.reflect_all()
        sources[0].revision += 1
        await service.reflect_all()
        assert gateway.route.call_count == 2
        assert uow.memories.reflected[sources[0].memory_id] == sources[0].revision
