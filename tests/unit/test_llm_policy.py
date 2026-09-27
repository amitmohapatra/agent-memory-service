"""Read permission is task-local, exception-safe, and cannot widen an outer denial."""

import asyncio

import pytest

from memory_service.modules.llm.policy import model_call_policy, model_calls_allowed
from tests.support_llm import mocked_gateway

pytestmark = pytest.mark.unit


async def test_denied_read_never_calls_either_completion_method():
    with mocked_gateway() as gw:
        assist = gw.assist(["query_expansion"])
        try:
            with model_call_policy(False):
                assert not assist.wants("query_expansion")
                assert await assist.complete("query_expansion", system="s", user="u") is None
                assert (
                    await assist.structured(
                        "query_expansion", system="s", user="u", schema={"type": "object"}
                    )
                    is None
                )
            assert gw.route.call_count == 0
            assert assist.wants("query_expansion")  # ingestion permission remains intact
        finally:
            await assist.provider.close()


def test_nested_permission_and_exception_restore():
    with pytest.raises(RuntimeError), model_call_policy(False), model_call_policy(True):
        assert not model_calls_allowed()
        raise RuntimeError("cancelled operation")
    assert model_calls_allowed()


async def test_concurrent_reads_keep_independent_permissions():
    ready = asyncio.Event()

    async def denied():
        with model_call_policy(False):
            ready.set()
            await asyncio.sleep(0)
            assert not model_calls_allowed()

    async def allowed():
        await ready.wait()
        assert model_calls_allowed()

    await asyncio.gather(denied(), allowed())
