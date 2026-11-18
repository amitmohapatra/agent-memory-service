"""A job runs as background work (``domain.work``): the in-process models serve a request
waiting beside it first."""

from __future__ import annotations

import pytest

from memory_service.domain.work import background, in_background
from memory_service.modules.jobs.registry import _accounted

pytestmark = pytest.mark.unit


async def test_a_job_handler_runs_as_background_work_and_a_request_does_not() -> None:
    seen: list[bool] = []

    async def handler(payload: dict) -> None:
        seen.append(in_background())

    await _accounted("t", handler)({})
    assert seen == [True]
    assert in_background() is False, "the flag does not leak out of the job"
    with background():
        assert in_background()
    assert not in_background()
