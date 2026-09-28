"""``retries`` means runs *after* the first: Procrastinate's ``job.attempts`` counts previous
runs, so ``max_attempts=N`` keeps retrying while fewer than N have failed. The webhook
delivery job depends on this arithmetic to make its last permitted attempt the DEAD one."""

from __future__ import annotations

from procrastinate.jobs import Job  # noqa: TID251  (the semantics under test are the SDK's)

from memory_service.adapters.tasks.procrastinate_queue import _retry_strategy
from memory_service.config.constants import WEBHOOKS


def _job(attempts: int) -> Job:
    return Job(
        id=1,
        queue="q",
        task_name="t",
        task_kwargs={},
        lock=None,
        queueing_lock=None,
        attempts=attempts,
    )


def test_retries_are_the_runs_after_the_first() -> None:
    strategy = _retry_strategy(2)
    assert strategy is not False
    assert strategy.get_retry_decision(exception=RuntimeError(), job=_job(0)) is not None
    assert strategy.get_retry_decision(exception=RuntimeError(), job=_job(1)) is not None
    assert strategy.get_retry_decision(exception=RuntimeError(), job=_job(2)) is None
    assert _retry_strategy(0) is False


def test_the_delivery_job_gets_exactly_max_attempts_runs() -> None:
    strategy = _retry_strategy(WEBHOOKS.max_attempts - 1)
    assert strategy is not False
    total = 1  # run number; the job's ``attempts`` during run n is n - 1
    while strategy.get_retry_decision(exception=RuntimeError(), job=_job(total - 1)) is not None:
        total += 1
    assert total == WEBHOOKS.max_attempts
