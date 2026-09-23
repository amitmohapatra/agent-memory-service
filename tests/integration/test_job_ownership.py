"""A task-queue job id is only readable by the tenant whose outbox dispatched it.

``GET /v1/jobs/{job_id}`` accepts two id shapes. The ``obx_`` form was always scoped - it
reads the outbox row and compares its tenant. The bare task-queue form was not: the queue's
ids are sequential integers and the ``TaskQueue`` port has never modelled a tenant, so the
route returned any tenant's task_name, queue, status, attempts and schedule to whoever named
an id, and sequential ids do not have to be guessed. Every other read in the service is
scoped by visibility keys; this one had nothing to scope by.

The outbox is what knows who dispatched a job, which is what ``dispatcher_of`` reads.
"""

from __future__ import annotations

import pytest

from memory_service.ports.tasks import JobSpec, Queue

pytestmark = [pytest.mark.integration]

TENANT = "acme"
OTHER = "globex"


async def _noop(payload):
    return payload


async def _dispatch_for(container, uow_factory, tenant: str) -> str:
    """Enqueue one job for ``tenant`` and return the task-queue id it was dispatched under."""
    container.tasks.register("noop", Queue.CHAT_FAST, _noop)
    async with uow_factory() as uow:
        await uow.enqueue(
            JobSpec(task_name="noop", queue=Queue.CHAT_FAST, payload={}, tenant_id=tenant)
        )
        await uow.commit()
        job_ids = uow.dispatched_job_ids
    assert job_ids, "the outbox did not dispatch the job, so there is no id to scope"
    return job_ids[0]


async def test_the_dispatching_tenant_is_reported(container, uow_factory) -> None:
    job_id = await _dispatch_for(container, uow_factory, TENANT)
    async with uow_factory() as uow:
        assert await uow.outbox.dispatcher_of(job_id) == (True, TENANT)


async def test_another_tenants_job_reports_that_tenant_not_the_caller(
    container, uow_factory
) -> None:
    """The check the route makes: the id resolves, and it resolves to somebody else."""
    job_id = await _dispatch_for(container, uow_factory, OTHER)
    async with uow_factory() as uow:
        dispatched, owner = await uow.outbox.dispatcher_of(job_id)
    assert dispatched and owner == OTHER
    assert owner != TENANT, "the route refuses when the owner is not the caller"


async def test_an_id_this_service_never_dispatched_is_not_found(container, uow_factory) -> None:
    """Sequential ids can be enumerated, so an unknown one must not read as ownerless."""
    async with uow_factory() as uow:
        assert await uow.outbox.dispatcher_of("999999999") == (False, None)


async def test_a_job_with_no_tenant_is_reported_as_ownerless(container, uow_factory) -> None:
    """The service's own periodic work belongs to no tenant; the obx_ read allows it too."""
    job_id = await _dispatch_for(container, uow_factory, None)  # type: ignore[arg-type]
    async with uow_factory() as uow:
        dispatched, owner = await uow.outbox.dispatcher_of(job_id)
    assert dispatched is True and owner is None
