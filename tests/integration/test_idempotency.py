import pytest

from memory_service.domain.errors import Conflict
from memory_service.modules.idempotency.service import IdempotencyService

pytestmark = [pytest.mark.integration]

TENANT = "acme"


async def test_idempotency_reserve_replay_and_conflict(container, uow_factory) -> None:
    svc: IdempotencyService = container.services["idempotency"]
    h = svc.request_hash({"content": "hello"})

    async with uow_factory() as uow:
        assert await svc.begin(uow.idempotency, TENANT, "k1", h) is None  # reserved
        await uow.commit()

    # in flight (not completed) -> retryable conflict
    async with uow_factory() as uow:
        with pytest.raises(Conflict) as exc:
            await svc.begin(uow.idempotency, TENANT, "k1", h)
        assert exc.value.retryable is True

    async with uow_factory() as uow:
        await svc.complete(uow.idempotency, TENANT, "k1", status=202, body={"message_id": "msg_1"})
        await uow.commit()
    await svc.warm_cache(TENANT, "k1", h, status=202, body={"message_id": "msg_1"})

    # replay from cache (fast path) and from DB
    cached = await svc.lookup_cached(TENANT, "k1", h)
    assert cached is not None and cached.status == 202 and cached.body == {"message_id": "msg_1"}
    async with uow_factory() as uow:
        replay = await svc.begin(uow.idempotency, TENANT, "k1", h)
        assert replay is not None and replay.body == {"message_id": "msg_1"}

    # same key, different payload -> conflict (cache and DB paths)
    with pytest.raises(Conflict):
        await svc.lookup_cached(TENANT, "k1", svc.request_hash({"content": "different"}))
    async with uow_factory() as uow:
        with pytest.raises(Conflict):
            await svc.begin(
                uow.idempotency, TENANT, "k1", svc.request_hash({"content": "different"})
            )

    # keys are tenant scoped
    async with uow_factory() as uow:
        assert await svc.begin(uow.idempotency, "other-tenant", "k1", h) is None


async def test_idempotency_survives_cache_outage(container, uow_factory) -> None:
    svc: IdempotencyService = container.services["idempotency"]
    container.cache.available = False
    h = svc.request_hash({"a": 1})
    assert await svc.lookup_cached(TENANT, "k2", h) is None
    async with uow_factory() as uow:
        assert await svc.begin(uow.idempotency, TENANT, "k2", h) is None
        await svc.complete(uow.idempotency, TENANT, "k2", status=200, body={})
        await uow.commit()
    await svc.warm_cache(TENANT, "k2", h, status=200, body={})  # must not raise
    async with uow_factory() as uow:
        assert (await svc.begin(uow.idempotency, TENANT, "k2", h)) is not None
