"""Outbound webhooks as a tenant operates them: subscribe, list, read, change, test, read the
deliveries, unsubscribe. Seven operations.

The receiver is a port nothing listens on, so the test delivery fails - which is the honest way
to exercise ``GET /v1/webhooks/{id}/deliveries``: a delivery row with an attempt, a status and
the error, which is what an operator opens that route to see.
"""

from __future__ import annotations

import base64

import pytest

from memory_service.api.app import create_app
from tests.agent.conftest import BOOTSTRAP, sdk
from tests.conftest import PG_AVAILABLE, _test_overrides
from trellis.memory import MemoryError

pytestmark = pytest.mark.e2e

ENVELOPE = base64.urlsafe_b64encode(b"p9b-webhook-envelope-key-32-byte").decode()
#: Loopback, and a port nothing listens on: refused at once, no waiting on a timeout.
RECEIVER = "http://127.0.0.1:9/hook"
EVENTS = ["memory.created", "webhook.test"]


@pytest.fixture
def app(make_settings):
    if not PG_AVAILABLE:
        pytest.skip("PostgreSQL not reachable")
    settings = make_settings(
        authentication={"mode": "api_key", "bootstrap_admin_key": BOOTSTRAP},
        agent_credentials={"active_key_id": "p9b", "encryption_keys": {"p9b": ENVELOPE}},
        # A deployed service refuses a loopback receiver; development permits it, and that is
        # the one deployment fact this group has.
        webhooks={"allow_local_targets": True},
    )
    return create_app(settings, overrides=_test_overrides(tasks="inline"))


async def _admin(app, tenant_id: str = "acme"):
    platform = sdk(app, BOOTSTRAP)
    tenant = await platform.admin.create_tenant(tenant_id.title(), tenant_id=tenant_id)
    return sdk(app, tenant.admin_key.token)


@pytest.mark.covers(
    "webhooks.create_webhook",
    "webhooks.list_webhooks",
    "webhooks.get_webhook",
    "webhooks.update_webhook",
    "webhooks.test_webhook",
    "webhooks.list_deliveries",
    "webhooks.delete_webhook",
)
async def test_a_tenant_subscribes_tests_and_unsubscribes(app, running) -> None:
    admin = await _admin(app)

    created = await admin.tenant.webhooks.create(
        RECEIVER, EVENTS, description="ops pager", idempotency_key="hook-1"
    )
    assert created.subscription_id and created.enabled is True
    assert created.url == RECEIVER and set(created.events) == set(EVENTS)
    assert created.secret, "the signing secret is shown once, on creation"
    assert created.tenant_id == "acme" and created.created_by

    # A replayed subscribe is the same subscription, and never a second secret. The retry is
    # the same request: a client that sends an Idempotency-Key with a different body is told
    # so (409), which is what that header is for.
    replay = await admin.tenant.webhooks.create(
        RECEIVER, EVENTS, description="ops pager", idempotency_key="hook-1"
    )
    assert replay.subscription_id == created.subscription_id and replay.secret is None
    with pytest.raises(MemoryError) as changed_body:
        await admin.tenant.webhooks.create(RECEIVER, ["webhook.test"], idempotency_key="hook-1")
    assert changed_body.value.status == 409

    listed = await admin.tenant.webhooks.list()
    assert [w.subscription_id for w in listed] == [created.subscription_id]
    assert all(not getattr(w, "secret", None) for w in listed), "a listing carries no secret"

    one = await admin.tenant.webhooks.get(created.subscription_id)
    assert one.description == "ops pager" and one.failures == 0

    changed = await admin.tenant.webhooks.update(
        created.subscription_id, events=["webhook.test"], description="quieter"
    )
    assert changed.events == ["webhook.test"] and changed.description == "quieter"
    assert changed.url == RECEIVER, "what was not sent is not changed"
    assert changed.updated_at >= created.updated_at

    delivery = await admin.tenant.webhooks.test(created.subscription_id)
    assert delivery.subscription_id == created.subscription_id
    assert delivery.event_type == "webhook.test" and delivery.event_id

    deliveries = await admin.tenant.webhooks.deliveries(created.subscription_id)
    assert [d.delivery_id for d in deliveries] == [delivery.delivery_id]
    attempted = deliveries[0]
    # Nothing is listening on the receiver, so the attempt is recorded as a failure with the
    # reason: an operator reads this route to find out why nothing arrived.
    assert attempted.attempts >= 1 and attempted.status in ("FAILED", "DEAD", "PENDING")
    if attempted.status in ("FAILED", "DEAD"):
        assert attempted.last_error and attempted.delivered_at is None

    await admin.tenant.webhooks.delete(created.subscription_id)
    with pytest.raises(MemoryError) as gone:
        await admin.tenant.webhooks.get(created.subscription_id)
    assert gone.value.status == 404
    assert await admin.tenant.webhooks.list() == []


@pytest.mark.covers_error(
    "webhooks.create_webhook",
    "webhooks.list_webhooks",
    "webhooks.get_webhook",
    "webhooks.update_webhook",
    "webhooks.test_webhook",
    "webhooks.list_deliveries",
    "webhooks.delete_webhook",
)
async def test_no_other_tenant_and_no_service_key_touches_a_subscription(app, running) -> None:
    admin = await _admin(app, "acme")
    other = await _admin(app, "globex")
    mine = await admin.tenant.webhooks.create(RECEIVER, EVENTS)
    key = await admin.tenant.keys.issue("service", "harness")
    service = sdk(app, key.token)

    # Another tenant's admin cannot see or steer this subscription. Every route that has to
    # resolve it answers "not found" - never "forbidden", which would confirm the id exists.
    for call in (
        other.tenant.webhooks.get(mine.subscription_id),
        other.tenant.webhooks.update(mine.subscription_id, enabled=False),
        other.tenant.webhooks.test(mine.subscription_id),
        other.tenant.webhooks.deliveries(mine.subscription_id),
    ):
        with pytest.raises(MemoryError) as refused:
            await call
        assert refused.value.status == 404, refused.value
    assert await other.tenant.webhooks.list() == []

    # Delete converges rather than resolving first (the repository key is (tenant, id), so it
    # deletes nothing), which is the same contract as deleting a workspace twice. What matters
    # is that the subscription is still the owner's afterwards, enabled and unchanged.
    await other.tenant.webhooks.delete(mine.subscription_id)
    still_mine = await admin.tenant.webhooks.get(mine.subscription_id)
    assert still_mine.enabled is True and still_mine.url == RECEIVER
    assert [w.subscription_id for w in await admin.tenant.webhooks.list()] == [mine.subscription_id]

    # A data-plane key subscribes to nothing: a webhook is the tenant's, not a harness's.
    for call in (
        service.tenant.webhooks.create(RECEIVER, EVENTS),
        service.tenant.webhooks.list(),
        service.tenant.webhooks.get(mine.subscription_id),
        service.tenant.webhooks.update(mine.subscription_id, enabled=False),
        service.tenant.webhooks.test(mine.subscription_id),
        service.tenant.webhooks.deliveries(mine.subscription_id),
        service.tenant.webhooks.delete(mine.subscription_id),
    ):
        with pytest.raises(MemoryError) as denied:
            await call
        assert denied.value.status == 403, denied.value
