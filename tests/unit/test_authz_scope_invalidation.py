"""A cached authorization scope is invalidated by grants, and by nothing else.

The scope cache used to be keyed on the TENANT and USER revisions, which every memory write
bumps: measured on the benchmark tenant, ingesting 369 turns bumped them 730 times, so an
actively-written tenant threw its scope away roughly every other request. Each miss is five
sequential ListObjects calls against the authorization service, and at ten requests a second
that collapsed into 503 DEPENDENCY_UNAVAILABLE from list_objects timeouts.

The mirror-image defect was worse: granting a thread or a document bumped nothing at all, so
a caller who had just been given access waited for the sixty-second TTL to see it.

Both directions are pinned here, because both were wrong at once and fixing one without the
other would have been worse than leaving them alone.
"""

from __future__ import annotations

from memory_service.domain.context import MemoryExecutionContext
from memory_service.modules.authz.service import AuthorizationService

CTX = MemoryExecutionContext(tenant_id="acme", user_id="u1", workspace_id="ws1")


def _fingerprint(**revisions: int) -> str:
    return AuthorizationService.revision_fingerprint(CTX, revisions)


def test_a_memory_write_does_not_invalidate_the_scope() -> None:
    """TENANT and USER move on every consolidate; the scope does not depend on them."""
    before = _fingerprint(**{"membership:": 1, "membership:u1": 1, "tenant:": 7, "user:u1": 7})
    after = _fingerprint(**{"membership:": 1, "membership:u1": 1, "tenant:": 8, "user:u1": 9})
    assert before == after


def test_a_tenant_wide_grant_invalidates_every_user_in_the_tenant() -> None:
    before = _fingerprint(**{"membership:": 1, "membership:u1": 1})
    after = _fingerprint(**{"membership:": 2, "membership:u1": 1})
    assert before != after


def test_a_grant_to_this_user_invalidates_only_this_user() -> None:
    mine = _fingerprint(**{"membership:": 1, "membership:u1": 2})
    theirs = _fingerprint(**{"membership:": 1, "membership:u1": 1})
    assert mine != theirs


def test_an_agent_scope_depends_on_the_agents_own_grants() -> None:
    ctx = CTX.model_copy(update={"agent_id": "ag1"})
    before = AuthorizationService.revision_fingerprint(
        ctx, {"membership:": 1, "membership:u1": 1, "membership:ag1": 1}
    )
    after = AuthorizationService.revision_fingerprint(
        ctx, {"membership:": 1, "membership:u1": 1, "membership:ag1": 2}
    )
    assert before != after
    # ...and a user-scoped caller never reads the agent's counter
    assert "ag1" not in _fingerprint(**{"membership:": 1, "membership:u1": 1})


def test_a_missing_counter_reads_as_zero_rather_than_failing() -> None:
    """A tenant that has never been granted anything still has a stable key."""
    assert _fingerprint() == _fingerprint(**{"membership:": 0, "membership:u1": 0})
