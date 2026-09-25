import pytest

from memory_service.domain.enums import ScopeLevel, Visibility
from memory_service.domain.memory import Scope
from memory_service.modules.authz.visibility import VisibilitySpecification, visibility_keys
from memory_service.ports.authorization import AuthorizedScope


def test_visibility_keys_by_visibility() -> None:
    scope = Scope(
        level=ScopeLevel.THREAD,
        tenant_id="acme",
        thread_id="thr1",
        user_id="u1",
    )
    assert visibility_keys("acme", Visibility.PRIVATE, owner_principal="agent:a1", scope=scope) == [
        "principal:acme/agent:a1"
    ]
    assert visibility_keys("acme", Visibility.USER, owner_principal="user:u1", scope=scope) == [
        "user:acme/u1"
    ]
    assert visibility_keys("acme", Visibility.THREAD, owner_principal="user:u1", scope=scope) == [
        "thread:acme/thr1"
    ]
    assert visibility_keys("acme", Visibility.TENANT, owner_principal="user:u1") == ["tenant:acme"]
    assert visibility_keys(
        "acme", Visibility.AGENT_GROUP, owner_principal="user:u1", agent_group_id="crew"
    ) == ["agroup:acme/crew"]
    # RUN is directional and carries no author key - the author key is what used to make it
    # an identity audience instead of a run one. "run:" is read by this run and the runs it
    # spawns; "runup:" is read ONLY by the run that spawned this one, so a sibling carrying
    # run:<parent> never sees it.
    assert visibility_keys(
        "acme",
        Visibility.RUN,
        owner_principal="agent:u1/a1",
        agent_run_id="run2",
        parent_agent_run_id="run1",
    ) == ["run:acme/run2", "runup:acme/run1"]
    assert visibility_keys(
        "acme", Visibility.RUN, owner_principal="agent:u1/a1", agent_run_id="run2"
    ) == ["run:acme/run2"]
    with pytest.raises(ValueError, match="RUN visibility requires agent_run_id"):
        visibility_keys("acme", Visibility.RUN, owner_principal="user:u1")


def test_specification_from_scope_and_filter() -> None:
    scope = AuthorizedScope(
        tenant_id="acme",
        principal="user:u1",
        user_id="u1",
        thread_ids=["thr1"],
        agent_group_ids=[],
    )
    spec = VisibilitySpecification.from_scope(scope)
    assert spec.allows("acme", ["thread:acme/thr1"])
    assert spec.allows("acme", ["user:acme/u1"])
    assert spec.allows("acme", ["principal:acme/user:u1"])
    assert not spec.allows("acme", ["thread:acme/thr2"])
    assert not spec.allows("acme", ["user:acme/u2"])
    assert not spec.allows("acme", ["principal:acme/agent:a1"])
    assert not spec.allows("globex", ["tenant:globex"])
    flt = spec.search_filter(collection="memories")
    assert flt.tenant_id == "acme" and flt.must == {"collection": "memories"}
    assert "thread:acme/thr1" in flt.must_any["visibility_keys"]
