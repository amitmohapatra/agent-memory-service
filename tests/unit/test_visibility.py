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
        workspace_id="ws1",
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
    assert visibility_keys(
        "acme", Visibility.WORKSPACE, owner_principal="user:u1", scope=scope
    ) == ["ws:acme/ws1"]
    assert visibility_keys("acme", Visibility.TENANT, owner_principal="user:u1") == ["tenant:acme"]
    assert visibility_keys("acme", Visibility.GLOBAL, owner_principal="user:u1") == ["global"]
    assert visibility_keys(
        "acme", Visibility.GROUP, owner_principal="user:u1", group_id="legal"
    ) == ["group:acme/legal"]
    with pytest.raises(ValueError, match="GROUP visibility requires group_id"):
        visibility_keys("acme", Visibility.GROUP, owner_principal="user:u1")


def test_specification_from_scope_and_filter() -> None:
    scope = AuthorizedScope(
        tenant_id="acme",
        principal="user:u1",
        user_id="u1",
        workspace_ids=["ws1"],
        group_ids=["legal"],
        thread_ids=["thr1"],
        work_ids=["w1"],
        agent_group_ids=[],
    )
    spec = VisibilitySpecification.from_scope(scope)
    assert spec.allows("acme", ["thread:acme/thr1"])
    assert spec.allows("acme", ["user:acme/u1"])
    assert spec.allows("acme", ["principal:acme/user:u1"])
    assert spec.allows("acme", ["group:acme/legal", "nothing"])
    assert not spec.allows("acme", ["thread:acme/thr2"])
    assert not spec.allows("acme", ["user:acme/u2"])
    assert not spec.allows("acme", ["principal:acme/agent:a1"])
    assert not spec.allows("globex", ["global"])  # tenant mismatch beats everything
    assert not spec.allows("globex", ["tenant:globex"])
    flt = spec.search_filter(collection="memories")
    assert flt.tenant_id == "acme" and flt.must == {"collection": "memories"}
    assert "thread:acme/thr1" in flt.must_any["visibility_keys"]
