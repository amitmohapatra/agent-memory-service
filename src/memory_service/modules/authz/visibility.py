"""Visibility semantics as a Specification.

Every stored object (memory, chunk, message) carries ``visibility_keys``: the set of
"audiences" that may read it, derived from its scope + visibility at write time. An
authorized principal has a bounded set of audience keys derived from its AuthorizedScope.
Read = non-empty intersection. This turns access control into one ``must_any`` filter that
runs *inside* the search store and inside SQL, before any candidate is returned.

Key grammar (tenant is always part of the key):
    tenant:<t>                  everyone in the tenant
    user:<t>/<user>             that user, any thread, and every agent acting for them
    thread:<t>/<thread>         one conversation
    agroup:<t>/<agent_group>    cooperating agents sharing a group id, at any depth
    run:<t>/<run>               one agent run, and the run that spawned it
    principal:<t>/<principal>   PRIVATE to exactly one principal (user:<id> or agent:<id>)
"""

from __future__ import annotations

from collections.abc import Iterable

from pydantic import BaseModel, ConfigDict, Field

from memory_service.domain.enums import Visibility
from memory_service.domain.memory import Scope
from memory_service.ports.authorization import AuthorizedScope
from memory_service.ports.search import SearchFilter

VISIBILITY_FIELD = "visibility_keys"


def visibility_keys(
    tenant_id: str,
    visibility: Visibility,
    *,
    owner_principal: str,
    scope: Scope | None = None,
    user_id: str | None = None,
    thread_id: str | None = None,
    agent_group_id: str | None = None,
    agent_run_id: str | None = None,
    parent_agent_run_id: str | None = None,
) -> list[str]:
    """Audience keys an object is readable by, given its visibility and anchors.

    The author's own ``principal:`` key is appended by ``readable_by`` for most visibilities,
    so a memory shared with a workspace or a group stays readable by whoever wrote it even if
    they later lose that membership. See ``_NO_OWNER_KEY`` for which are excluded, and for
    why THREAD is not among them despite arguably needing to be.

    That append used to live in ``keys_for``, one layer above, and the split is what hid a
    defect for so long: tests/security/test_isolation.py built its objects with THIS
    function, which never appended the key, so both sides of its property agreed on a key
    shape no stored row has - a release-blocking isolation gate proving nothing.
    """
    if scope is not None:
        user_id = user_id or scope.user_id
        thread_id = thread_id or scope.thread_id
        agent_group_id = agent_group_id or scope.agent_group_id
    t = tenant_id
    match visibility:
        case Visibility.PRIVATE:
            return [f"principal:{t}/{owner_principal}"]
        case Visibility.USER:
            if not user_id:
                raise ValueError("USER visibility requires user_id")
            return [f"user:{t}/{user_id}"]
        case Visibility.AGENT_GROUP:
            if not agent_group_id:
                raise ValueError("AGENT_GROUP visibility requires agent_group_id")
            return [f"agroup:{t}/{agent_group_id}"]
        case Visibility.RUN:
            if not agent_run_id:
                raise ValueError("RUN visibility requires agent_run_id")
            # This run, and the run that spawned it. The parent's key is written HERE, at
            # write time, which is what lets a supervisor read what its specialists produced.
            #
            # It used to carry the author's principal instead, and that made RUN an IDENTITY
            # audience rather than a run one: principal_id is agent:{user}/{agent} with no run
            # component, so every past and future run of the same agent read every other one's
            # scratch for the whole TTL. Measured: five parallel workers sharing one agent_id
            # saw each other's notes, a retry inherited the failed attempt's reasoning, and
            # the supervisor that spawned them saw none of it. Hand-off ran backwards.
            # Two directions, two namespaces, so hand-off does not become a party line.
            #   run:<mine>      anything I write, read by me and by the runs I spawn
            #   runup:<parent>  written by a child, read ONLY by the run that spawned it
            # A sibling carries run:<its own>, run:<parent> and runup:<its own>, so it never
            # matches runup:<parent> and never sees its sibling's notes. Peers that DO want to
            # collaborate say so with AGENT_GROUP, which is not bounded by the run tree.
            keys = [f"run:{t}/{agent_run_id}"]
            if parent_agent_run_id:
                keys.append(f"runup:{t}/{parent_agent_run_id}")
            return keys
        case Visibility.THREAD:
            if not thread_id:
                raise ValueError("THREAD visibility requires thread_id")
            return [f"thread:{t}/{thread_id}"]
        case Visibility.TENANT:
            return [f"tenant:{t}"]
    raise ValueError(f"unknown visibility {visibility}")  # pragma: no cover


#: Visibilities that do NOT carry the author's own principal key, because they already name
#: the principal themselves.
#:
#: THREAD is here because the author IS the audience, so the escape cancelled the scope: the
#: author matched their own key from any thread, the thread key was never reached, and
#: ``visibility=THREAD`` meant "this thread, or anywhere if you wrote it". An integrator
#: reported it as a new conversation recalling the previous one's turns.
#:
#: Removing it alone is not enough and breaks worse. A thread used to be granted in exactly
#: one place - ``conversation/service.py``, POST /v1/threads - so an observation naming a
#: thread nobody created was readable ONLY through the author key, and taking it away left
#: the author unable to read their own memory in the thread they wrote it in. The
#: observations route therefore ensures the thread, and ``from_scope`` narrows the audience
#: to the thread being read in. All three are needed; any one alone regresses something.
_NO_OWNER_KEY = frozenset({Visibility.PRIVATE, Visibility.RUN, Visibility.THREAD})


def readable_by(
    tenant_id: str, visibility: Visibility, *, owner_principal: str, **anchors: str | None
) -> list[str]:
    """``visibility_keys`` plus the author's own key, which is what a stored row carries."""
    keys = visibility_keys(
        tenant_id,
        visibility,
        owner_principal=owner_principal,
        **anchors,  # type: ignore[arg-type]
    )
    if visibility in _NO_OWNER_KEY:
        return keys
    owner = f"principal:{tenant_id}/{owner_principal}"
    return keys if owner in keys else [*keys, owner]


class VisibilitySpecification(BaseModel):
    """The audience keys a principal may read. Built once per request from AuthorizedScope."""

    model_config = ConfigDict(frozen=True)

    tenant_id: str
    keys: frozenset[str] = Field(default_factory=frozenset)
    truncated: bool = False

    @classmethod
    def from_scope(
        cls,
        scope: AuthorizedScope,
        *,
        current_thread_id: str | None = None,
        current_agent_run_id: str | None = None,
    ) -> VisibilitySpecification:
        """Audience keys this caller reads with.

        ``current_thread_id`` narrows the THREAD audiences to the one conversation the
        caller is in. Omitted, every granted thread is in scope, which is right for a
        deliberate cross-thread search and wrong for a turn inside a thread.

        ``current_agent_run_id`` adds the upward audience its own children write to, which is
        how a supervisor reads what it spawned without its children reading each other.
        """
        t = scope.tenant_id
        keys: set[str] = {f"tenant:{t}", f"principal:{t}/{scope.principal}"}
        if scope.user_id:
            keys.add(f"user:{t}/{scope.user_id}")
            # an agent acting for a user also sees that user's private memories? No:
            # PRIVATE means exactly one principal. Agents see USER-level memories of their user.
        # The thread being READ IN, not every thread ever granted. Being authorized for a
        # thread is not the same as working in it: a caller who owns twenty conversations
        # carried all twenty audiences into every query, so a memory scoped to one of them
        # was readable from all the others.
        #
        # Always an INTERSECTION with what was granted, never the named thread on its own.
        # Naming a thread must grant nothing - measured: a second user who names someone
        # else's thread matches no key today, and adding the bare name here would turn that
        # into a genuine cross-user leak.
        threads = scope.thread_ids
        if current_thread_id is not None:
            threads = [th for th in threads if th == current_thread_id]
        keys.update(f"thread:{t}/{th}" for th in threads)
        keys.update(f"agroup:{t}/{ag}" for ag in scope.agent_group_ids)
        keys.update(f"run:{t}/{r}" for r in scope.run_ids)
        if current_agent_run_id:
            # what my own children addressed upwards to me
            keys.add(f"runup:{t}/{current_agent_run_id}")
        return cls(tenant_id=t, keys=frozenset(keys), truncated=scope.truncated)

    def allows(self, object_tenant_id: str, object_keys: Iterable[str]) -> bool:
        if object_tenant_id != self.tenant_id:
            return False
        return any(k in self.keys for k in object_keys)

    def search_filter(self, **must: str | int | bool) -> SearchFilter:
        """Store-side filter: tenant equality AND any-of visibility keys."""
        return SearchFilter(
            tenant_id=self.tenant_id,
            must=dict(must),
            must_any={VISIBILITY_FIELD: sorted(self.keys)},
        )
