"""Revision-based cache invalidation.

Every mutation increments the revisions it affects. Cache keys embed the relevant
revisions, so stale entries simply stop being addressed; no cache scans, no deletion
storms. The revision counters live only in PostgreSQL (the ``revisions`` table): a bump
commits with the write it describes, and a reader takes every counter it depends on in one
indexed statement per cache lookup (``ContextBuilder._revisions``). They are not cached
anywhere, because a cached counter is exactly the stale read this scheme exists to prevent.
"""

from __future__ import annotations

from collections.abc import Iterable
from enum import StrEnum


class RevisionKind(StrEnum):
    TENANT = "tenant"
    USER = "user"
    THREAD = "thread"
    #: Bumped by ingestion when a document's content changes; no bundle reads it. What a
    #: bundle reads when a document becomes searchable is its audience's revisions
    #: (``document_revision_keys``).
    DOCUMENT = "document"
    #: A graph change whose audience is unknown (a fact of a deleted memory, an edge with no
    #: visibility keys). Graph changes with an audience bump that audience's revisions
    #: instead (ADR 0031): one tenant-wide counter moved by every memory-index job dropped
    #: every cached bundle of the tenant on every write.
    GRAPH = "graph"
    AGENT = "agent"
    #: What an authorization scope depends on: the grants themselves. Kept apart from
    #: TENANT and USER because those are bumped by every memory write, and a scope that
    #: is invalidated by content churn is resolved again for no reason - measured at 730
    #: bumps over 369 ingested turns, each one costing five sequential ListObjects calls.
    MEMBERSHIP = "membership"


#: Audiences that can cross owner and agent identities. Their readers already subscribe to
#: TENANT, so a write for one of them bumps TENANT; no scope-resolution round trip is needed
#: on cache hits. Threadless searches include every granted thread while their cache key has
#: no single thread id, so the thread counter alone cannot invalidate them.
_SHARED_AUDIENCES = frozenset({"tenant", "agroup", "run", "runup", "thread", "workspace"})


def audience_revision_keys(
    tenant_id: str, visibility_keys: Iterable[str]
) -> set[tuple[RevisionKind, str]]:
    """The revisions a cached bundle of every reader of these audience keys depends on.

    ``visibility_keys`` are the stored audience keys of a memory, chunk or graph fact
    (``modules/authz/visibility.py``): ``user:<tenant>/<id>`` moves that user's counter,
    ``principal:<tenant>/user:<id>`` and ``.../agent:[<user>/]<id>`` the user's or the
    agent's, and every shared audience TENANT. Keys of another tenant are ignored. An empty
    result means the audiences named nothing a reader subscribes to; the caller decides
    whether that is TENANT (content) or nothing.
    """
    keys: set[tuple[RevisionKind, str]] = set()
    for audience in visibility_keys:
        kind, _, value = audience.partition(":")
        if kind in _SHARED_AUDIENCES:
            keys.add((RevisionKind.TENANT, ""))
        elif kind == "user":
            tenant, separator, identifier = value.partition("/")
            if tenant == tenant_id and separator and identifier:
                keys.add((RevisionKind.USER, identifier))
        elif kind == "principal":
            tenant, _, principal = value.partition("/")
            if tenant != tenant_id:
                continue
            principal_kind, _, identifier = principal.partition(":")
            if principal_kind == "user" and identifier:
                keys.add((RevisionKind.USER, identifier))
            elif principal_kind == "agent" and identifier:
                # Both bound agent:user/id and unattended agent:id subscribe to AGENT:id.
                keys.add((RevisionKind.AGENT, identifier.rsplit("/", 1)[-1]))
            else:
                keys.add((RevisionKind.TENANT, ""))
    return keys


def document_revision_keys(
    tenant_id: str, thread_id: str | None, visibility_keys: Iterable[str]
) -> set[tuple[RevisionKind, str]]:
    """The revisions every reader of a document's chunks and graph facts depends on: its
    audiences, and its thread; TENANT when they name nothing (a document already gone)."""
    keys = audience_revision_keys(tenant_id, visibility_keys)
    if thread_id:
        keys.add((RevisionKind.THREAD, thread_id))
    return keys or {(RevisionKind.TENANT, "")}
