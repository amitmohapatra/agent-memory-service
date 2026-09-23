"""Revision-based cache invalidation.

Every mutation increments the revisions it affects. Cache keys embed the relevant
revisions, so stale entries simply stop being addressed; no cache scans, no deletion
storms. The revision counters live in PostgreSQL (canonical) with a Dragonfly read cache.
"""

from __future__ import annotations

from enum import StrEnum


class RevisionKind(StrEnum):
    TENANT = "tenant"
    WORKSPACE = "workspace"
    GROUP = "group"
    USER = "user"
    THREAD = "thread"
    DOCUMENT = "document"
    GRAPH = "graph"
    AGENT = "agent"
    #: What an authorization scope depends on: the grants themselves. Kept apart from
    #: TENANT and USER because those are bumped by every memory write, and a scope that
    #: is invalidated by content churn is resolved again for no reason - measured at 730
    #: bumps over 369 ingested turns, each one costing five sequential ListObjects calls.
    MEMBERSHIP = "membership"
