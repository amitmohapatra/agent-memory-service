"""Beliefs: what the memory service holds about one person, consolidated from their facts.

"Observation" already names a raw turn here (``observations``, ``memory.process_observation``),
so the consolidated records are **beliefs**. One belief is one facet of a person (a line,
dates kept), written by the consolidation job from that person's memories and citing them
(``source_ids``). A change of state reuses supersession, as memories do: the new version is
current and the old one points at it (``superseded_by``); a belief a fact ended carries
``valid_to``; a belief left without sources is deleted, its words with it.

The unit of consolidation is one person in one scope (``scope_key``, the memories' own), so
a belief never mixes scopes; its readers are its sources' common audience
(``visibility_keys``), so a reader sees a belief only when they may read every memory it
cites. Beliefs are derived data: deleting them and consolidating again restores them.
"""

from __future__ import annotations

from datetime import datetime
from typing import Final

from pydantic import BaseModel, ConfigDict, Field

from memory_service.domain.ids import content_hash, new_id

#: The kind of principal a person is: phase 1 consolidates people the service knows as users.
PERSON_PREFIX: Final = "user:"
#: Superseded versions kept per belief beyond the current one (design note §5); the hourly
#: consolidation sweep deletes older ones.
VERSIONS_KEPT: Final = 3
#: The facets a belief is filed under, in the order a person's page shows them: who they are,
#: then what they like, value and believe, then the rest. The one place the order lives (the
#: consolidation prompt offers these labels, the page sorts by them). Put last, values were
#: what a capped page cut: 109 of 154 such lines in the Haiku-extraction LoCoMo corpus.
FACETS: Final = (
    "identity",
    "values",
    "family",
    "friends",
    "work",
    "places",
    "hobbies",
    "events",
    "plans",
    "other",
)


def person_name(person_key: str) -> str:
    """What a page calls the person: ``user:caroline`` -> ``caroline``."""
    return person_key.split(":", 1)[1] if ":" in person_key else person_key


class Belief(BaseModel):
    model_config = ConfigDict(frozen=True)

    belief_id: str
    tenant_id: str
    scope_key: str
    person_key: str
    visibility_keys: list[str]
    text: str
    #: a short topic label the page groups lines by
    facet: str = ""
    #: the span of the dates the source facts were said
    first_seen_at: datetime
    last_seen_at: datetime
    supersedes: str | None = None
    superseded_by: str | None = None
    valid_from: datetime | None = None
    valid_to: datetime | None = None
    content_hash: str
    index_fingerprint: str | None = None
    revision: int = 1
    created_at: datetime
    updated_at: datetime
    deleted_at: datetime | None = None
    #: when a memory it cites was superseded (a newer value said): hidden from every reader
    #: until the person's next run has weighed the newer value against it
    stale_since: datetime | None = None
    source_ids: list[str] = Field(default_factory=list)

    @property
    def proof_count(self) -> int:
        return len(self.source_ids)

    @property
    def current(self) -> bool:
        return self.superseded_by is None and self.valid_to is None and self.deleted_at is None

    @property
    def shown(self) -> bool:
        """Current and not waiting on a newer value: what a reader may be shown."""
        return self.current and self.stale_since is None

    @classmethod
    def new(
        cls,
        *,
        tenant_id: str,
        scope_key: str,
        person_key: str,
        visibility_keys: list[str],
        text: str,
        facet: str,
        seen: tuple[datetime, datetime],
        source_ids: list[str],
        now: datetime,
        supersedes: str | None = None,
    ) -> Belief:
        return cls(
            belief_id=new_id("belief"),
            tenant_id=tenant_id,
            scope_key=scope_key,
            person_key=person_key,
            visibility_keys=sorted(visibility_keys),
            text=text,
            facet=facet,
            first_seen_at=seen[0],
            last_seen_at=seen[1],
            supersedes=supersedes,
            valid_from=seen[0],
            content_hash=content_hash(text),
            created_at=now,
            updated_at=now,
            source_ids=list(dict.fromkeys(source_ids)),
        )


class BeliefProgress(BaseModel):
    """Where one person's consolidation in one scope stands."""

    model_config = ConfigDict(frozen=True)

    tenant_id: str
    scope_key: str
    person_key: str
    #: the facts the last run consolidated (their ids and revisions), hashed
    source_fingerprint: str = ""
    #: the write time of the last fact consolidated: the next run reads the facts after it.
    #: None: the next run is a full rebuild.
    last_consolidated_at: datetime | None = None
    #: set when a new fact arrives, cleared when nothing is left to consolidate: the lag
    pending_since: datetime | None = None
    failures: int = 0
    #: set after repeated failures: automatic runs stop until a fact or an operator resets it
    paused_at: datetime | None = None
