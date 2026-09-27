"""Bounded actor-specific search plans derived from authorized memory candidates.

An aggregate question can name several people while its first search favours one. Separate
the named actors from the topic and search each actor's memories with the same topic. This
is query decomposition, not generated evidence: plans contain only names already present
in the question and canonical subjects returned by an authorized search.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from memory_service.modules.retrieval.engine import Candidate

MAX_ACTORS = 2


@dataclass(frozen=True)
class MemoryQueryPlan:
    topic: str
    subjects: tuple[str, ...]


def plan_memory_queries(query: str, candidates: list[Candidate]) -> MemoryQueryPlan | None:
    subjects: dict[str, str] = {}
    for candidate in candidates:
        subject = candidate.payload.get("subject")
        if candidate.kind != "memory" or not isinstance(subject, str):
            continue
        prefix, separator, name = subject.partition(":")
        if not separator or prefix not in {"user", "agent", "person"} or len(name) < 3:
            continue
        pattern = rf"(?<!\w){re.escape(name)}(?:['\u2019]s)?(?!\w)"
        if re.search(pattern, query, flags=re.IGNORECASE):
            subjects[subject] = pattern
    # Ambiguous multi-party queries should retain normal retrieval, not silently select
    # whichever two actors happened to rank first.
    if not 1 <= len(subjects) <= MAX_ACTORS:
        return None
    topic = query
    for pattern in subjects.values():
        topic = re.sub(pattern, " ", topic, flags=re.IGNORECASE)
    topic = re.sub(r"\b(?:both|and)\b", " ", topic, flags=re.IGNORECASE)
    topic = " ".join(topic.split()).strip(" ?")
    if not topic:
        return None
    return MemoryQueryPlan(topic=topic, subjects=tuple(sorted(subjects)))
