"""A user's short turn that refers back to the agent's answer keeps what it refers to.

"Approve it", "go with that", "use those two": said right after the agent's answer, the turn's
meaning is in that answer. Kept alone ("Approve it as my north order for this item."), a
later chat recalls that the user approved something and not what - the figure was in the
answer (break-and-fix round 2, a reorder quantity of 19 units). The turn is kept as said, and
beside it the same words with the answer's headline, so recall and the context carry both.
"""

from __future__ import annotations

import re
from typing import Final

from memory_service.domain.enums import Lifetime, MemoryType, ObservationKind
from memory_service.domain.observation import Observation
from memory_service.ports.intelligence import MemoryCandidate

#: A turn this long or longer says what it means itself.
MAX_CHARS: Final = 200
#: The headline of the answer kept beside the turn, at most.
HEADLINE_CHARS: Final = 240
#: Words that point at something said before (not at something named in the turn).
REFERS: Final = re.compile(
    r"\b(?:it|that|this|those|these|them|the (?:first|second|third|last|other) one|"
    r"the same|that one|this one)\b",
    re.IGNORECASE,
)
_MARKUP: Final = re.compile(r"[*_`#>]+|^\s*[-•]\s*")


def headline(answer: str) -> str:
    """The answer's first line with text in it, without its markup, at most
    :data:`HEADLINE_CHARS`."""
    for line in answer.splitlines():
        text = " ".join(_MARKUP.sub("", line).split())
        if text:
            return text[:HEADLINE_CHARS]
    return ""


def referred_turn(
    observation: Observation, candidates: list[MemoryCandidate], prior: Observation | None
) -> list[MemoryCandidate]:
    """The user's short referring turn, with the headline of the agent's answer before it:
    one candidate beside the turn kept as said (its verbatim candidate), or none."""
    text = observation.content.strip()
    if (
        prior is None
        or not prior.agent_authored
        or observation.kind is not ObservationKind.MESSAGE
        or observation.agent_authored
        or len(text) >= MAX_CHARS
        or not REFERS.search(text)
    ):
        return []
    said = next((c for c in candidates if c.category == "verbatim_turn"), None)
    about = headline(prior.content)
    if said is None or not about:
        return []
    return [
        said.model_copy(
            update={
                "content": f"{text} (in reply to: {about})",
                "memory_type": MemoryType.OBSERVATION,
                "lifetime": Lifetime.LONG_TERM,
                # above the bare turn: it says what the bare turn only points at
                "importance": max(said.importance, 0.5),
                "category": "referred_turn",
            }
        )
    ]
