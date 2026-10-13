"""What makes a verdict a person's vote, and who cast it (ADR 0039).

A verdict on a run, a memory or a procedure is a vote unless the service can tell it is not
one: it waits for the tenant's administrator, and a voter counts once per target. What is not
a vote is decided by who sent it, never by the ``source`` it claims (a client may write
``source=judge`` on a thumb): the tenant's administrator credential or the service itself
(``trusted``), a run's own status or judge's word sent with a credential that may act as that
agent (``run_vouched``), a decision on a tool call, an edit of a memory carrying new text. The
voter is the person (``user:<id>``) whichever agent carried the vote; an agent with no person
behind it is a voter of its own.
"""

from __future__ import annotations

from typing import Any, Final

from memory_service.domain.feedback import (
    CORRECTING_VERDICTS,
    HUMAN_SOURCES,
    THUMBS_UP_VERDICTS,
    Feedback,
    FeedbackSource,
    FeedbackTargetKind,
)

#: The targets a person votes on.
VOTE_TARGETS: Final = frozenset(
    {FeedbackTargetKind.RUN, FeedbackTargetKind.MEMORY, FeedbackTargetKind.PROCEDURE}
)
#: Why a vote changed nothing: the same voter's same thumb already counts.
ALREADY_COUNTED_REASON: Final = "the same vote from the same user already counts"


def correction_text(correction: Any) -> str:
    """The text a correction carries: a string, or ``{content: ...}``; "" otherwise."""
    if isinstance(correction, str):
        return correction.strip()
    if isinstance(correction, dict) and isinstance(correction.get("content"), str):
        return str(correction["content"]).strip()
    return ""


def rewrites(feedback: Feedback) -> bool:
    """A correction or edit carrying new text: on a memory, an owner's edit applied at once."""
    return feedback.verdict in CORRECTING_VERDICTS and bool(correction_text(feedback.correction))


#: What an agent's own credential may say without it being a vote: its run's status, a judge's
#: word it relays.
AUTOMATED_SOURCES: Final = frozenset({FeedbackSource.SYSTEM, FeedbackSource.JUDGE})


def classify(feedback: Feedback, *, trusted: bool, run_vouched: bool) -> bool:
    """Whether ``feedback`` is a vote, by who sent it: everything on a run, a memory or a
    procedure is, except the administrator's or the service's own word, an automated word
    from a credential vouched for the agent, and an edit of a memory."""
    if trusted or feedback.target_kind not in VOTE_TARGETS:
        return False
    if run_vouched and feedback.source in AUTOMATED_SOURCES:
        return False
    return not (feedback.target_kind is FeedbackTargetKind.MEMORY and rewrites(feedback))


def is_vote(feedback: Feedback) -> bool:
    """As classified when stored; a record older than the flag by its source."""
    if feedback.vote is not None:
        return feedback.vote
    if feedback.source not in HUMAN_SOURCES or feedback.target_kind not in VOTE_TARGETS:
        return False
    return not (feedback.target_kind is FeedbackTargetKind.MEMORY and rewrites(feedback))


def thumbs_up(feedback: Feedback) -> bool:
    return feedback.verdict in THUMBS_UP_VERDICTS


def same_thumb(a: Feedback, b: Feedback) -> bool:
    return thumbs_up(a) == thumbs_up(b)


def voter_key(user_id: str | None, agent_id: str | None) -> str | None:
    """Who a vote counts against: the person when there is one, else the agent."""
    if user_id is not None:
        return f"user:{user_id}"
    if agent_id is not None:
        return f"agent:{agent_id}"
    return None
