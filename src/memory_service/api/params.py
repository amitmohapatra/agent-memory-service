"""Query and path parameters the routes share, each with the one description it carries
wherever it appears (ADR 0030).

A parameter repeated on twenty routes was described on none: ``agent_run_id`` meant the same
thing everywhere and the OpenAPI document said nothing about any of them. These aliases are
the one place the wording lives, so a route that names a scope query parameter or a path id
documents it by using it, and ``tests/contract/test_openapi_descriptions.py`` keeps every
parameter of every operation described.
"""

from __future__ import annotations

from typing import Annotated, Any, Final

from fastapi import Path, Query

#: The identifier rule every id below follows (``domain.ids.ID_PATTERN``).
ID_RULE: Final = "An id: a letter or digit, then letters, digits and ._:-, at most 200 characters."


def _scope(text: str) -> Any:
    return Query(description=f"{text} {ID_RULE}")


ThreadIdQuery = Annotated[
    str | None,
    _scope(
        "The conversation thread the call acts in: THREAD-visible records of this thread are "
        "readable, and writes are anchored to it."
    ),
]
SessionIdQuery = Annotated[
    str | None, _scope("The open session within the thread (one sitting of a conversation).")
]
WorkIdQuery = Annotated[
    str | None,
    _scope(
        "The unit of work spanning several agents and turns: WORK-visible records of it are "
        "readable."
    ),
]
TaskIdQuery = Annotated[str | None, _scope("The task inside the unit of work.")]
AgentIdQuery = Annotated[
    str | None,
    _scope(
        "The logical agent acting (e.g. research): the call acts as agent:<id> for the user, "
        "so the agent's own memories and keys apply."
    ),
]
AgentGroupIdQuery = Annotated[
    str | None,
    _scope("The group of cooperating agents: AGENT_GROUP-visible records of it are readable."),
]
AgentRunIdQuery = Annotated[
    str | None,
    _scope(
        "This execution of the agent (requires agent_id): RUN-visible records written by the "
        "run, and by the run that spawned it, are readable."
    ),
]
ParentAgentRunIdQuery = Annotated[
    str | None,
    _scope("The run that spawned this one, whose RUN-visible records this run may read."),
]


def limit_query(maximum: int, what: str = "items") -> Any:
    """``limit`` of a paged list: how many ``what`` one page holds, 1..``maximum``."""
    return Query(
        ge=1,
        le=maximum,
        description=f"The most {what} one page holds (1-{maximum}); the next page is named by "
        '`next_cursor` / `Link: rel="next"`.',
    )


def _path(text: str) -> Any:
    return Path(description=text)


MemoryIdPath = Annotated[
    str,
    _path(
        "The memory: its id (mem_...), or the handle a context cited it by (m3) together with "
        "that context's bundle_id."
    ),
]
ThreadIdPath = Annotated[str, _path(f"The conversation thread. {ID_RULE}")]
MessageIdPath = Annotated[str, _path("The message id (msg_...).")]
JobIdPath = Annotated[
    str, _path("The job: an outbox reference (obx_<n>) from a write's job_ids, or a queue id.")
]
DocumentIdPath = Annotated[str, _path("The document id (doc_...) an upload returned.")]
FeedbackIdPath = Annotated[str, _path("The feedback record's id (fb_...).")]
EntityIdPath = Annotated[str, _path("The graph entity's id (ent_...).")]
KeyIdPath = Annotated[
    str, _path("The API key's id (the part of mk_<key_id>.<secret> before the dot).")
]
WorkspaceIdPath = Annotated[str, _path(f"The workspace (team). {ID_RULE}")]
PrincipalRefPath = Annotated[str, _path("The member: user:<id> or agent:<id>.")]
TenantIdPath = Annotated[str, _path(f"The tenant. {ID_RULE}")]
SuggestionIdPath = Annotated[
    str, _path("The approval suggestion's id, as GET /v1/tools/approval-suggestions lists it.")
]
SkillDraftIdPath = Annotated[
    str, _path("The skill draft's id (its procedure's), as GET /v1/tools/skill-drafts lists it.")
]
AgentToolNamePath = Annotated[
    str, _path("The memory tool to call, as GET /v1/agent-tools lists it (e.g. memory_search).")
]
ProfileBlockPath = Annotated[
    str,
    _path(
        "The block: user (the person the agent acts for), agent (this agent, for this user), "
        "workspace (the team), or one of them followed by .<name> (user.preferences)."
    ),
]
