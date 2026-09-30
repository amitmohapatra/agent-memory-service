"""Pinned profile blocks and durable thread summaries.

A profile block is text an agent always sees: ``user`` (the person it acts for), ``agent`` (the
agent itself, for this user), ``workspace`` (the team), or ``<level>.<name>`` beside them. The
block's level decides the scope it belongs to, from the caller's context, so a caller only
ever reads and writes the blocks of its own user, agent and workspace.

A block may carry a ``source_query``: a standing question the profile job answers from what
the block's scope may read, and keeps answering as that changes.

A thread summary is the durable, rolling digest of a thread up to ``covers_to_sequence``; the
context carries it and the messages after it.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from typing import Any, Final, Literal

from pydantic import BaseModel, ConfigDict, Field

from memory_service.domain.context import MemoryExecutionContext
from memory_service.domain.errors import ValidationFailed

#: A block's text is bounded: it is in every pushed context.
PROFILE_BLOCK_MAX_CHARS: Final = 4000
#: ``user``, ``agent``, ``workspace``, optionally ``.<name>``.
BLOCK_NAME = re.compile(r"^(user|agent|workspace)(\.[a-z0-9][a-z0-9_-]{0,39})?$")
#: The block the profile job maintains from USER and PREFERENCE memories.
USER_BLOCK: Final = "user"
SOURCE_QUERY_MAX_CHARS: Final = 1000

BlockSource = Literal["learned", "edited"]


class ProfileBlock(BaseModel):
    model_config = ConfigDict(frozen=True)

    tenant_id: str
    scope_key: str
    block: str
    text: str = Field(default="", max_length=PROFILE_BLOCK_MAX_CHARS)
    version: int = 0
    #: learned: written by the profile job; edited: last written by a person or an agent,
    #: whose text the job only ever appends to
    source: BlockSource = "learned"
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    #: the standing question the profile job answers into this block, when it has one
    source_query: str | None = Field(default=None, max_length=SOURCE_QUERY_MAX_CHARS)
    #: the context the question is answered in (who set it, so what it may read)
    source_context: dict[str, Any] | None = None
    #: when the job answers it next
    refresh_due_at: datetime | None = None


def block_scope(block: str, ctx: MemoryExecutionContext) -> str:
    """The scope a block of this name belongs to for this caller; refused when the name is
    not a block name or the caller's context has no such level."""
    match = BLOCK_NAME.match(block)
    if match is None:
        raise ValidationFailed(
            "a profile block is user, agent or workspace, optionally .<name>",
            details={"block": block},
        )
    scope = profile_scopes(ctx).get(match.group(1))
    if scope is None:
        raise ValidationFailed(
            f"a {match.group(1)} block needs a {match.group(1)} in the scope",
            details={"block": block},
        )
    return scope


def profile_scopes(ctx: MemoryExecutionContext) -> dict[str, str]:
    """level -> scope key, for the levels this caller has. The agent's block is bound to the
    principal (the agent acting for this user), never to the bare, unauthenticated agent id."""
    scopes: dict[str, str] = {}
    if ctx.user_id:
        scopes["user"] = f"user:{ctx.user_id}"
    if ctx.agent_id:
        scopes["agent"] = ctx.principal_id
    if ctx.workspace_id:
        scopes["workspace"] = f"workspace:{ctx.workspace_id}"
    return scopes


class ThreadSummary(BaseModel):
    model_config = ConfigDict(frozen=True)

    tenant_id: str
    thread_id: str
    version: int
    text: str
    covers_to_sequence: int
    #: the model that wrote it, or ``extractive`` without one
    model: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
