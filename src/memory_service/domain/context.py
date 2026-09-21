"""Dynamic execution context.

``MemoryClient`` is static; ``MemoryExecutionContext`` is per request. It carries the
security scope (tenant/workspace/user/groups), the conversation lineage
(thread/session/turn), the multi-agent lineage (agent/run/parent run) and the
correlation identifiers used by tracing and lineage.

Security fields are *never* overridable by ``custom_metadata``: the model rejects
custom metadata keys that collide with reserved names.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from memory_service.domain.ids import is_valid_id, new_id

SECURITY_FIELDS: frozenset[str] = frozenset(
    {"tenant_id", "workspace_id", "user_id", "group_ids", "principal_id", "principal_type"}
)
LINEAGE_FIELDS: frozenset[str] = frozenset(
    {
        "thread_id",
        "session_id",
        "turn_id",
        "work_id",
        "task_id",
        "agent_id",
        "agent_group_id",
        "agent_run_id",
        "parent_agent_run_id",
    }
)
CORRELATION_FIELDS: frozenset[str] = frozenset(
    {"request_id", "correlation_id", "causation_id", "trace_id"}
)
RESERVED_METADATA_KEYS: frozenset[str] = SECURITY_FIELDS | LINEAGE_FIELDS | CORRELATION_FIELDS


def _validate_optional_id(value: str | None) -> str | None:
    if value is None:
        return None
    if not is_valid_id(value):
        raise ValueError(f"invalid identifier: {value!r}")
    return value


class MemoryExecutionContext(BaseModel):
    """Per-request identity, scope and lineage. Immutable once built."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    # --- security scope -------------------------------------------------
    tenant_id: str = Field(..., description="Tenant boundary. Mandatory; never inferred.")
    workspace_id: str | None = Field(default=None, description="Workspace inside the tenant.")
    user_id: str | None = Field(default=None, description="End user on whose behalf we act.")
    group_ids: list[str] = Field(default_factory=list, description="User group memberships.")

    # --- conversation lineage ------------------------------------------
    thread_id: str | None = Field(
        default=None, description="Conversation thread (ChatGPT-style chat)."
    )
    session_id: str | None = Field(default=None, description="Open UI session within the thread.")
    turn_id: str | None = Field(default=None, description="One user turn (question + answer).")

    # --- work lineage ---------------------------------------------------
    work_id: str | None = Field(
        default=None, description="Unit of work spanning several agents/turns."
    )
    task_id: str | None = Field(default=None, description="Task inside a unit of work.")

    # --- multi-agent lineage --------------------------------------------
    agent_id: str | None = Field(
        default=None, description="Logical agent identity (e.g. 'research')."
    )
    agent_group_id: str | None = Field(default=None, description="Cooperating agent group.")
    agent_run_id: str | None = Field(default=None, description="This agent execution.")
    parent_agent_run_id: str | None = Field(default=None, description="Run that spawned this one.")

    # --- correlation ----------------------------------------------------
    request_id: str = Field(default_factory=lambda: new_id("request"))
    correlation_id: str = Field(default_factory=lambda: new_id("request"))
    causation_id: str | None = None
    trace_id: str = Field(default_factory=lambda: new_id("request"))

    # --- free-form ------------------------------------------------------
    custom_metadata: dict[str, Any] = Field(default_factory=dict)

    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    # ------------------------------------------------------------------
    @field_validator(
        "tenant_id",
        "workspace_id",
        "user_id",
        "thread_id",
        "session_id",
        "turn_id",
        "work_id",
        "task_id",
        "agent_id",
        "agent_group_id",
        "agent_run_id",
        "parent_agent_run_id",
        "request_id",
        "correlation_id",
        "causation_id",
        "trace_id",
    )
    @classmethod
    def _ids_are_valid(cls, value: str | None) -> str | None:
        return _validate_optional_id(value)

    @field_validator("group_ids")
    @classmethod
    def _groups_are_valid(cls, value: list[str]) -> list[str]:
        for gid in value:
            if not is_valid_id(gid):
                raise ValueError(f"invalid group id: {gid!r}")
        # deterministic order => stable scope hashes
        return sorted(set(value))

    @model_validator(mode="after")
    def _metadata_cannot_override_security(self) -> Self:
        clash = RESERVED_METADATA_KEYS.intersection(self.custom_metadata)
        if clash:
            raise ValueError(f"custom_metadata may not contain reserved keys: {sorted(clash)}")
        if self.agent_run_id is not None and self.agent_id is None:
            raise ValueError("agent_run_id requires agent_id")
        if self.session_id is not None and self.thread_id is None:
            raise ValueError("session_id requires thread_id")
        if self.turn_id is not None and self.session_id is None:
            raise ValueError("turn_id requires session_id")
        return self

    # ------------------------------------------------------------------
    @property
    def principal_id(self) -> str:
        """The acting principal: an agent run if present, else the user, else the service.

        An agent principal is **bound to the user it runs for**, because ``agent_id`` arrives
        in the request body and is not authenticated — only ``tenant_id``, ``user_id`` and
        ``workspace_id`` come from trusted headers (api/deps.py:85-104). A bare
        ``agent:{agent_id}`` therefore let any caller assume any agent's identity simply by
        naming it, and read everything that agent had marked PRIVATE.

        Reproduced against a live service before this was changed: user ``mallory`` sending
        ``agent_id=worker`` read a PRIVATE memory owned by ``agent:worker`` and written for a
        different user — HTTP 200. Without the agent_id the same request was correctly 403.

        Two users each running an agent called "research" are two principals, which is also
        the behaviour anyone would expect. An agent running with no user at all — an
        ingestion job, an unattended scheduled run — keeps the bare form, since there is no
        user to bind it to and nothing for a caller to impersonate their way into.
        """
        if self.agent_id is not None:
            if self.user_id is not None:
                return f"agent:{self.user_id}/{self.agent_id}"
            return f"agent:{self.agent_id}"
        if self.user_id is not None:
            return f"user:{self.user_id}"
        return "service:anonymous"

    @property
    def is_agent(self) -> bool:
        return self.agent_id is not None

    def child_agent(
        self,
        *,
        agent_id: str,
        agent_run_id: str | None = None,
        agent_group_id: str | None = None,
        task_id: str | None = None,
        custom_metadata: dict[str, Any] | None = None,
    ) -> MemoryExecutionContext:
        """Derive a context for an internal agent run.

        Inherits tenant/workspace/user/thread/session/turn/trace, adds agent lineage.
        """
        return self.model_copy(
            update={
                "agent_id": agent_id,
                "agent_group_id": agent_group_id or self.agent_group_id,
                "agent_run_id": agent_run_id or new_id("agent_run"),
                "parent_agent_run_id": self.agent_run_id,
                "task_id": task_id or self.task_id,
                "request_id": new_id("request"),
                "causation_id": self.request_id,
                "custom_metadata": {**self.custom_metadata, **(custom_metadata or {})},
            }
        )

    def scope_fingerprint(self) -> str:
        """Stable digest of the security scope; used in every sensitive cache key."""
        from memory_service.domain.ids import stable_key

        return stable_key(
            self.tenant_id,
            self.workspace_id or "",
            self.user_id or "",
            ",".join(self.group_ids),
            self.agent_id or "",
            self.agent_group_id or "",
            self.agent_run_id or "",
            self.parent_agent_run_id or "",
        )

    def log_fields(self) -> dict[str, str]:
        """Fields attached to every log line / span for this request."""
        out: dict[str, str] = {"tenant_id": self.tenant_id, "trace_id": self.trace_id}
        for name in ("thread_id", "session_id", "turn_id", "agent_run_id", "request_id"):
            value = getattr(self, name)
            if value:
                out[name] = value
        return out
