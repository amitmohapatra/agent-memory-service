"""Tool memory contracts (TOOL_MEMORY.md §30.0-§30.1).

The service never executes a tool. It keeps a *descriptor* per tool so it can record,
cache and reason about calls, one :class:`ToolInvocation` per call, and a
:class:`RunOutcome` per agent run. Everything derived from these — chains, procedures,
suggestions — is rebuilt from the invocation records, so the records are the only thing
that must be durable and exactly-once.

Policy defaults are deliberately conservative: an unregistered or per-call-declared tool is
non-deterministic, not cacheable and of unknown side effects, so nothing is ever replayed
from cache until an explicit registration widens the policy.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from memory_service.domain.ids import new_id

ToolSource = Literal["bifrost-mcp", "langgraph", "adk", "crewai", "mcp", "manual"]
ToolStatus = Literal["ok", "error", "timeout", "rejected"]
SideEffects = Literal["none", "read", "write", "external", "unknown"]
CacheScope = Literal["run", "thread", "user", "tenant"]


def stable_hash(value: Any) -> str:
    """SHA-256 over a canonical JSON form: same arguments, same hash, any key order."""
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


class ToolPolicy(BaseModel):
    """How the service may treat a tool. Widened only by an explicit registration."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    deterministic: bool = False
    side_effects: SideEffects = "unknown"
    cacheable: bool = False
    cache_ttl_seconds: int = Field(default=300, ge=0)
    cache_scope: CacheScope = "run"
    cost_hint: float | None = Field(default=None, ge=0.0)
    redact: list[str] = Field(
        default_factory=list,
        description="Dotted argument paths whose values never reach storage (e.g. 'auth.token').",
    )

    @property
    def replayable(self) -> bool:
        """A cached result may be served only for a deterministic, cacheable, side-effect-free
        tool: replaying anything else would hide a real call the agent must make."""
        return self.deterministic and self.cacheable and self.side_effects in ("none", "read")


class ToolDescriptor(BaseModel):
    """A tool the service knows about. Identity is (tenant, name, schema_hash)."""

    model_config = ConfigDict(extra="forbid")

    tool_id: str = Field(default_factory=lambda: new_id("tool"))
    tenant_id: str
    workspace_id: str | None = None
    name: str = Field(..., min_length=1, max_length=200)
    version: int = Field(default=1, ge=1)
    description: str = ""
    input_schema: dict[str, Any] | None = None
    output_schema: dict[str, Any] | None = None
    tags: list[str] = Field(default_factory=list)
    source: ToolSource = "manual"
    server: str | None = Field(default=None, description="MCP server name for gateway tools.")
    policy: ToolPolicy = ToolPolicy()
    schema_hash: str = ""
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @field_validator("name")
    @classmethod
    def _strip(cls, value: str) -> str:
        return value.strip()

    def with_schema_hash(self) -> ToolDescriptor:
        digest = stable_hash(
            {"input": self.input_schema, "output": self.output_schema, "name": self.name}
        )
        return self.model_copy(update={"schema_hash": digest})

    def redacted_args(self, args: dict[str, Any]) -> dict[str, Any]:
        """Drop every configured path before the arguments are persisted or hashed for storage."""
        if not self.policy.redact:
            return args
        out = json.loads(json.dumps(args, default=str))
        for path in self.policy.redact:
            node: Any = out
            parts = path.split(".")
            for part in parts[:-1]:
                if not isinstance(node, dict) or part not in node:
                    node = None
                    break
                node = node[part]
            if isinstance(node, dict) and parts[-1] in node:
                node[parts[-1]] = "[redacted]"
        return out


class SubCall(BaseModel):
    """One ``server.tool(...)`` call parsed out of a code-mode script, in script order."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    ordinal: int = Field(..., ge=0)
    tool: str
    args: dict[str, Any] = Field(default_factory=dict)
    bindings: dict[str, str] = Field(
        default_factory=dict,
        description="argument path -> expression it was bound from (e.g. 'id': 'step0.result.id')",
    )


class ToolInvocation(BaseModel):
    """One recorded tool call. Idempotent on (run_id, step, tool_id, args_hash)."""

    model_config = ConfigDict(extra="forbid")

    invocation_id: str = Field(default_factory=lambda: new_id("tool_invocation"))
    tenant_id: str
    tool_id: str
    tool_name: str
    tool_version: int = 1
    run_id: str | None = None
    thread_id: str | None = None
    turn_id: str | None = None
    workspace_id: str | None = None
    user_id: str | None = None
    agent_id: str | None = None
    principal_id: str | None = None
    step: int = Field(default=0, ge=0)
    args_redacted: dict[str, Any] = Field(default_factory=dict)
    args_hash: str = ""
    output_summary: str = ""
    output_digest: str | None = None
    output_blob_ref: str | None = Field(
        default=None, description="Archive reference when the output exceeded the inline limit."
    )
    output_fields: dict[str, Any] = Field(
        default_factory=dict,
        description="Flattened scalar output fields (path -> value) used to mine data flow.",
    )
    status: ToolStatus = "ok"
    error_class: str | None = None
    latency_ms: float | None = Field(default=None, ge=0.0)
    cost: float | None = Field(default=None, ge=0.0)
    task: str = ""
    task_pattern: str | None = None
    sub_calls: list[SubCall] = Field(default_factory=list)
    visibility_keys: list[str] = Field(default_factory=list)
    occurred_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    def idempotency_key(self) -> str:
        return stable_hash(
            {
                "run": self.run_id or "",
                "step": self.step,
                "tool": self.tool_id,
                "args": self.args_hash,
            }
        )

    @property
    def succeeded(self) -> bool:
        return self.status == "ok"


class RunOutcome(BaseModel):
    """Whether an agent run achieved its task. Only successful runs validate a procedure."""

    model_config = ConfigDict(extra="forbid")

    tenant_id: str
    run_id: str
    success: bool
    note: str | None = None
    source: Literal["explicit", "evidence", "window"] = "explicit"
    recorded_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ToolOutcomeStats(BaseModel):
    """Aggregated behaviour of one tool, reported by ``GET /v1/tools``."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    tool_name: str
    invocations: int = 0
    successes: int = 0
    failures: int = 0
    median_latency_ms: float | None = None
    total_cost: float = 0.0
    last_used_at: datetime | None = None

    @property
    def success_rate(self) -> float:
        return self.successes / self.invocations if self.invocations else 0.0
