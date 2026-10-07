"""Tool memory contracts (design/TOOL_MEMORY.md §30.0-§30.1).

The service never executes a tool. The catalog keeps a :class:`ToolDescriptor` per tool
(what it is, which arguments name which kinds of entity, and what calling it does), one
:class:`ToolInvocation` per recorded call, and a :class:`RunOutcome` per agent run.
Everything derived from these — statistics, procedures, hints — is rebuilt from the records,
so the records are the only thing that must be durable and exactly-once.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from typing import Any, Final, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from memory_service.domain.ids import new_id

ToolStatus = Literal["ok", "error", "timeout", "rejected", "cancelled"]
#: What calling a tool does: ``read`` changes nothing, ``write`` changes something that can
#: be changed back, ``irreversible`` cannot be undone (a payment, an email sent).
SideEffects = Literal["read", "write", "irreversible"]
#: Where a tool comes from (mcp, local, openapi, a2a, ...): descriptive, not a closed set.
SOURCE_MAX_CHARS: Final = 50
#: Where a catalog entry comes from: the contracts' ``ToolSource`` (what a harness publishes)
#: and ``manual``, an entry an administrator wrote through the API.
ToolSource = Literal["manual", "local", "mcp", "memory", "openapi", "a2a"]
TOOL_SOURCE_DESCRIPTION: Final = (
    "Where the tool comes from: manual (written through this API, the default), local (a "
    "function of the harness), mcp (an MCP server, named in server), memory (this service's "
    "own agent tools), openapi (an OpenAPI operation) or a2a (another agent)."
)


class ToolAnnotations(BaseModel):
    """The MCP tool annotations a gateway listing carries (hints, not guarantees)."""

    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)

    read_only: bool | None = Field(
        default=None,
        alias="readOnlyHint",
        description="MCP readOnlyHint: the tool changes nothing (risk read).",
    )
    destructive: bool | None = Field(
        default=None,
        alias="destructiveHint",
        description="MCP destructiveHint: a call may destroy or overwrite (risk irreversible).",
    )
    idempotent: bool | None = Field(
        default=None,
        alias="idempotentHint",
        description="MCP idempotentHint: repeating a call with the same arguments has no "
        "further effect.",
    )
    open_world: bool | None = Field(
        default=None,
        alias="openWorldHint",
        description="MCP openWorldHint: the tool reaches outside systems (the web, third parties).",
    )


def risk_tier(side_effects: SideEffects | None, annotations: ToolAnnotations) -> SideEffects:
    """What calling the tool risks: the catalog's ``side_effects`` when set, else the MCP
    annotations (``readOnlyHint`` -> read, ``destructiveHint`` -> irreversible), else write."""
    if side_effects is not None:
        return side_effects
    if annotations.read_only:
        return "read"
    if annotations.destructive:
        return "irreversible"
    return "write"


def stable_hash(value: Any) -> str:
    """SHA-256 over a canonical JSON form: same arguments, same hash, any key order."""
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


class ToolDescriptor(BaseModel):
    """One catalog entry. Identity is (tenant, workspace, name); a changed input schema is a
    new ``version`` of the same entry."""

    model_config = ConfigDict(extra="forbid")

    tool_id: str = Field(default_factory=lambda: new_id("tool"))
    tenant_id: str
    workspace_id: str | None = None
    name: str = Field(..., min_length=1, max_length=200)
    version: int = Field(default=1, ge=1)
    description: str = ""
    input_schema: dict[str, Any] | None = None
    required: list[str] = Field(default_factory=list)
    argument_entity_types: dict[str, str] = Field(
        default_factory=dict,
        description="argument name -> the entity type its value names (e.g. supplier: ORG)",
    )
    side_effects: SideEffects | None = None
    source: str = Field(
        default="manual", max_length=SOURCE_MAX_CHARS, description=TOOL_SOURCE_DESCRIPTION
    )
    server: str | None = Field(default=None, description="MCP server name for gateway tools.")
    examples: list[dict[str, Any]] = Field(default_factory=list)
    redact: list[str] = Field(
        default_factory=list,
        description="Dotted argument paths whose values never reach storage (e.g. 'auth.token').",
    )
    annotations: ToolAnnotations = Field(default_factory=ToolAnnotations)
    approve_when: str | None = Field(
        default=None,
        description="ask a person before a call when this expression over the arguments is "
        "true (trellis.memory.approval); None: the risk tier decides",
    )
    schema_hash: str = ""
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @field_validator("name")
    @classmethod
    def _strip(cls, value: str) -> str:
        return value.strip()

    def with_schema_hash(self) -> ToolDescriptor:
        return self.model_copy(
            update={"schema_hash": stable_hash({"input": self.input_schema, "name": self.name})}
        )

    def catalog_fields(self) -> dict[str, Any]:
        """What an upsert compares: an entry whose fields are unchanged is left as it is."""
        return self.model_dump(
            include={
                "description",
                "input_schema",
                "required",
                "argument_entity_types",
                "side_effects",
                "source",
                "server",
                "examples",
                "redact",
                "annotations",
                "approve_when",
            }
        )

    @property
    def risk(self) -> SideEffects:
        return risk_tier(self.side_effects, self.annotations)

    def index_text(self) -> str:
        """What the tool is searched by: its name, description and argument names."""
        fields = sorted(((self.input_schema or {}).get("properties") or {}).keys())
        return " ".join(filter(None, (self.name, self.description, " ".join(fields))))

    def redacted_args(self, args: dict[str, Any]) -> dict[str, Any]:
        """Drop every configured path before the arguments are persisted or hashed for storage."""
        if not self.redact:
            return args
        out = json.loads(json.dumps(args, default=str))
        for path in self.redact:
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
    #: the feedback source that decided it (``domain.feedback.OUTCOME_PRECEDENCE``)
    source: Literal["system", "judge", "interrupt", "human"] = "system"
    recorded_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ToolStats(BaseModel):
    """What the service has seen one tool do, kept per call and per feedback (O(1) each)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    tool_name: str
    calls: int = 0
    successes: int = 0
    latency_ms_total: float = 0.0
    latency_calls: int = 0
    approvals: int = 0
    rejections: int = 0
    edits: int = 0
    last_used_at: datetime | None = None

    @property
    def failures(self) -> int:
        return self.calls - self.successes

    @property
    def success_rate(self) -> float | None:
        return self.successes / self.calls if self.calls else None

    @property
    def avg_latency_ms(self) -> float | None:
        return self.latency_ms_total / self.latency_calls if self.latency_calls else None


#: candidate: not enough support yet; active: offered; retired: stopped working or aged
#: out; rejected: dismissed by an administrator, or rejected by a reviewer's verdict (kept
#: until its steps change)
ProcedureStatus = Literal["candidate", "active", "retired", "rejected"]

#: The audience key prefix of an agent's learned procedures (not a visibility key: who reads
#: them is decided by ``agent_id``, ``users`` and ``sole_user``).
AGENT_AUDIENCE_PREFIX: Final = "agent:"


def agent_audience(tenant_id: str, agent_id: str) -> str:
    """The audience every user's runs of one agent are learned under."""
    return f"{AGENT_AUDIENCE_PREFIX}{tenant_id}/{agent_id}"


class StoredProcedure(BaseModel):
    """A procedure the learning job keeps for one task pattern and one audience: the agent's
    learned skill for that kind of task.

    ``scope_key`` is the audience of the records it was mined from. An agent's own calls (its
    PRIVATE records, whichever user it ran for) are learned together, under
    ``agent_audience(tenant, agent)``: the agent learns from all its users. Such a procedure is
    read by the agent's users once at least two of them produced it (``users``), and before
    that only by the one who did (``sole_user``), so one person's wording of a task never
    reaches another. Records shared wider (a group, a workspace) keep their own audience key.
    Only an ``active`` procedure is offered; it is admitted when enough runs support it and
    enough of them succeeded, retired when it stops working, and rejected when dismissed."""

    model_config = ConfigDict(extra="forbid")

    procedure_id: str = Field(default_factory=lambda: new_id("procedure"))
    tenant_id: str
    scope_key: str
    pattern: str
    title: str = ""
    strategy: str = ""
    steps: list[dict[str, Any]] = Field(default_factory=list)
    bindings: list[dict[str, Any]] = Field(default_factory=list)
    success_rate: float = 0.0
    support: int = 0
    status: ProcedureStatus = "candidate"
    #: the fingerprint of the steps; the title and strategy were distilled from ``distilled``
    steps_hash: str = ""
    distilled: str = ""
    #: whose model key distils it: the principal that recorded the calls
    owner_principal: str | None = None
    workspace_id: str | None = None
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    #: the agent it was learned for (an agent-audience procedure), else None
    agent_id: str | None = None
    #: how many distinct users' runs it was mined from (runs with no user count as one)
    users: int = 0
    #: the one user who produced it while ``users`` is 1 (None: no user, or several)
    sole_user: str | None = None

    @property
    def tools(self) -> list[str]:
        return [str(step.get("tool")) for step in self.steps]


class ToolCandidate(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    #: the ranking score (unbounded: task match, track record, plan and recency)
    score: float
    #: the score as 0..1 (``1 - e^-score``): the same order, a number a caller can threshold
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    success_rate: float | None = None
    why: str = ""


class SkillView(BaseModel):
    """A learned skill as an agent is offered it: in the context and as ``tool_search``'s
    plan (``modules.tools.skills.skill_view``)."""

    model_config = ConfigDict(frozen=True)

    id: str
    name: str
    #: the tools it calls, in order (without the skill machinery)
    steps: list[str] = Field(default_factory=list)
    #: the agent's own skill its runs opened: what it adds to that skill
    with_skill: str | None = None
    #: what worked when a step failed: "tool on error: fix"
    fixes: list[str] = Field(default_factory=list)
    success_rate: float = 0.0
    support: int = 0


PrefillSource = Literal["procedure", "graph", "profile", "memory", "task"]


class Prefill(BaseModel):
    model_config = ConfigDict(frozen=True)

    tool: str
    value: Any = None
    source: PrefillSource = Field(
        description="where the value came from: procedure (a learned binding: an earlier "
        "step's output in this run, or the literal every successful run used), graph (an "
        "entity of the argument's type named in the task, or the id a tool returned for it), "
        "profile (a pinned profile line), memory (a memory whose predicate is the argument), "
        "task (a value the task names)"
    )
    evidence_id: str | None = Field(
        default=None, description="the call, entity, relation, block or memory it came from"
    )


class MissingArgument(BaseModel):
    model_config = ConfigDict(frozen=True)

    tool: str
    arg: str
    entity_type: str | None = None
    question: str


class ToolHints(BaseModel):
    """Which tools fit a task, the learned plan, the next step and its arguments."""

    model_config = ConfigDict(frozen=True)

    candidates: list[ToolCandidate] = Field(default_factory=list)
    plan: SkillView | None = None
    next: str | None = None
    prefill: dict[str, Prefill] = Field(
        default_factory=dict, description="argument values found, keyed tool.arg"
    )
    missing: list[MissingArgument] = Field(default_factory=list)
