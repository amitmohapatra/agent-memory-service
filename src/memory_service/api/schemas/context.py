"""Public response models for context, tool choices and grounding.

The context models describe what ``modules/context/views.py`` sends: only what a caller uses,
nothing twice, every number in 0..1, and an optional field absent rather than null or empty.
They are separate from the domain models on purpose: the domain is free to grow, while these
are the contract the SDK relies on, so nothing extra is allowed through.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from memory_service.domain.enums import EvidenceStatus, MessageRole
from memory_service.domain.grounding import ClaimVerdict, GroundingMethod

_CLAIM_EXAMPLE: dict[str, Any] = {
    "claim": "Adjusted EBITDA increased to EUR 98 million",
    "verdict": "supported",
    "support": 0.97,
    "contradiction": 0.01,
    "evidence_ids": ["chk_01J8ZK7Q9V3W2X1Y0ZABCDEFGH"],
    "contradicted_by": [],
    "citations": ["1"],
    "method": "nli",
    "notes": [],
}


class ClaimVerdictBody(BaseModel):
    model_config = ConfigDict(extra="forbid", json_schema_extra={"examples": [_CLAIM_EXAMPLE]})

    claim: str = Field(description="One checkable statement split from the answer.")
    verdict: ClaimVerdict = Field(
        ...,
        description="supported: every fact in the claim follows from the evidence; "
        "unsupported: the evidence does not say it; contradicted: retrieved evidence says "
        "otherwise; borderline: the NLI could not decide and no judge settled it.",
    )
    support: float = Field(default=0.0, ge=0.0, le=1.0, description="best entailment score")
    contradiction: float = Field(default=0.0, ge=0.0, le=1.0, description="best contradiction")
    evidence_ids: list[str] = Field(default_factory=list, description="items the verdict rests on")
    contradicted_by: list[str] = Field(
        default_factory=list, description="retrieved-but-unused items that contradict the claim"
    )
    citations: list[str] = Field(default_factory=list, description="citation markers in the claim")
    method: GroundingMethod = Field(
        default="nli",
        description="The cheapest stage that decided the claim: citation (the cited item "
        "settles it), nli (cross-encoder entailment) or judge (LLM, borderline claims only).",
    )
    notes: list[str] = Field(
        default_factory=list,
        description="What the cascade noticed deciding it (e.g. a number that differs).",
    )


class GroundingReportBody(BaseModel):
    """GroundingReport: one verdict per claim and the per-claim hallucination rate."""

    model_config = ConfigDict(extra="forbid")

    claims: list[ClaimVerdictBody] = Field(
        default_factory=list, description="One verdict per claim, in answer order."
    )
    supported: int = Field(default=0, description="How many claims were supported.")
    unsupported: int = Field(
        default=0, description="How many claims the evidence does not support."
    )
    contradicted: int = Field(default=0, description="How many claims the evidence contradicts.")
    borderline: int = Field(default=0, description="How many claims nothing could decide.")
    per_claim_hallucination_rate: float = Field(
        default=0.0, ge=0.0, le=1.0, description="(unsupported + contradicted) / claims"
    )
    nli_provider: str = Field(
        default="", description="The entailment model that judged (its id and weights digest)."
    )
    representative: bool = Field(
        default=False, description="False when the NLI is a deterministic stand-in"
    )
    judge_consulted: int = Field(default=0, description="borderline claims sent to the LLM judge")
    llm_tokens: int = Field(default=0, description="LLM tokens spent on this report")
    evidence_count: int = Field(
        default=0, description="How many evidence items the answer was checked against."
    )
    unused_count: int = Field(
        default=0,
        description="How many retrieved items the answer was not given but that were "
        "checked for contradiction.",
    )
    notes: list[str] = Field(
        default_factory=list, description="What the check noticed about the answer as a whole."
    )


# --------------------------------------------------------------------------- tools

_RELEVANCE = "0..1, how close the item is to the question; comparable across the bundle"
_CONFIDENCE = "0..1, how well the tool fits the task (1 - e^-score: the order of the score)"


class ToolChoiceBrief(BaseModel):
    """A tool that fits the task, as the prompt form carries it: enough to narrow on."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(description="The tool's name, as the catalog and the agent call it.")
    confidence: float = Field(..., ge=0.0, le=1.0, description=_CONFIDENCE)


class MissingArgumentBody(BaseModel):
    """A required argument nothing could fill: ask the user ``question``."""

    model_config = ConfigDict(extra="forbid")

    arg: str = Field(description="The argument's name.")
    question: str = Field(description="A question the agent can ask the user to fill it.")
    entity_type: str | None = Field(default=None, description="the entity type it names")


class ToolChoiceBody(BaseModel):
    """A tool that fits the task: how well, how it has done, and its arguments."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(description="The tool's name, as the catalog and the agent call it.")
    confidence: float = Field(..., ge=0.0, le=1.0, description=_CONFIDENCE)
    success_rate: float | None = Field(
        default=None, ge=0.0, le=1.0, description="share of its recorded calls that succeeded"
    )
    next: bool | None = Field(default=None, description="true for the learned plan's next step")
    args: dict[str, Any] | None = Field(
        default=None, description="argument values already found, typed as the schema says"
    )
    missing: list[MissingArgumentBody] | None = Field(
        default=None, description="required arguments nothing found"
    )


class ProcedureBody(BaseModel):
    """A procedure learned for the task: its tool sequence and how often it worked."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(description="The procedure's id (prc_...).")
    title: str | None = Field(
        default=None, description="What the procedure does, in words, when one was distilled."
    )
    steps: list[str] = Field(description="tool names, in order")
    success_rate: float = Field(
        ..., ge=0.0, le=1.0, description="0..1, the share of runs that followed it and succeeded."
    )
    runs: int = Field(description="the successful runs it was learned from")


class ToolHintsResponse(BaseModel):
    """Which tools fit a task, best first, and the learned plan."""

    model_config = ConfigDict(extra="forbid")

    tools: list[ToolChoiceBody] = Field(description="The tools that fit, best first.")
    plan: ProcedureBody | None = Field(
        default=None, description="The procedure learned for this kind of task, when there is one."
    )


# --------------------------------------------------------------------------- context

_EVIDENCE_STATUS = (
    "COMPLETE, INCOMPLETE (evidence for part of the question, or about someone else) or "
    "INSUFFICIENT (nothing to go on: answer that you do not know rather than guess)"
)


class DatedMentionBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(description='the words, e.g. "last week"')
    date: str = Field(description="the date or range they mean, e.g. 2023-05-01..2023-05-07")


class ContextMemory(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(
        description="The memory's handle in this context (m1, m2 ...); resolves with bundle_id."
    )
    text: str = Field(description="The memory, as the model reads it.")
    relevance: float = Field(..., ge=0.0, le=1.0, description=_RELEVANCE)
    observed_at: str | None = Field(default=None, description="when it was said or learned")
    subject: str | None = Field(default=None, description="who or what it is about")
    dates: list[DatedMentionBody] | None = Field(
        default=None, description="relative dates in the text, resolved"
    )
    sources: list[str] | None = Field(default=None, description="the messages or documents")


class ContextPassage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(
        description="The passage's handle in this context (d1, d2 ...); resolves with bundle_id."
    )
    text: str = Field(description="The passage text.")
    relevance: float = Field(..., ge=0.0, le=1.0, description=_RELEVANCE)
    kind: Literal["relation", "memory"] | None = Field(
        default=None,
        description="Absent for a document passage (a chunk); relation or memory for a fact or "
        "memory carried in as a passage's required companion.",
    )
    document_id: str | None = Field(
        default=None, description="The document (doc_...) it comes from."
    )
    page: int | None = Field(
        default=None,
        description="The 1-based page of the document it is on, when the document has pages.",
    )
    section: str | None = Field(default=None, description="e.g. 'Financial Results > EBITDA'")


class ContextFact(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(
        description="The fact's handle in this context (f1, f2 ...); resolves with bundle_id."
    )
    subject: str = Field(description="The entity the fact is about.")
    predicate: str = Field(description="The relation, e.g. ceo or reported.")
    object: str = Field(description="The entity or value it relates to.")
    relevance: float = Field(..., ge=0.0, le=1.0, description=_RELEVANCE)
    observed_at: str | None = Field(
        default=None, description="When the service learned it (YYYY-MM-DD)."
    )
    valid_from: str | None = Field(
        default=None, description="When it became true (YYYY-MM-DD), when known."
    )
    valid_to: str | None = Field(
        default=None, description="When it stopped being true (YYYY-MM-DD), when known."
    )
    document_id: str | None = Field(
        default=None, description="The document it was extracted from, if any."
    )


class ContextSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(description="The summary's handle in this context (s1, s2 ...).")
    text: str = Field(description="The summary text.")
    relevance: float = Field(..., ge=0.0, le=1.0, description=_RELEVANCE)


class WindowMessageBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(description="The message's id (msg_...).")
    role: MessageRole = Field(
        description="Who said it: USER, ASSISTANT, SYSTEM, TOOL, AGENT or EVENT (something "
        "that happened, told to the service)."
    )
    text: str = Field(description="What was said.")


class ConversationBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    thread_id: str | None = Field(default=None, description="The thread the window comes from.")
    messages: list[WindowMessageBody] = Field(
        description="The thread's most recent messages after its summary, oldest first."
    )


class ProfileBlockBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    block: str = Field(description="user, agent, workspace or <level>.<name>")
    text: str = Field(description="The block's text.")


class PromptContextResponse(BaseModel):
    """format=prompt: what an agent puts in front of its model."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "bundle_id": "6f1c0e2a9b",
                    "rendered": "## Tools\n- erp-create_po (confidence 0.74, next step): "
                    "amount = 700, cost_centre = 'CC-7'; missing supplier_id: Which supplier "
                    "id should erp-create_po use?\n\n## Memories\n- [m1] 2026-10-04 Sun "
                    "priya: We buy steel from Acme Steel.",
                    "token_estimate": 180,
                    "evidence_status": "COMPLETE",
                    "tools": [{"name": "erp-create_po", "confidence": 0.74}],
                }
            ]
        },
    )

    bundle_id: str = Field(description="for /v1/verify and for resolving the handles")
    rendered: str = Field(description="prompt-ready; items are cited by handle ([m1], [d2]...)")
    token_estimate: int = Field(
        description="Approximate tokens of the rendered text (what it costs in the prompt)."
    )
    evidence_status: EvidenceStatus = Field(..., description=_EVIDENCE_STATUS)
    tools: list[ToolChoiceBrief] | None = Field(
        default=None, description="the tools that fit, best first (only when tools were given)"
    )
    diagnostics: dict[str, Any] | None = Field(default=None, description="only with debug")


class ContextResponse(BaseModel):
    """format=full: the same content as structured data, for a caller building its own
    prompt. A list or field with nothing in it is absent."""

    model_config = ConfigDict(extra="forbid")

    bundle_id: str = Field(description="for /v1/verify and for resolving the handles")
    evidence_status: EvidenceStatus = Field(..., description=_EVIDENCE_STATUS)
    token_estimate: int = Field(
        description="Approximate tokens of the content (what it costs in a prompt)."
    )
    missing_evidence: list[str] | None = Field(
        default=None, description="required companion evidence that is not there"
    )
    conversation: ConversationBody | None = Field(
        default=None, description="the thread's recent messages (with window)"
    )
    thread_summary: str | None = Field(default=None, description="the thread's durable summary")
    profile: list[ProfileBlockBody] | None = Field(
        default=None, description="The pinned profile blocks (user, agent, workspace)."
    )
    procedures: list[ProcedureBody] | None = Field(
        default=None, description="procedures learned for the task (only with tools)"
    )
    tools: list[ToolChoiceBody] | None = Field(
        default=None, description="the tools that fit, best first (only with tools)"
    )
    memories: list[ContextMemory] | None = Field(
        default=None, description="What is remembered that bears on the question, best first."
    )
    knowledge: list[ContextPassage] | None = Field(
        default=None, description="Document passages that bear on it, best first."
    )
    graph_facts: list[ContextFact] | None = Field(
        default=None, description="Knowledge-graph facts that bear on it."
    )
    summaries: list[ContextSummary] | None = Field(
        default=None, description="Document summaries that bear on it."
    )
    diagnostics: dict[str, Any] | None = Field(default=None, description="only with debug")
