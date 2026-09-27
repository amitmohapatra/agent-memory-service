"""Standing questions and knowledge pages share one evidence-backed refresh lifecycle."""

from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from memory_service.domain.context import MemoryExecutionContext
from memory_service.domain.context_bundle import ContextItem
from memory_service.domain.ids import new_id, stable_key


class BriefSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["mental_model", "knowledge_page"] = Field(
        default="mental_model",
        description=(
            "mental_model maintains a standing answer; knowledge_page maintains a titled "
            "evidence page. Both share refresh and authorization rules."
        ),
    )
    title: str = Field(min_length=1, max_length=200)
    question: str = Field(min_length=1, max_length=4000)
    use_llm: bool = False
    refresh_seconds: int = Field(default=3600, ge=60, le=86400)


class BriefInfo(BaseModel):
    brief_id: str
    spec: BriefSpec


class BriefOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    text: str
    sources: list[ContextItem] = Field(default_factory=list)
    generated: bool = False
    generation_profile: str | None = Field(
        default=None,
        description="Opaque model/prompt fingerprint; absent for extractive or legacy output.",
    )
    revision_fingerprint: str
    valid_until: datetime
    built_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class StoredBrief(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    brief_id: str = Field(default_factory=lambda: new_id("brief"))
    context: MemoryExecutionContext
    spec: BriefSpec
    generation: int = 1
    output: BriefOutput | None = None
    next_refresh_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


def brief_scope(ctx: MemoryExecutionContext) -> str:
    """Exact source audience, including thread/run boundaries; not merely owner identity."""
    return stable_key(ctx.scope_fingerprint(), ctx.thread_id or "", ctx.session_id or "")
