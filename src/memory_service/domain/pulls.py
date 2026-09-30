"""Pulls: the memory an agent asked for through its tools, and what it then used.

Every agent-tool call is a pull: the request's pattern (the typed-placeholder form of what was
asked), the tool, its arguments, the ids it returned. A returned id is *used* when the run later
cites it (a verified answer's evidence, an answer verdict) or acts on it (updates or forgets
it). The prefetch job folds settled pulls into per (principal, pattern, item) counts; the pushed
context pre-includes the items that were used often enough for requests of the same pattern.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any, Final

from pydantic import BaseModel, ConfigDict, Field

from memory_service.domain.ids import new_id

#: A pull is folded into the counts once this old: long enough for the run to cite it.
PULL_SETTLE: Final = timedelta(minutes=10)
#: Pulls one prefetch run folds.
PREFETCH_BATCH: Final = 1000
#: An item is pre-included once pulled this often for the pattern ...
PREFETCH_MIN_PULLS: Final = 3
#: ... and used in at least this share of those pulls.
PREFETCH_MIN_RATE: Final = 0.5
#: Items one context pre-includes.
PREFETCH_MAX: Final = 5
#: Ids one pull keeps (what a search returns is bounded far below this).
PULL_RESULTS_MAX: Final = 50


class AgentPull(BaseModel):
    model_config = ConfigDict(frozen=True)

    pull_id: str = Field(default_factory=lambda: new_id("pull"))
    tenant_id: str
    #: the principal that pulled: counts are per principal, whose items only it may read
    scope_key: str
    run_id: str | None = None
    pattern: str = ""
    tool: str
    args: dict[str, Any] = Field(default_factory=dict)
    result_ids: list[str] = Field(default_factory=list)
    used_ids: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
