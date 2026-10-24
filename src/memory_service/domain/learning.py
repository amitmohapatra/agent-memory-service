"""What the service learns from use: how a memory's standing moves its rank, the shape of a
tool call's arguments an approval is about, and when approvals support a rule.

Pure functions and the constants that tune them; the projectors and the ranking call these.
"""

from __future__ import annotations

import math
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict

# --------------------------------------------------------------------------- ranking

#: The confidence a memory is written with when nothing has confirmed or doubted it yet; the
#: ranking factor is neutral here.
NEUTRAL_CONFIDENCE: Final = 0.5
#: How far one unit of confidence above or below neutral moves the fused score.
CONFIDENCE_WEIGHT: Final = 0.3
#: How much each doubling of reinforcement moves the fused score.
REINFORCEMENT_WEIGHT: Final = 0.03
#: The factor never moves a score by more than this either way: standing reorders near-ties,
#: it never lifts an irrelevant memory over a relevant one.
MAX_STANDING_SHIFT: Final = 0.15


def standing_factor(confidence: float | None, reinforcement: int | None) -> float:
    """The bounded multiplier a memory's standing puts on its fused retrieval score."""
    shift = CONFIDENCE_WEIGHT * ((confidence or NEUTRAL_CONFIDENCE) - NEUTRAL_CONFIDENCE)
    shift += REINFORCEMENT_WEIGHT * math.log2(max(1, reinforcement or 1))
    return 1.0 + max(-MAX_STANDING_SHIFT, min(MAX_STANDING_SHIFT, shift))


# --------------------------------------------------------------------------- feedback

#: How much one verdict on an answer moves the confidence of each memory it cited.
CITED_CONFIDENCE_STEP: Final = 0.05
#: The lowest confidence a verdict can push a memory to (retraction is a separate verdict).
CONFIDENCE_FLOOR: Final = 0.05
#: Memories one answer verdict may adjust (the evidence references it names, in order).
CITED_MEMORIES_MAX: Final = 20


# --------------------------------------------------------------------------- approvals

#: Decisions on one (tool, argument shape) before a rule is suggested.
APPROVAL_MIN_SUPPORT: Final = 5
#: Approval rate at or above which "approve automatically" is suggested.
AUTO_APPROVE_RATE: Final = 0.95
#: Approval rate at or below which "always ask" is suggested.
ALWAYS_ASK_RATE: Final = 0.5
#: Suggestions one listing returns.
APPROVAL_SUGGESTIONS_MAX: Final = 100

Suggestion = Literal["auto_approve", "always_ask"]


class ApprovalCounts(BaseModel):
    """Decisions on one (agent, tool, argument shape)."""

    model_config = ConfigDict(frozen=True)

    agent_id: str
    tool: str
    arg_shape: str
    approvals: int = 0
    rejections: int = 0
    edits: int = 0

    @property
    def support(self) -> int:
        return self.approvals + self.rejections + self.edits

    @property
    def approve_rate(self) -> float:
        return self.approvals / self.support if self.support else 0.0

    def suggestion(self) -> Suggestion | None:
        """The rule these decisions support, if any: never applied by the service."""
        if self.support < APPROVAL_MIN_SUPPORT:
            return None
        if self.approve_rate >= AUTO_APPROVE_RATE:
            return "auto_approve"
        if self.approve_rate <= ALWAYS_ASK_RATE:
            return "always_ask"
        return None
