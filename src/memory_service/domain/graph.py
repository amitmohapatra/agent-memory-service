"""Knowledge-graph layer taxonomy.

Every relation belongs to exactly one layer, assigned deterministically from its predicate:

- ``causal``     — why something happened (driven_by, caused_by, offset_by, ...)
- ``temporal``   — when a fact held or which fact replaced which (closed_on, founded_in,
                   supersedes, invalidated_by, ...)
- ``structural`` — where a thing appears in the corpus (mentions, co_occurs_with, defined_in)
- ``entity``     — everything else: typed facts between entities and values

The retrieval stage walks the layers in a query-dependent order ("why" questions read the
causal layer first, "when" questions the temporal one) and ``/v1/graph/query`` can restrict
a traversal to a subset of layers.
"""

from __future__ import annotations

from typing import Literal, get_args

GraphLayer = Literal["entity", "temporal", "causal", "structural"]
LAYERS: tuple[GraphLayer, ...] = get_args(GraphLayer)

INVALIDATED_BY = "invalidated_by"

_CAUSAL = frozenset(
    """
    driven_by caused_by causes leads_to results_in due_to reflects offset_by impacted_by
    attributable_to supported_by helped_by hurt_by enables contributed_to triggered_by
    depends_on because_of
    """.split()
)
_TEMPORAL = frozenset(
    """
    valid_from valid_to as_of supersedes superseded_by invalidated_by closed_on founded_in
    adopted_in occurred_on announced_on effective_from started_on ended_on dated
    scheduled_for expires_on preceded_by followed_by
    """.split()
)
_STRUCTURAL = frozenset(
    """
    mentions mentioned_in co_occurs_with discusses defined_in refers_to appears_in cites
    contained_in links
    """.split()
)


def layer_for(predicate: str) -> GraphLayer:
    p = predicate.strip().casefold()
    if p in _CAUSAL:
        return "causal"
    if p in _TEMPORAL:
        return "temporal"
    if p in _STRUCTURAL:
        return "structural"
    return "entity"


def layer_order(query: str) -> tuple[GraphLayer, ...]:
    """Layer priority for a question: causal first for why/how, temporal first for
    when/since/before/after/until/as-of, entity first otherwise; structural is always last."""
    q = f" {query.casefold()} "
    if any(cue in q for cue in (" why ", " how ", " because ", " reason", " driver", " cause")):
        return ("causal", "entity", "temporal", "structural")
    if any(
        cue in q
        for cue in (" when ", " since ", " before ", " after ", " until ", " as of ", " date")
    ):
        return ("temporal", "entity", "causal", "structural")
    return ("entity", "causal", "temporal", "structural")


def layer_weights(query: str) -> dict[str, float]:
    order = layer_order(query)
    return {layer: 1.0 - 0.2 * i for i, layer in enumerate(order)}
