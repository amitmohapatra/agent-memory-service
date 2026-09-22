"""Deterministic QueryRouter. Rules first; a model is consulted only when configured and the
rules cannot decide (never in ``llm.enabled=false`` mode)."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from memory_service.domain.enums import QueryType

_ID = re.compile(
    r"\b(?:thr|ses|trn|msg|doc|chk|mem|obs|run|job|nod|sum_nod|rel|ent)_[0-9A-Za-z]{10,}\b|\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b|\b[A-Z]{2,6}-\d{2,7}\b"
)
_CONVERSATION = re.compile(
    r"\b(what did (?:i|you|we) (?:say|ask|mention)|earlier (?:in this|you)|previous(?:ly)? (?:said|message)|as (?:i|you) (?:said|mentioned)|in this (?:chat|conversation|thread)|scroll back|remind me what)\b",
    re.IGNORECASE,
)
_USER_MEMORY = re.compile(
    r"\b(my (?:preference|preferences|favorite|favourite|name|timezone|role|team|settings|setup|usual)|do i (?:like|prefer|use)|what do i|about me|i (?:usually|always|never)|remember (?:that )?i)\b",
    re.IGNORECASE,
)
_DECISION = re.compile(
    r"\b(why did we (?:decide|choose|pick|go with)|decision|decided|agreed|rationale|trade-?off|chose|why was .* (?:chosen|rejected))\b",
    re.IGNORECASE,
)
#: The bare interrogatives were missing, and they are how people actually ask about time.
#: Measured on LoCoMo, only 1 of 26 temporal questions routed TEMPORAL, because every one
#: of them opens "When did ..." and the pattern below had "since|before|after" but no "when".
_TEMPORAL = re.compile(
    r"\b(when (?:did|was|were|does|do|is|are|will)|what (?:year|month|date|day|time)|how long (?:ago|did|has|have)|what time|as of|since|before|after|until|between|last (?:week|month|quarter|year)|this (?:week|month|quarter|year)|in (?:19|20)\d{2}|q[1-4]\b|fy\d{2}|\d{4}-\d{2}-\d{2}|yesterday|today|latest|current(?:ly)?|previous|history of|changed over|over time|timeline)\b",
    re.IGNORECASE,
)
_GLOBAL = re.compile(
    r"\b(overall|summar(?:y|ise|ize)|across (?:the|this|all)|main (?:themes|risks|points)|major (?:risks|themes|changes)|key (?:takeaways|points|risks)|what is this (?:document|report|file) about|high[- ]level|big picture|tl;?dr)\b",
    re.IGNORECASE,
)
#: Contrastive connectives only ("despite", "compare", "difference between") miss the other
#: shape of a multi-fact question: enumeration. "What kind of music does X like?" and "Where
#: has X travelled?" need every mention across sessions gathered, which is the same
#: traversal. Measured on LoCoMo, 1 of 16 multi_hop questions routed DOCUMENT_MULTI_HOP.
#: The forms are kept person-shaped on purpose: a bare "what X does Y" also matched "What
#: items does Adjusted Operating Margin exclude?", which is an entity question about a
#: document term, and the retrieval gate caught it.
_MULTI_HOP = re.compile(
    r"\b(why did .* (?:despite|although|even though|while)|despite|even though|although|compare|comparison|difference between|how does .* (?:affect|impact|relate to)|relationship between|reconcile|explain (?:how|why) .* and\b|both .* and|how many times|what (?:kind|kinds|sort|sorts|type|types) of \w+ (?:does|do|did|has|have)|which \w+ (?:does|do|did|has|have) \w+ (?:like|enjoy|play|do|visit|own|prefer)|where (?:has|have|did) \w+ (?:been|gone|travell?ed|lived|worked|camped|visited)|list (?:all|every|the))\b",
    re.IGNORECASE,
)
#: Enumeration questions about a *person*, case-sensitive on purpose: "What activities does
#: Melanie do?" and "Where has Caroline travelled?" need every mention gathered across
#: sessions. Two things are required, and the first attempt had only the second: an
#: enumeration cue (a plural noun, "what kind of", "where has", "how many") and a person - one
#: capitalised token followed by a lowercase word ("does Melanie partake"). Without the cue
#: the form matched 113 of 304 LoCoMo questions and "Which team does Acme use?", and every
#: extra route is a graph traversal on the query path; a document term is Title Case all the
#: way ("does Adjusted Operating Margin exclude") and never matches the person shape.
_MULTI_HOP_NAMED = re.compile(
    r"\b(?:(?:[Ww]hat|[Ww]hich)\s+(?:(?:kinds?|types?|sorts?)\s+of\s+\w+"
    r"|(?!(?:is|was|has|does|this|his|its|as|us|was)\b)[a-z]+s)|[Ww]here\s+(?:has|have)|[Hh]ow\s+many)\b"
    r"[^?]{0,40}?\b(?:has|have|does|do|did)\s+[A-Z][a-z]+(?:'s)?\s+(?=[a-z])"
)
_ENTITY = re.compile(
    r"\b(who (?:is|owns|leads|manages|reports to|approved)|which (?:team|company|person|agent)|related to|connected to|works? (?:with|for)|owner of|members? of|part of|"
    # relation cues the knowledge graph answers directly (typed facts + their evidence)
    r"(?:does|do|did|is|are|was|were) .{2,60}? (?:exclude|excludes|include|includes|operate in|provide|provides|serve|serves|acquire|acquired|approve|approved)|"
    r"(?:what|which) (?:items|costs|charges|segments|regions|products|companies)|how much did .{2,40}? pay|segments? of|excluded from|approved by|acquired by|driven by)\b",
    re.IGNORECASE,
)
_DOC_LOCAL = re.compile(
    r"\b(page \d+|section \d+(?:\.\d+)*|table \d+|figure \d+|appendix [a-z]|footnote|in the (?:report|document|contract|pdf|file|manual|filing)|according to the)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class RoutedQuery:
    query: str
    query_type: QueryType
    identifiers: list[str] = field(default_factory=list)
    signals: dict[str, bool] = field(default_factory=dict)
    needs_conversation: bool = True
    needs_memories: bool = True
    needs_knowledge: bool = True
    needs_graph: bool = False
    needs_summaries: bool = False


class QueryRouter:
    def route(self, query: str, *, has_thread: bool = True) -> RoutedQuery:
        q = query.strip()
        ids = _ID.findall(q)
        signals = {
            "identifier": bool(ids),
            "conversation": bool(_CONVERSATION.search(q)),
            "user_memory": bool(_USER_MEMORY.search(q)),
            "decision": bool(_DECISION.search(q)),
            "temporal": bool(_TEMPORAL.search(q)),
            "global": bool(_GLOBAL.search(q)),
            "multi_hop": bool(_MULTI_HOP.search(q) or _MULTI_HOP_NAMED.search(q)),
            "entity": bool(_ENTITY.search(q)),
            "doc_local": bool(_DOC_LOCAL.search(q)),
        }
        if signals["identifier"]:
            qt = QueryType.EXACT_IDENTIFIER
        elif signals["conversation"]:
            qt = QueryType.CONVERSATION_HISTORY
        elif signals["user_memory"]:
            qt = QueryType.USER_MEMORY
        elif signals["decision"]:
            qt = QueryType.DECISION
        elif signals["global"]:
            qt = QueryType.GLOBAL_SUMMARY
        elif signals["multi_hop"]:
            qt = QueryType.DOCUMENT_MULTI_HOP
        elif signals["entity"]:
            qt = QueryType.ENTITY_RELATION
        elif signals["temporal"]:
            qt = QueryType.TEMPORAL
        elif signals["doc_local"]:
            qt = QueryType.DOCUMENT_LOCAL
        else:
            qt = QueryType.GENERAL_SEMANTIC
        return self.routed(q, qt, identifiers=ids, signals=signals, has_thread=has_thread)

    def routed(
        self,
        query: str,
        qt: QueryType,
        *,
        identifiers: list[str],
        signals: dict[str, bool],
        has_thread: bool = True,
    ) -> RoutedQuery:
        """The RoutedQuery for an already decided type (rules, or a model when they could not)."""
        return RoutedQuery(
            query=query,
            query_type=qt,
            identifiers=identifiers,
            signals=signals,
            needs_conversation=has_thread
            and qt
            in (
                QueryType.CONVERSATION_HISTORY,
                QueryType.GENERAL_SEMANTIC,
                QueryType.DECISION,
                QueryType.USER_MEMORY,
                QueryType.DOCUMENT_MULTI_HOP,
                QueryType.TEMPORAL,
                QueryType.DOCUMENT_LOCAL,
                QueryType.GLOBAL_SUMMARY,
                QueryType.ENTITY_RELATION,
            ),
            needs_memories=qt is not QueryType.EXACT_IDENTIFIER,
            needs_knowledge=qt not in (QueryType.CONVERSATION_HISTORY, QueryType.USER_MEMORY),
            needs_graph=qt
            in (
                QueryType.ENTITY_RELATION,
                QueryType.DOCUMENT_MULTI_HOP,
                QueryType.TEMPORAL,
                QueryType.DECISION,
            ),
            needs_summaries=qt is QueryType.GLOBAL_SUMMARY,
        )
