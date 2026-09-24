"""How a benchmark is wired: one place, read from ``BENCH_*`` variables.

The service's own ``MEMORY__*`` surface is topology and credentials only. What a benchmark
adds on top - whether Qdrant is the real server or the in-process local mode, which stand-ins
replace the stores a harness never wants to talk to, how deep the judged runs retrieve - is
decided here and handed to ``build_container(overrides=...)``, never smuggled through settings
the product does not have. The Makefile passes exactly these variables and nothing else.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from typing import Any, Literal

from memory_service.application.container import Overrides
from memory_service.config.constants import CONTEXT, FROZEN_MODELS, RETRIEVAL, local_model_path

#: Retrieval depth of the judged LoCoMo / LongMemEval runs - the values every judged result
#: on disk was produced with (Makefile -e flags, before they became constants here). The
#: shipped depth is ``constants.RETRIEVAL`` / ``constants.CONTEXT``; ``BENCH_DEPTH=judged``
#: selects these. They are benchmark constants, not settings: the kill list forbids any env
#: override of prefetch/fused/final/memories_max/token_budget.
PREFETCH_K = 200
FUSED_K = 200
FINAL_K = 100
MEMORIES_MAX = 100
TOKEN_BUDGET = 12000
#: the LLM ceiling and patience the judged runs need (a reasoning model spends its output
#: budget thinking before it emits anything; a judged question is a long call)
MAX_TOKENS = 16384
TIMEOUT = 120

JUDGED_RETRIEVAL = RETRIEVAL.model_copy(
    update={"prefetch_k": PREFETCH_K, "fused_k": FUSED_K, "final_k": FINAL_K}
)
JUDGED_CONTEXT = CONTEXT.model_copy(
    update={"memories_max": MEMORIES_MAX, "token_budget": TOKEN_BUDGET}
)


def _flag(name: str, default: str) -> str:
    return (os.environ.get(name) or default).strip().lower()


def default_embedding() -> Literal["frozen", "hash"]:
    """``frozen`` when the dense weights are under a model root, ``hash`` otherwise.

    The container path (Makefile ``bench-run``) always passes ``BENCH_EMBEDDING=frozen``: the
    image bakes the weights under ``/models`` and a missing model there is a build error. A
    host checkout may not have run ``make models``, and ``make gates`` on such a host must
    still produce its artifacts - marked ``representative: false`` - rather than load the real
    encoder or die trying. The stand-in is selected by the absence of the weights, never by a
    quiet fallback inside the adapter: the adapter itself still refuses to run without them.
    """
    return "frozen" if local_model_path(FROZEN_MODELS.dense.local_dir) is not None else "hash"


@dataclass(frozen=True)
class BenchEnv:
    #: ``qdrant``: the real server at ``MEMORY__SEARCH__QDRANT_URL``. ``memory``: qdrant-client
    #: local mode, an exact brute-force scan with no HNSW index - fine for a fixture-sized
    #: corpus and quietly O(n) beyond it (measured on SciFact: nDCG@10 0.012 against a
    #: published ~0.65, and it was the backend, not the retrieval).
    search: Literal["qdrant", "memory"] = "memory"
    #: ``native`` for the conversational benchmarks (LoCoMo, LongMemEval); ``disabled`` for
    #: the document benchmarks, which measure retrieval and not the graph.
    graph_enrichment: Literal["native", "disabled"] = "native"
    #: ``shipped``: the frozen constants. ``judged``: the depth above.
    depth: Literal["shipped", "judged"] = "shipped"
    #: ``hash`` replaces the frozen encoder with the deterministic stand-in; every result
    #: produced that way is labelled ``representative: false``. The default follows the
    #: weights (``default_embedding``): frozen when they are present, the stand-in when not.
    embedding: Literal["frozen", "hash"] = field(default_factory=default_embedding)
    #: ``openfga``: the real ReBAC service at ``MEMORY__AUTHORIZATION__*``. ``memory``: the
    #: in-process model.
    #:
    #: The in-process model is not a small simplification. Every judged score in this
    #: repository was produced without OpenFGA in the path, which is why the scope-cache
    #: invalidation defect - content writes throwing the authorized scope away 730 times over
    #: 369 ingested turns, and each miss costing five sequential ListObjects - survived every
    #: benchmark and only appeared in the first HTTP load test, as 503s.
    authorization: Literal["openfga", "memory"] = "memory"
    #: ``dragonfly``: the real cache at ``MEMORY__CACHE__URL``. ``memory``: a dict.
    #:
    #: A dict never evicts, never fails and never races, so bundle-cache correctness under a
    #: real cache has never been measured by a harness either.
    cache: Literal["dragonfly", "memory"] = "memory"
    #: ``deberta``: the frozen NLI head. ``lexical``: the deterministic stand-in whose reports
    #: are labelled ``representative: false``. The head costs ~700 MB per process, which is
    #: why it is not the default here.
    nli: Literal["deberta", "lexical"] = "lexical"

    @classmethod
    def from_environ(cls) -> BenchEnv:
        values = {
            "search": _flag("BENCH_SEARCH", cls.search),
            "graph_enrichment": _flag("BENCH_GRAPH_ENRICHMENT", cls.graph_enrichment),
            "depth": _flag("BENCH_DEPTH", cls.depth),
            "embedding": _flag("BENCH_EMBEDDING", default_embedding()),
            "authorization": _flag("BENCH_AUTHZ", cls.authorization),
            "cache": _flag("BENCH_CACHE", cls.cache),
            "nli": _flag("BENCH_NLI", cls.nli),
        }
        allowed = {
            "search": ("qdrant", "memory"),
            "graph_enrichment": ("native", "disabled"),
            "depth": ("shipped", "judged"),
            "embedding": ("frozen", "hash"),
            "authorization": ("openfga", "memory"),
            "cache": ("dragonfly", "memory"),
            "nli": ("deberta", "lexical"),
        }
        for name, value in values.items():
            if value not in allowed[name]:
                raise SystemExit(f"BENCH_{name.upper()}={value!r}: expected one of {allowed[name]}")
        return cls(**values)  # type: ignore[arg-type]

    def overrides(self, **changes: Any) -> Overrides:
        """The stand-ins every harness runs with.

        An in-process cache and queue (a benchmark drains its own jobs synchronously, which
        only the recording queue supports), the in-memory authorization model and blob store
        (neither is what a benchmark measures), the lexical NLI (grounding is not scored by
        these harnesses and the DeBERTa head costs 700 MB per process), local-mode Qdrant
        unless ``BENCH_SEARCH=qdrant``, and the depth ``BENCH_DEPTH`` names.

        Under the hash stand-in the document parser is the builtin one too: a host without
        the weights has no docling artifacts either, and the artifacts such a run produces
        are already non-representative.
        """
        base = Overrides(
            cache="memory" if self.cache == "memory" else None,
            tasks="memory",
            authorization="memory" if self.authorization == "memory" else None,
            blob="memory",
            nli="lexical" if self.nli == "lexical" else None,
            search="memory" if self.search == "memory" else None,
            graph_enrichment="disabled" if self.graph_enrichment == "disabled" else None,
            embedding="hash" if self.embedding == "hash" else None,
            document_parser="builtin" if self.embedding == "hash" else None,
            retrieval=JUDGED_RETRIEVAL if self.depth == "judged" else None,
            context=JUDGED_CONTEXT if self.depth == "judged" else None,
        )
        return replace(base, **changes) if changes else base


BENCH = BenchEnv.from_environ()


def bench_overrides(**changes: Any) -> Overrides:
    return BENCH.overrides(**changes)


def bench_llm_settings() -> dict[str, Any]:
    """LLM fields the judged configuration pins: the output ceiling a reasoning model needs,
    the patience a judged question needs, and retries off (the pacer's rate is then the
    actual request rate). Applied by ``benchmark.retrieval._settings`` under
    ``BENCH_DEPTH=judged``; the shipped values apply otherwise."""
    if BENCH.depth != "judged":
        return {}
    return {"max_tokens": MAX_TOKENS, "timeout_seconds": TIMEOUT, "max_retries": 0}


def bench_retrieval(overrides: Overrides) -> Any:
    """The retrieval tuning a container built with ``overrides`` runs with."""
    return overrides.retrieval or RETRIEVAL


def bench_context(overrides: Overrides) -> Any:
    return overrides.context or CONTEXT
