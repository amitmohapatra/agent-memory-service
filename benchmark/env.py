"""How a benchmark is wired: one place, read from ``BENCH_*`` variables.

The service's own ``MEMORY__*`` surface is topology and credentials only. What a benchmark
adds on top - whether Qdrant is the real server or the in-process local mode, which stand-ins
replace the stores a harness never wants to talk to, how deep the judged runs retrieve - is
decided here and handed to ``build_container(overrides=...)``, never smuggled through settings
the product does not have. The Makefile passes exactly these variables and nothing else.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, replace
from typing import Any, Literal

from memory_service.application.container import Overrides
from memory_service.config.constants import CONTEXT, FROZEN_MODELS, RETRIEVAL, local_model_path
from memory_service.ports.search import VectorName

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

#: The candidate-recall SWEEP depth. ``complete_evidence_in_candidates`` is measured at
#: ``final_k`` and therefore cannot see a gold turn carried by a memory ranked below it: at
#: the shipped depth of 50 the metric has a hard ceiling at rank 49, so "the evidence is
#: absent" and "the evidence is at rank 73" are the same reading. Retrieving 200 and scoring
#: @50 / @100 / @200 off ONE run's ``evidence_ranks`` separates them, and the answer decides
#: whether candidate-seeded expansion is needed at all or whether fusion is discarding
#: evidence the arms already found. Measurement only - never a shipped depth, and its latency
#: is meaningless because the render carries four times the memories.
SWEEP_PREFETCH_K = 400
SWEEP_FUSED_K = 400
SWEEP_FINAL_K = 200
SWEEP_MEMORIES_MAX = 200
SWEEP_TOKEN_BUDGET = 32000

JUDGED_RETRIEVAL = RETRIEVAL.model_copy(
    update={"prefetch_k": PREFETCH_K, "fused_k": FUSED_K, "final_k": FINAL_K}
)
JUDGED_CONTEXT = CONTEXT.model_copy(
    update={"memories_max": MEMORIES_MAX, "token_budget": TOKEN_BUDGET}
)
SWEEP_RETRIEVAL = RETRIEVAL.model_copy(
    update={
        "prefetch_k": SWEEP_PREFETCH_K,
        "fused_k": SWEEP_FUSED_K,
        "final_k": SWEEP_FINAL_K,
    }
)
SWEEP_CONTEXT = CONTEXT.model_copy(
    update={"memories_max": SWEEP_MEMORIES_MAX, "token_budget": SWEEP_TOKEN_BUDGET}
)

#: D6 step 2 spends a fitted fusion on DEPTH: half the candidates per arm, half the fused
#: list, half the memory recall - and the same final cut and render as the shipped arm, so
#: every depth the source harness scores (@10/@20/@50) stays comparable to the run that
#: measured the shipped depth. What halves is the work: the store's memory query is
#: ``max(fused_k, derived_k(memory_recall_k))``, which is 200 shipped and 100 here.
HALVED_RETRIEVAL = RETRIEVAL.model_copy(
    update={
        "prefetch_k": RETRIEVAL.prefetch_k // 2,
        "fused_k": RETRIEVAL.fused_k // 2,
        "memory_recall_k": RETRIEVAL.memory_recall_k // 2,
    }
)


#: ``shipped`` is absent from both maps on purpose: it means "change nothing", and a None
#: from ``.get`` is exactly what ``Overrides`` reads as "leave the frozen constant alone".
#: ``halved`` is absent from the CONTEXT map for the same reason: it changes candidate depth,
#: not what the render carries.
_DEPTH_RETRIEVAL = {
    "judged": JUDGED_RETRIEVAL,
    "sweep": SWEEP_RETRIEVAL,
    "halved": HALVED_RETRIEVAL,
}
_DEPTH_CONTEXT = {"judged": JUDGED_CONTEXT, "sweep": SWEEP_CONTEXT}


def _flag(name: str, default: str) -> str:
    return (os.environ.get(name) or default).strip().lower()


_TRUE = ("1", "true", "yes", "on")
_FALSE = ("0", "false", "no", "off")
#: ``BENCH_HYBRID_WEIGHTS=equal`` pins equal weights explicitly, for the same reason
#: ``_switch`` is tri-state.
EQUAL_WEIGHTS = "equal"


def _switch(name: str) -> bool | None:
    """A tri-state arm switch: unset leaves the constant alone, ``on``/``off`` pins it.

    Unset and "off" are not the same thing. The day a constant is promoted, the control arm
    still has to be able to SAY off - otherwise it quietly measures the new default and
    reports it as the control, which is how a promotion stops being falsifiable.
    """
    raw = (os.environ.get(name) or "").strip().lower()
    if not raw:
        return None
    if raw not in (*_TRUE, *_FALSE):
        raise SystemExit(f"{name}={raw!r}: expected one of {(*_TRUE, *_FALSE)}")
    return raw in _TRUE


def _weights(raw: str) -> tuple[tuple[VectorName, float], ...] | None:
    """``BENCH_HYBRID_WEIGHTS`` as arm/weight pairs, refusing anything it cannot weigh.

    The fit (``benchmark.fit_rrf_weights``) writes a JSON object keyed by arm name, so the
    arm is pasted from the fit rather than retyped. An unknown arm name is a typo that would
    otherwise weigh nothing and be reported as a fitted run, so it is refused here. Unset is
    ``None`` (leave the constant alone) and ``equal`` is the empty weighting, explicitly.
    """
    text = raw.strip()
    if not text:
        return None
    if text.lower() == EQUAL_WEIGHTS:
        return ()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as error:
        raise SystemExit(f"BENCH_HYBRID_WEIGHTS={text!r} is not JSON: {error}") from error
    if not isinstance(parsed, dict) or not parsed:
        raise SystemExit("BENCH_HYBRID_WEIGHTS must be a non-empty JSON object of arm -> weight")
    out: list[tuple[VectorName, float]] = []
    for name, weight in parsed.items():
        try:
            vector = VectorName(name)
        except ValueError:
            raise SystemExit(
                f"BENCH_HYBRID_WEIGHTS names {name!r}: expected one of "
                f"{tuple(v.value for v in VectorName)}"
            ) from None
        if not isinstance(weight, int | float) or isinstance(weight, bool) or float(weight) < 0:
            raise SystemExit(f"BENCH_HYBRID_WEIGHTS[{name}]={weight!r}: expected a weight >= 0")
        out.append((vector, float(weight)))
    return tuple(sorted(out, key=lambda pair: pair[0].value))


def default_embedding() -> Literal["frozen", "hash"]:
    """``frozen`` when the dense weights are under a model root, ``hash`` otherwise.

    The container path (Makefile ``bench-run``) always passes ``BENCH_EMBEDDING=frozen``: the
    image bakes the weights under ``/models`` and a missing model there is a build error. A
    host checkout may not have run ``make models``, and ``make gates`` on such a host must
    still produce its artifacts - marked ``representative: false`` - rather than load the real
    encoder or die trying. The stand-in is selected by the absence of the weights, never by a
    quiet fallback inside the adapter: the adapter itself still refuses to run without them.
    """
    present = all(
        local_model_path(model.local_dir) is not None
        for model in (FROZEN_MODELS.dense, FROZEN_MODELS.dense_ml)
    )
    return "frozen" if present else "hash"


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
    #: ``shipped``: the frozen constants. ``judged``: the judged depth above. ``sweep``:
    #: retrieve 200 so candidate completeness can be read at @50/@100/@200 from one run.
    #: ``halved``: half the candidate depth, the shipped render (D6 step 2's own arm).
    depth: Literal["shipped", "judged", "sweep", "halved"] = "shipped"
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
    #: ``frozen``: the frozen NLI head (mDeBERTa, FP32 ONNX). ``lexical``: the deterministic
    #: stand-in whose reports are labelled ``representative: false``. The head costs ~1.1 GB
    #: per process, which is why it is not the default here.
    nli: Literal["frozen", "lexical"] = "lexical"
    #: ``ensemble``: both dense spaces (English + multilingual), the shipped configuration.
    #: ``english``: the English encoder alone, searched for every script - the arm every
    #: number before the ensemble was measured with.
    dense: Literal["ensemble", "english"] = "ensemble"
    #: The store's RRF weight per hybrid arm, from ``BENCH_HYBRID_WEIGHTS`` as the JSON object
    #: the offline fit prints (``{"bm25": 2.0, "dense_en": 1.0, "dense_ml": 0.5}``), or
    #: ``equal`` for the empty weighting. ``None`` is unset: the frozen constant stands. This
    #: is the ARM, not a promotion - ``RetrievalSettings.hybrid_weights`` stays ``None`` until
    #: the arm run with these weights clears D6 step 2's gate.
    hybrid_weights: tuple[tuple[VectorName, float], ...] | None = None
    #: Entity -> memory routing as one more RRF list (``BENCH_ENTITY_PREFETCH``), D6 step 3's
    #: arm. ``None`` is unset, the way every other switch here reads unset.
    entity_prefetch: bool | None = None

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
            "dense": _flag("BENCH_DENSE", cls.dense),
        }
        tuning: dict[str, Any] = {
            "hybrid_weights": _weights(os.environ.get("BENCH_HYBRID_WEIGHTS") or ""),
            "entity_prefetch": _switch("BENCH_ENTITY_PREFETCH"),
        }
        allowed = {
            "search": ("qdrant", "memory"),
            "graph_enrichment": ("native", "disabled"),
            "depth": ("shipped", "judged", "sweep", "halved"),
            "embedding": ("frozen", "hash"),
            "authorization": ("openfga", "memory"),
            "cache": ("dragonfly", "memory"),
            "nli": ("frozen", "lexical"),
            "dense": ("ensemble", "english"),
        }
        for name, value in values.items():
            if value not in allowed[name]:
                raise SystemExit(f"BENCH_{name.upper()}={value!r}: expected one of {allowed[name]}")
        return cls(**values, **tuning)  # type: ignore[arg-type]

    def _retrieval(self) -> Any:
        """The retrieval tuning this arm runs with, or ``None`` for "change nothing".

        ``None`` is what ``Overrides`` reads as "leave the frozen constant alone", so an arm
        that sets no query-side switch is byte-identical to every run recorded before them.
        """
        tuning: dict[str, Any] = {}
        if self.hybrid_weights is not None:
            # the empty weighting IS equal weights, which the setting spells ``None``
            tuning["hybrid_weights"] = dict(self.hybrid_weights) or None
        if self.entity_prefetch is not None:
            tuning["entity_prefetch"] = self.entity_prefetch
        depth = _DEPTH_RETRIEVAL.get(self.depth)
        if not tuning:
            return depth
        return (depth or RETRIEVAL).model_copy(update=tuning)

    def overrides(self, **changes: Any) -> Overrides:
        """The stand-ins every harness runs with.

        An in-process cache and queue (a benchmark drains its own jobs synchronously, which
        only the recording queue supports), the in-memory authorization model and blob store
        (neither is what a benchmark measures), the lexical NLI (grounding is not scored by
        these harnesses and the frozen head costs ~1.1 GB per process), local-mode Qdrant
        unless ``BENCH_SEARCH=qdrant``, and the depth ``BENCH_DEPTH`` names.

        Under the hash stand-in the document parser is the builtin one too: a host without
        the weights has no docling artifacts either, and the artifacts such a run produces
        are already non-representative.

        The two query-side arms (``BENCH_HYBRID_WEIGHTS``, ``BENCH_ENTITY_PREFETCH``) are
        applied on TOP of whatever depth is selected, so an arm is one measured change and
        not a second one smuggled in with it.
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
            multilingual_dense="disabled" if self.dense == "english" else None,
            document_parser="builtin" if self.embedding == "hash" else None,
            retrieval=self._retrieval(),
            context=_DEPTH_CONTEXT.get(self.depth),
        )
        return replace(base, **changes) if changes else base


BENCH = BenchEnv.from_environ()


def bench_overrides(**changes: Any) -> Overrides:
    return BENCH.overrides(**changes)


def bench_llm_settings() -> dict[str, Any]:
    """LLM fields every judged run pins: the output ceiling a reasoning model needs, the
    patience a judged question needs, and retries off (the pacer's rate is then the actual
    request rate).

    These belong to the JUDGE, which is the instrument, not to the system under test: the
    only thing that enables an LLM in a benchmark process is ``BENCH_LLM_ENV``, and it pins
    ``uses=["grounding_judge"]``. Gating them on ``BENCH_DEPTH=judged`` coupled the ruler to
    what it was measuring. A shipped-depth run fell back to the shipped ceiling of 1024
    tokens, and a reasoning model spent all 1024 of them reasoning and emitted no text at all
    (``finish_reason='length'``, ``reasoning_tokens=1024``, zero content). Those rows score
    WRONG by construction, so the same code read 0.7467 at shipped depth against 0.7993 at
    judged depth - 28 of 304 rows failed the judge, and 25 of those 28 had every gold
    evidence item already in the bundle. The depth being measured must not change what the
    ruler is able to say about it.
    """
    return {"max_tokens": MAX_TOKENS, "timeout_seconds": TIMEOUT, "max_retries": 0}


def bench_retrieval(overrides: Overrides) -> Any:
    """The retrieval tuning a container built with ``overrides`` runs with."""
    return overrides.retrieval or RETRIEVAL


def bench_context(overrides: Overrides) -> Any:
    return overrides.context or CONTEXT
