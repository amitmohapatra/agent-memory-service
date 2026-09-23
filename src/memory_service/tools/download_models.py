"""Fetch the frozen model set into ``models/`` (git-ignored).

    uv run python -m memory_service.tools.download_models            # the frozen set + docling
    uv run python -m memory_service.tools.download_models --list
    uv run python -m memory_service.tools.download_models --challengers benchmark/challengers.txt
    python -m memory_service.tools.download_models --export-onnx models/<dir>   # in the image
    python -m memory_service.tools.download_models --measure-onnx models/<dir> --out r.json

``--export-onnx`` and ``--measure-onnx`` are the other half of "the service never downloads
at runtime": the hub publishes no ONNX graph for granite-embedding-small-english-r2, so the
graph the ONNX runner loads is exported from the checkpoint that is already on disk, by a
command in this repository, and then measured against the torch runner before anything is
allowed to believe it. Both need torch and onnxruntime, so both run inside the runtime image.

The catalogue *is* ``constants.FROZEN_MODELS``: the directories this fetches are the ones the
service loads, named after the model they hold, so the manifest, the directory and the code
cannot drift apart. The service never downloads at runtime; a missing model is a startup
error, not a silent fall back to a stand-in. ``models/MANIFEST.json`` records the exact
revision of each so a benchmark result can name the weights it was produced with.

Benchmark challengers (other encoders, rerankers, learned sparse) are not in the catalogue
any more - ~7 GB of weights with no production code path. ``benchmark/challengers.txt``
lists them for an ad-hoc ``--challengers`` fetch.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Only weights and configuration; no PyTorch .bin duplicates of the safetensors, no ONNX
# exports we do not use, and no OpenVINO IR (sentence-transformers exports those on demand).
ALLOW = ["*.json", "*.txt", "*.safetensors", "*.model", "*.py", "*.md"]
IGNORE = ["*.bin", "*.h5", "*.msgpack", "*.ckpt", "onnx/*", "openvino/*", "*.onnx"]

# ...except for ONNX runtimes, which run the graph and nothing else. Fetching those under
# the rules above produced a directory of configuration with no model in it, and the
# "already downloaded" check looked for safetensors that were never going to be there — so
# every run re-downloaded a model that could never load.
ONNX_ALLOW = ["*.json", "*.txt", "*.model", "*.onnx", "*.onnx_data"]
ONNX_IGNORE = ["*.bin", "*.h5", "*.msgpack", "*.ckpt", "*.safetensors", "openvino/*"]

#: filename glob that says a directory already holds the weights, per runtime
PRESENT = {"torch": "*.safetensors", "onnx": "*.onnx"}


@dataclass(frozen=True)
class Model:
    directory: str
    repo: str
    role: str
    note: str
    default: bool = False
    #: "torch" loads safetensors through sentence-transformers; "onnx" fetches the graph only
    runtime: str = "torch"

    @property
    def allow(self) -> list[str]:
        return ONNX_ALLOW if self.runtime == "onnx" else ALLOW

    @property
    def ignore(self) -> list[str]:
        return ONNX_IGNORE if self.runtime == "onnx" else IGNORE


def _frozen() -> tuple[Model, ...]:
    """The catalogue, derived from the frozen set so the two cannot disagree."""
    from memory_service.config.constants import FROZEN_MODELS

    dense, nli, reranker = FROZEN_MODELS.dense, FROZEN_MODELS.nli, FROZEN_MODELS.reranker
    out = [
        Model(
            dense.local_dir,
            dense.id,
            "embedding",
            f"the dense encoder, {dense.dimension}-dim ({dense.backend} backend)",
            default=True,
            runtime="onnx" if dense.backend != "torch" else "torch",
        ),
        Model(
            nli.local_dir,
            nli.id,
            "nli",
            "claim-support classifier for the grounding cascade",
            default=True,
        ),
    ]
    if reranker is not None:
        out.append(
            Model(
                reranker.local_dir, reranker.id, "reranker", "cross-encoder reranker", default=True
            )
        )
    return tuple(out)


MODELS: tuple[Model, ...] = _frozen()

#: Docling resolves its layout and table models through the HuggingFace cache and downloads
#: them on first use — which is a runtime download the service does not permit, and which
#: fails outright in the runtime image (it runs as `nobody` with HOME=/nonexistent, so the
#: first PDF came back as CorruptSource). Fetched here instead, into the directory
#: MEMORY_DOCLING_ARTIFACTS points at.
DOCLING_DIRECTORY = "docling"


def manifest_path(root: Path) -> Path:
    return root / "MANIFEST.json"


def load_manifest(root: Path) -> dict[str, Any]:
    """The existing manifest, minus entries for weights this catalogue no longer knows.

    Each run merges into the file rather than rewriting it, so that fetching one model does
    not erase the provenance of the others. Without pruning, that merge is append-only: a
    model removed from ``MODELS`` keeps its entry forever, and ``benchmark/common.py`` stamps
    it into every result as if those weights were the ones in use. A ``late-interaction``
    entry outlived the strategy that used it by a fortnight this way, naming a directory that
    is not on disk. The catalogue is the only authority on what a directory name may mean.
    """
    path = manifest_path(root)
    if not path.is_file():
        return {}
    try:
        stored = json.loads(path.read_text())
    except ValueError:
        return {}
    if not isinstance(stored, dict):
        return {}
    known = {m.directory for m in MODELS} | {DOCLING_DIRECTORY} | _challenger_directories()
    return {k: v for k, v in stored.items() if k in known}


def _challenger_directories() -> set[str]:
    """Challenger entries stay in the manifest while the file that names them exists: a
    benchmark result produced with one must still be able to name its weights."""
    path = Path(__file__).resolve().parents[3] / "benchmark" / "challengers.txt"
    try:
        return {m.directory for m in load_challengers(path)} if path.is_file() else set()
    except SystemExit:
        return set()


def fetch(model: Model, root: Path, *, force: bool = False) -> dict[str, Any]:
    from huggingface_hub import HfApi, snapshot_download

    target = root / model.directory
    present = PRESENT[model.runtime]
    if (
        target.exists()
        and not force
        and (any(target.glob(present)) or any(target.glob(f"*/{present}")))
    ):
        revision = HfApi().model_info(model.repo).sha
        return {"repo": model.repo, "revision": revision, "role": model.role, "cached": True}
    snapshot_download(
        model.repo,
        local_dir=str(target),
        allow_patterns=model.allow,
        ignore_patterns=model.ignore,
    )
    revision = HfApi().model_info(model.repo).sha
    return {"repo": model.repo, "revision": revision, "role": model.role, "cached": False}


def _fetch_docling(root: Path, *, force: bool = False) -> None:
    """Docling ships its own downloader; use it rather than guessing repository names."""
    target = root / DOCLING_DIRECTORY
    if target.is_dir() and any(target.iterdir()) and not force:
        sys.stdout.write(f"{DOCLING_DIRECTORY:38} cached\n")
        return
    try:
        from docling.utils.model_downloader import download_models as fetch
    except ImportError:
        sys.stdout.write(
            f"{DOCLING_DIRECTORY:38} skipped (docling not installed in this environment)\n"
        )
        return
    sys.stdout.write(f"{DOCLING_DIRECTORY:38} <- docling layout/table models ... ")
    sys.stdout.flush()
    target.mkdir(parents=True, exist_ok=True)
    # Exactly the models the parser is configured to use, and no others. The defaults also
    # pull RapidOCR, code-formula and picture-classifier weights; the parser enables none of
    # them, and the OCR download fails outright in a slim image (cv2 needs libxcb). Fetching
    # what is not used is not free - it is disk, image size and a longer first start.
    fetch(
        output_dir=target,
        force=force,
        progress=False,
        with_layout=True,
        with_tableformer=True,
        with_code_formula=False,
        with_picture_classifier=False,
        with_rapidocr=False,
        with_easyocr=False,
    )
    sys.stdout.write("downloaded\n")


# ---------------------------------------------------------------------------
# ONNX export and the measurement that decides whether to use it
# ---------------------------------------------------------------------------

#: Opset 17 is where LayerNormalization became one node instead of eight, which is most of
#: what makes the fp32 CPU graph worth exporting at all.
OPSET = 17
GRAPH_DIR = "onnx"
FP32_GRAPH = "model.onnx"
INT8_GRAPH = "model_qint8.onnx"

#: Forty short queries and ten passages, fixed so two runs of --measure-onnx are comparable.
#: Nothing is measured on text that has never been seen; these are LoCoMo-shaped.
QUERIES = [
    "What did Caroline say about the painting class?",
    "When did Melanie move to Seattle?",
    "Who introduced John to his running club?",
    "How long has Angela been playing the cello?",
    "Where did they go on the anniversary trip?",
    "What was the name of the dog?",
    "Which restaurant did they argue about?",
    "Why did Nate leave the startup?",
    "What did the doctor recommend for the knee?",
    "How many siblings does Joanna have?",
    "What time does the ferry leave?",
    "Which book did Caroline recommend in March?",
    "What did they plant in the garden last spring?",
    "Who paid for the concert tickets?",
    "When was the wedding anniversary?",
    "What is the name of Melanie's sister?",
    "Where does Angela work now?",
    "What did John think of the new apartment?",
    "How did the job interview go?",
    "Which city did they visit first?",
    "What did she buy at the farmers market?",
    "Who taught him to cook?",
    "When did the renovation finish?",
    "What happened to the old car?",
    "Which team did he support?",
    "What did the therapist suggest?",
    "Where did they meet for coffee?",
    "How much did the camera cost?",
    "What did they name the new cat?",
    "When does the lease end?",
    "Who sent the birthday card?",
    "What did Melanie study at university?",
    "Which medication was changed?",
    "What did they watch on Friday night?",
    "How far is the hike?",
    "Who organised the surprise party?",
    "What did he say about the promotion?",
    "When did they adopt the second dog?",
    "Which flight was delayed?",
    "What did the landlord agree to fix?",
]
PASSAGES = [
    "Caroline mentioned that the painting class she joined in February meets every Tuesday "
    "evening at the community centre, and that the instructor, a retired art teacher named "
    "Dev, has been unusually patient with beginners.",
    "Melanie moved to Seattle in the autumn of 2022 after her partner accepted a position "
    "at a hospital there; she described the first winter as darker than she expected and "
    "said she missed the dry heat of Phoenix more than she thought she would.",
    "John was introduced to the running club by a colleague from his previous job, and he "
    "now runs with them three mornings a week along the river path, which he says has done "
    "more for his sleep than anything a doctor prescribed.",
    "Angela has played the cello since she was nine, taking a long break during her "
    "twenties, and returned to it two years ago after finding her old instrument in her "
    "mother's attic during a move.",
    "The anniversary trip was to a small town on the coast where they had stayed once "
    "before, ten years earlier, and they were surprised to find the same bakery open and "
    "run by the same family.",
    "Adjusted EBITDA increased to EUR 98 million from EUR 81 million despite lower revenue, "
    "driven mainly by restructuring savings realised in the second half and a one-off "
    "release of a litigation provision that the footnotes describe in some detail.",
    "The data centre migration finished on schedule in March, two weeks ahead of the "
    "contractual deadline, although the team reported that the final cutover weekend "
    "required a rollback of one storage cluster and a second attempt.",
    "Nate left the startup after the second funding round, citing a disagreement about the "
    "direction of the product rather than anything personal, and he took several months off "
    "before joining a larger company in a similar role.",
    "The doctor recommended physiotherapy twice a week for the knee rather than surgery, "
    "and said that the imaging showed wear consistent with age and running rather than a "
    "tear that would need repair.",
    "They planted tomatoes, beans and far too much mint in the garden last spring; the mint "
    "took over the bed by July and they spent most of August pulling it out again.",
]


def _texts() -> list[str]:
    return [*QUERIES, *PASSAGES]


def _percentiles(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)

    def at(q: float) -> float:
        return ordered[min(len(ordered) - 1, int(q * len(ordered)))]

    return {
        "mean_ms": round(sum(ordered) / len(ordered), 2),
        "p50_ms": round(at(0.50), 2),
        "p95_ms": round(at(0.95), 2),
    }


def _export_once(source: str, target: Path, *, attention: str, dynamo: bool) -> None:
    """Trace the checkpoint to *target*, or leave whatever was there untouched.

    The trace is written to a sibling and moved into place on success, because an attempt
    that dies part-way through writing 191 MB otherwise leaves a truncated graph behind —
    one that ``export_onnx`` has already reported as failed and that the ONNX runner would
    then load, or refuse to, at the next start.
    """
    import torch
    from transformers import AutoModel

    model = AutoModel.from_pretrained(
        source, dtype=torch.float32, attn_implementation=attention
    ).eval()
    # Two rows of different lengths so the dynamic axes are exercised by the trace rather
    # than folded into constants.
    ids = torch.cat(
        [torch.randint(1, 1000, (1, 24)), torch.randint(1, 1000, (1, 24))], dim=0
    ).long()
    mask = torch.ones_like(ids)
    mask[1, 16:] = 0
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".partial")
    try:
        torch.onnx.export(
            model,
            (ids, mask),
            str(partial),
            input_names=["input_ids", "attention_mask"],
            output_names=["last_hidden_state"],
            dynamic_axes={
                "input_ids": {0: "batch", 1: "sequence"},
                "attention_mask": {0: "batch", 1: "sequence"},
                "last_hidden_state": {0: "batch", 1: "sequence"},
            },
            opset_version=OPSET,
            dynamo=dynamo,
        )
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    os.replace(partial, target)


def export_onnx(directory: Path) -> int:
    """Write ``<directory>/onnx/model.onnx`` and ``model_qint8.onnx`` from the checkpoint.

    Every attempt that fails is reported with what it was and why. A graph is never written
    from a fallback that silently changed the model: if none of the attempts traces, this
    says so and exits non-zero, because a wrong graph produces vectors rather than errors.
    """
    if not (directory / "config.json").is_file():
        sys.stderr.write(f"{directory} does not hold a transformers checkpoint\n")
        return 2
    try:
        import onnx  # noqa: F401  - onnxruntime.quantization imports it
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError as exc:
        sys.stderr.write(
            f"--export-onnx needs torch, transformers and onnx ({exc}); run it inside the "
            "runtime image, not on the host\n"
        )
        return 2

    graph = directory / GRAPH_DIR / FP32_GRAPH
    # ModernBERT's masking helper traces under the TorchScript exporter with eager attention;
    # the other two are here because the roadmap says to try them before giving up, not
    # because either is known to work.
    attempts = (("eager", False), ("eager", True), ("sdpa", False))
    failures: list[str] = []
    for attention, dynamo in attempts:
        label = f"attn={attention} dynamo={dynamo}"
        sys.stdout.write(f"export {label} ... ")
        sys.stdout.flush()
        try:
            _export_once(str(directory), graph, attention=attention, dynamo=dynamo)
        except Exception as exc:
            failures.append(f"{label}: {type(exc).__name__}: {str(exc)[:300]}")
            sys.stdout.write("failed\n")
            continue
        sys.stdout.write(f"wrote {graph} ({graph.stat().st_size / 1e6:.0f} MB)\n")
        break
    else:
        sys.stderr.write("no exporter configuration traced this checkpoint:\n")
        for failure in failures:
            sys.stderr.write(f"  {failure}\n")
        return 1

    from onnxruntime.quantization import QuantType, quantize_dynamic

    int8 = directory / GRAPH_DIR / INT8_GRAPH
    sys.stdout.write("quantize weight_type=QInt8 ... ")
    sys.stdout.flush()
    quantize_dynamic(str(graph), str(int8), weight_type=QuantType.QInt8)
    sys.stdout.write(f"wrote {int8} ({int8.stat().st_size / 1e6:.0f} MB)\n")

    for name, cosine in _agreement(directory, (FP32_GRAPH, INT8_GRAPH)).items():
        sys.stdout.write(f"{name:22} min cosine vs torch {cosine:.4f}\n")
    return 0


def _agreement(directory: Path, graphs: tuple[str, ...]) -> dict[str, float]:
    """The lowest cosine between each graph's vectors and the torch runner's, over the same
    fifty texts. This is the number that says whether a graph is the same model."""
    import asyncio

    from memory_service.adapters.models.embeddings import (
        OnnxEmbedding,
        SentenceTransformersEmbedding,
    )
    from memory_service.config.constants import FROZEN_MODELS

    texts = _texts()
    spec = FROZEN_MODELS.dense.model_copy(update={"model_path": str(directory)})
    reference = asyncio.run(SentenceTransformersEmbedding(spec).embed_documents(texts))
    out: dict[str, float] = {}
    for name in graphs:
        graph_spec = spec.model_copy(update={"runtime": "onnx", "graph_file": f"onnx/{name}"})
        ours = asyncio.run(OnnxEmbedding(graph_spec).embed_documents(texts))
        out[name] = min(
            sum(a * b for a, b in zip(u, v, strict=True))
            for u, v in zip(ours, reference, strict=True)
        )
    return out


def ensure_dense_graph(root: Path, dense: Any = None) -> int:
    """Export the dense encoder's ONNX graph when the frozen runtime is the one that needs it.

    The hub publishes no ONNX graph for granite-embedding-small-english-r2, so the graph the
    ONNX runner loads is exported from the checkpoint by ``--export-onnx``. That used to be a
    separate command nothing called: the ``--dir`` bootstrap that docker-compose runs
    downloaded the weights and returned, so flipping ``DenseModel.runtime`` to ``onnx`` would
    have left a fresh ``docker compose up`` raising ``DependencyUnavailable`` on a file
    nothing ever wrote. That is the difference between one package that starts by plain
    docker compose and one that does not.

    A graph that is already there is left alone, so the bootstrap stays idempotent and a
    cached deployment does not pay the export twice. Nothing happens at all while the frozen
    runtime is ``torch``, which is what makes this safe to land before the flip.
    """
    from memory_service.adapters.models.embeddings import DEFAULT_GRAPH_FILE
    from memory_service.config.constants import FROZEN_MODELS

    spec = dense if dense is not None else FROZEN_MODELS.dense
    if spec.runtime != "onnx":
        return 0
    directory = root / spec.local_dir
    graph = directory / (spec.graph_file or DEFAULT_GRAPH_FILE)
    if graph.is_file():
        sys.stdout.write(f"onnx graph: {graph} present\n")
        return 0
    sys.stdout.write(f"onnx graph: exporting {graph}\n")
    return export_onnx(directory)


def measure_onnx(directory: Path, out: Path, *, threads: int, warmups: int = 5) -> int:
    """Time one query at a time on each runner and write the artifact.

    Single-query, because that is the shape of the hot path: at 20 rps a 10 ms batching
    window collects a mean batch of 1.2, so a throughput number measured on batches of 32
    says nothing about ``POST /v1/context``.
    """
    import asyncio
    import platform
    import time
    from datetime import UTC, datetime

    from memory_service.adapters.models.embeddings import (
        OnnxEmbedding,
        SentenceTransformersEmbedding,
    )
    from memory_service.config.constants import FROZEN_MODELS

    spec = FROZEN_MODELS.dense.model_copy(update={"model_path": str(directory)})
    runners: dict[str, Any] = {"torch_fp32": SentenceTransformersEmbedding(spec, threads=threads)}
    for label, name in (("onnx_fp32", FP32_GRAPH), ("onnx_int8", INT8_GRAPH)):
        graph = directory / GRAPH_DIR / name
        if not graph.is_file():
            sys.stderr.write(f"{graph} is not there; run --export-onnx first\n")
            return 2
        runners[label] = OnnxEmbedding(
            spec.model_copy(update={"runtime": "onnx", "graph_file": f"onnx/{name}"}),
            threads=threads,
        )

    async def time_all() -> dict[str, list[float]]:
        """Rotate the runners one query at a time, rather than timing each to completion.

        Timing them in sequence compares three different machines whenever anything else is
        on the box, because each runner meets whatever else was happening during its own
        stretch. That is not hypothetical here: the first reading of this command was taken
        while the unit suite held the same cores and came back 1.7x slower, which is recorded
        in MEASUREMENTS section 7 as a discarded run. Rotating cannot remove contention, but
        it spreads it evenly across the runners, so the ratios - the only figures that travel
        off this box - survive a noisy one.
        """
        for runner in runners.values():
            for query in QUERIES[:warmups]:
                await runner.embed_query(query)
        timings: dict[str, list[float]] = {label: [] for label in runners}
        for query in QUERIES:
            for label, runner in runners.items():
                start = time.perf_counter()
                await runner.embed_query(query)
                timings[label].append((time.perf_counter() - start) * 1000.0)
        return timings

    measured = asyncio.run(time_all())
    results: dict[str, Any] = {}
    for label, runner in runners.items():
        timings = measured[label]
        results[label] = {
            **_percentiles(timings),
            "encodes": len(timings),
            "fingerprint": runner.fingerprint(),
            "dimension": runner.dimension,
        }
        sys.stdout.write(f"{label:12} {results[label]['mean_ms']:8.1f} ms mean\n")

    baseline = results["torch_fp32"]["mean_ms"]
    for row in results.values():
        row["speedup_vs_torch_fp32"] = round(baseline / row["mean_ms"], 2)
    for name, cosine in _agreement(directory, (FP32_GRAPH, INT8_GRAPH)).items():
        results["onnx_fp32" if name == FP32_GRAPH else "onnx_int8"]["min_cosine_vs_torch"] = round(
            cosine, 6
        )

    document = {
        "benchmark": "encoder_runtime",
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "provenance": {
            "model_dir": str(directory),
            "model": FROZEN_MODELS.dense.id,
            "threads": threads,
            "warmups": warmups,
            "queries": len(QUERIES),
            "cosine_texts": len(_texts()),
            "platform": platform.platform(),
            "processor": platform.processor() or "unknown",
            "cpu_count": os.cpu_count(),
            "versions": {
                name: _version(name)
                for name in ("torch", "onnxruntime", "sentence-transformers", "transformers")
            },
            "caveat": (
                "A 4-core Docker VM without AVX2. Absolute milliseconds here describe this "
                "box and nothing else; only the ratios between the three runners travel to "
                "the 8 vCPU target VM, and even those move with AVX2/AVX-512."
            ),
        },
        "runners": results,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    sys.stdout.write(f"\nwrote {out}\n")
    return 0


def _version(package: str) -> str:
    from importlib.metadata import version

    try:
        return version(package)
    except Exception:
        return "unknown"


def _catalogue() -> None:
    """``--list``: the frozen set, which is everything ``--only`` will accept."""
    for m in MODELS:
        sys.stdout.write(f"frozen    {m.directory:38} {m.repo:46} {m.note}\n")
    sys.stdout.write(
        f"frozen    {DOCLING_DIRECTORY:38} {'(docling model_downloader)':46} "
        "layout + table models for the docling parser\n"
    )


def load_challengers(path: Path) -> tuple[Model, ...]:
    """``benchmark/challengers.txt``: one ``directory repo role [runtime]`` per line."""
    out: list[Model] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) not in (3, 4):
            raise SystemExit(f"{path}: expected 'directory repo role [runtime]', got {raw!r}")
        directory, repo, role = parts[:3]
        runtime = parts[3] if len(parts) == 4 else "torch"
        out.append(Model(directory, repo, role, "benchmark challenger", runtime=runtime))
    return tuple(out)


def _selection(
    only: set[str], catalogue: tuple[Model, ...], *, challengers: bool
) -> tuple[list[Model], bool]:
    """Which entries to fetch, and whether docling is among them.

    ``docling`` is not a HuggingFace repository in MODELS — docling's own downloader resolves
    it — but it still has to be addressable by name, or ``--only docling`` fetches nothing and
    exits 0, which reads as success. A challenger fetch never pulls docling.
    """
    if not only:
        return list(catalogue), not challengers
    return [m for m in catalogue if m.directory in only], DOCLING_DIRECTORY in only


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", default="models", help="target directory (default: models)")
    parser.add_argument(
        "--challengers",
        type=Path,
        default=None,
        help="fetch the benchmark challengers listed in this file instead of the frozen set",
    )
    parser.add_argument("--only", nargs="*", help="fetch these directory names only")
    parser.add_argument("--force", action="store_true", help="re-download even when present")
    parser.add_argument("--list", action="store_true", help="show the catalogue and exit")
    parser.add_argument(
        "--export-onnx",
        type=Path,
        default=None,
        metavar="DIR",
        help="export DIR's checkpoint to DIR/onnx/{model,model_qint8}.onnx (runtime image only)",
    )
    parser.add_argument(
        "--measure-onnx",
        type=Path,
        default=None,
        metavar="DIR",
        help="time torch fp32 vs the two graphs in DIR/onnx and write --out",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("benchmark/results/encoder_runtime.json"),
        help="where --measure-onnx writes its artifact",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=None,
        help="intra-op threads for --measure-onnx (default: the frozen DenseModel.threads)",
    )
    args = parser.parse_args(argv)

    if args.list:
        _catalogue()
        return 0
    if args.export_onnx is not None:
        return export_onnx(args.export_onnx)
    if args.measure_onnx is not None:
        from memory_service.config.constants import FROZEN_MODELS

        return measure_onnx(
            args.measure_onnx, args.out, threads=args.threads or FROZEN_MODELS.dense.threads
        )

    catalogue = load_challengers(args.challengers) if args.challengers else MODELS
    wanted, want_docling = _selection(
        set(args.only or ()), catalogue, challengers=args.challengers is not None
    )
    if not wanted and not want_docling:
        sys.stderr.write(f"no model matches {args.only}; try --list\n")
        return 2

    root = Path(args.dir)
    root.mkdir(parents=True, exist_ok=True)
    if want_docling:
        _fetch_docling(root, force=args.force)
    manifest = load_manifest(root)
    failures: list[str] = []
    for model in wanted:
        sys.stdout.write(f"{model.directory} <- {model.repo} ... ")
        sys.stdout.flush()
        try:
            entry = fetch(model, root, force=args.force)
        except Exception as exc:
            manifest[model.directory] = {
                "repo": model.repo,
                "role": model.role,
                "error": f"{type(exc).__name__}: {str(exc)[:160]}",
            }
            failures.append(f"{model.directory} ({type(exc).__name__})")
            sys.stdout.write("FAILED\n")
        else:
            manifest[model.directory] = entry
            sys.stdout.write(
                f"{'cached' if entry['cached'] else 'downloaded'} {entry['revision'][:12]}\n"
            )
        manifest_path(root).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    sys.stdout.write(f"\nmanifest: {manifest_path(root)}\n")
    if not failures and (rc := ensure_dense_graph(root)):
        return rc
    if failures:
        sys.stdout.write(
            "could not fetch: "
            + ", ".join(failures)
            + "\nGated repositories need `huggingface-cli login` and access granted on the model page.\n"
        )
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
