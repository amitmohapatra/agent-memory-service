"""Fetch the local model weights into ``models/`` (git-ignored).

    uv run python -m memory_service.tools.download_models            # the defaults
    uv run python -m memory_service.tools.download_models --all      # + benchmark challengers
    uv run python -m memory_service.tools.download_models --list

The service reads weights from local directories and never downloads at runtime: a missing
model is a startup error, not a silent fall back to a stand-in. This is the one command that
puts them in place, and it records the exact revision of each in ``models/MANIFEST.json`` so a
benchmark result can name the weights it was produced with.

Each directory is named after the model it holds, so the manifest, the directory and the
configuration cannot drift apart: swapping the default embedding is a visible change to
``MEMORY__MODELS__EMBEDDING__MODEL_PATH``, not a different model quietly appearing under a
generic ``models/embedding`` path.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Only weights and configuration; no PyTorch .bin duplicates of the safetensors, no ONNX
# exports we do not use, and no OpenVINO IR (sentence-transformers exports those on demand).
ALLOW = ["*.json", "*.txt", "*.safetensors", "*.model", "*.py", "*.md"]
IGNORE = ["*.bin", "*.h5", "*.msgpack", "*.ckpt", "onnx/*", "openvino/*", "*.onnx"]

# ...except for the fastembed adapters, which run the ONNX graph and nothing else. Fetching
# them under the rules above produced a directory of configuration with no model in it, and
# the "already downloaded" check looked for safetensors that were never going to be there —
# so every run re-downloaded a model that could never load.
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
    #: "torch" loads safetensors through sentence-transformers; "onnx" is a fastembed export
    runtime: str = "torch"

    @property
    def allow(self) -> list[str]:
        return ONNX_ALLOW if self.runtime == "onnx" else ALLOW

    @property
    def ignore(self) -> list[str]:
        return ONNX_IGNORE if self.runtime == "onnx" else IGNORE


MODELS: tuple[Model, ...] = (
    # ---- defaults: what a normal deployment runs -------------------------------------
    Model(
        "granite-embedding-small-english-r2",
        "ibm-granite/granite-embedding-small-english-r2",
        "embedding",
        "default dense encoder, 384-dim: lowest query p95 of the candidates benchmarked",
        default=True,
    ),
    Model(
        "ms-marco-MiniLM-L6-v2",
        "cross-encoder/ms-marco-MiniLM-L6-v2",
        "reranker",
        "default cross-encoder: the only one measured inside a CPU latency budget",
        default=True,
    ),
    Model(
        "deberta-v3-base-mnli-fever-anli",
        "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
        "nli",
        "claim-support classifier for the grounding cascade",
        default=True,
    ),
    # ---- challengers: only needed to re-run the benchmarks ---------------------------
    Model(
        "bge-small-en-v1.5",
        "BAAI/bge-small-en-v1.5",
        "embedding",
        "384-dim, indexes ~3x faster but a higher query p95",
    ),
    Model(
        "granite-embedding-english-r2",
        "ibm-granite/granite-embedding-english-r2",
        "embedding",
        "English-only, 768-dim",
    ),
    Model("bge-base-en-v1.5", "BAAI/bge-base-en-v1.5", "embedding", "768-dim, ~2x the query cost"),
    Model("bge-m3", "BAAI/bge-m3", "embedding", "multilingual, 1024-dim, 8k context"),
    Model(
        "qwen3-embedding-0.6b",
        "Qwen/Qwen3-Embedding-0.6B",
        "embedding",
        "multilingual, 1024-dim; 2.0s per query on CPU, needs a GPU",
    ),
    Model(
        "gte-multilingual-base",
        "Alibaba-NLP/gte-multilingual-base",
        "embedding",
        "multilingual, 768-dim",
    ),
    Model(
        "bge-reranker-v2-m3",
        "BAAI/bge-reranker-v2-m3",
        "reranker",
        "higher quality, GPU only: 35.8s for 20 candidates on CPU",
    ),
    Model(
        "granite-embedding-reranker-english-r2",
        "ibm-granite/granite-embedding-reranker-english-r2",
        "reranker",
        "English-only CPU-light reranker",
    ),
    # Qdrant's mirror, not the author's repository: fastembed loads its own ONNX export, and
    # prithivida/Splade_PP_en_v1 ships one whose inputs are named input_mask/segment_ids —
    # which fastembed does not feed, so it fails with "Required inputs are missing".
    Model(
        "Splade_PP_en_v1",
        "Qdrant/Splade_PP_en_v1",
        "sparse",
        "SPLADE learned sparse (gated)",
        runtime="onnx",
    ),
    Model("gliner2-base", "fastino/gliner2-base-v1", "extraction", "zero-shot NER/RE tier"),
)

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
    known = {m.directory for m in MODELS} | {DOCLING_DIRECTORY}
    return {k: v for k, v in stored.items() if k in known}


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


def _catalogue() -> None:
    """``--list``: everything ``--only`` will accept."""
    for m in MODELS:
        mark = "default  " if m.default else "challenger"
        sys.stdout.write(f"{mark} {m.directory:38} {m.repo:46} {m.note}\n")
    sys.stdout.write(
        f"default   {DOCLING_DIRECTORY:38} {'(docling model_downloader)':46} "
        "layout + table models for the docling parser\n"
    )


def _selection(only: set[str], fetch_all: bool) -> tuple[list[Model], bool]:
    """Which catalogue entries to fetch, and whether docling is among them.

    ``docling`` is not a HuggingFace repository in MODELS — docling's own downloader resolves
    it — but it still has to be addressable by name, or ``--only docling`` fetches nothing and
    exits 0, which reads as success.
    """
    if not only:
        return [m for m in MODELS if m.default or fetch_all], True
    return [m for m in MODELS if m.directory in only], DOCLING_DIRECTORY in only


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", default="models", help="target directory (default: models)")
    parser.add_argument("--all", action="store_true", help="also fetch the benchmark challengers")
    parser.add_argument("--only", nargs="*", help="fetch these directory names only")
    parser.add_argument("--force", action="store_true", help="re-download even when present")
    parser.add_argument("--list", action="store_true", help="show the catalogue and exit")
    args = parser.parse_args(argv)

    if args.list:
        _catalogue()
        return 0

    wanted, want_docling = _selection(set(args.only or ()), args.all)
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
