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


@dataclass(frozen=True)
class Model:
    directory: str
    repo: str
    role: str
    note: str
    default: bool = False


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
    Model("sparse", "Qdrant/Splade_PP_en_v1", "sparse", "SPLADE learned sparse (gated)"),
    Model(
        "late-interaction",
        "answerdotai/answerai-colbert-small-v1",
        "late-interaction",
        "ColBERT multivector rescoring (gated)",
    ),
    Model("gliner2-base", "fastino/gliner2-base-v1", "extraction", "zero-shot NER/RE tier"),
)


def manifest_path(root: Path) -> Path:
    return root / "MANIFEST.json"


def load_manifest(root: Path) -> dict[str, Any]:
    path = manifest_path(root)
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text())
    except ValueError:
        return {}


def fetch(model: Model, root: Path, *, force: bool = False) -> dict[str, Any]:
    from huggingface_hub import HfApi, snapshot_download

    target = root / model.directory
    if target.exists() and not force and any(target.glob("*.safetensors")):
        revision = HfApi().model_info(model.repo).sha
        return {"repo": model.repo, "revision": revision, "role": model.role, "cached": True}
    snapshot_download(
        model.repo,
        local_dir=str(target),
        allow_patterns=ALLOW,
        ignore_patterns=IGNORE,
    )
    revision = HfApi().model_info(model.repo).sha
    return {"repo": model.repo, "revision": revision, "role": model.role, "cached": False}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", default="models", help="target directory (default: models)")
    parser.add_argument("--all", action="store_true", help="also fetch the benchmark challengers")
    parser.add_argument("--only", nargs="*", help="fetch these directory names only")
    parser.add_argument("--force", action="store_true", help="re-download even when present")
    parser.add_argument("--list", action="store_true", help="show the catalogue and exit")
    args = parser.parse_args(argv)

    if args.list:
        for m in MODELS:
            mark = "default  " if m.default else "challenger"
            sys.stdout.write(f"{mark} {m.directory:38} {m.repo:46} {m.note}\n")
        return 0

    wanted = [m for m in MODELS if m.default or args.all]
    if args.only:
        wanted = [m for m in MODELS if m.directory in set(args.only)]
        if not wanted:
            sys.stderr.write(f"no model matches {args.only}; try --list\n")
            return 2

    root = Path(args.dir)
    root.mkdir(parents=True, exist_ok=True)
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
