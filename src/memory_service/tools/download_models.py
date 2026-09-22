"""Fetch the frozen model set into ``models/`` (git-ignored).

    uv run python -m memory_service.tools.download_models            # the frozen set + docling
    uv run python -m memory_service.tools.download_models --list
    uv run python -m memory_service.tools.download_models --challengers benchmark/challengers.txt

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
    args = parser.parse_args(argv)

    if args.list:
        _catalogue()
        return 0

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
