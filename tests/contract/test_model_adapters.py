"""Contract tests for the real-model adapters using tiny *randomly initialised* models built
offline (no downloads). They prove the adapter code paths — local loading with
``local_files_only``, normalisation, dimension discovery, batching, fingerprints, the
provider-policy check, and the CrossEncoder scoring/ordering contract — not model quality.

The ``models``-marked test at the bottom runs the same contract against real weights when
``BENCH_MODELS_DIR`` (or ``./models``) points at a directory containing ``granite-embedding-small-english-r2``
(and optionally ``ms-marco-MiniLM-L6-v2``).
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import pytest

from memory_service.config.constants import CrossEncoderModel, DenseModel
from memory_service.domain.errors import DependencyUnavailable

pytestmark = pytest.mark.contract

from tests.support_models import NO_RUNTIME  # noqa: E402

st = pytest.importorskip("sentence_transformers", reason=NO_RUNTIME)
st_models = pytest.importorskip("sentence_transformers.models", reason=NO_RUNTIME)
transformers = pytest.importorskip("transformers", reason=NO_RUNTIME)

_WORDS = """
adjusted ebitda increased to eur million despite lower revenue restructuring savings
the of and a in litigation settlement footnote definition table page
"""
_LETTERS = "abcdefghijklmnopqrstuvwxyz"
VOCAB = (
    ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    + _WORDS.split()
    + list(_LETTERS)
    + [f"##{c}" for c in _LETTERS]
)


def _tokenizer(path: Path):
    vocab = path / "vocab.txt"
    vocab.write_text("\n".join(VOCAB) + "\n", encoding="utf-8")
    tok = transformers.BertTokenizer(str(vocab), do_lower_case=True)
    tok.save_pretrained(str(path))
    return tok


@pytest.fixture(scope="module")
def tiny_st_model(tmp_path_factory) -> Path:
    """A 2-layer, 32-dim BERT with random weights wrapped as a SentenceTransformer."""
    import torch

    torch.manual_seed(0)
    root = tmp_path_factory.mktemp("tiny-st")
    base = root / "base"
    base.mkdir()
    _tokenizer(base)
    cfg = transformers.BertConfig(
        vocab_size=len(VOCAB),
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        max_position_embeddings=64,
    )
    transformers.BertModel(cfg).save_pretrained(str(base))
    model = st.SentenceTransformer(
        modules=[
            st_models.Transformer(str(base), max_seq_length=48),
            st_models.Pooling(32, pooling_mode="mean"),
            st_models.Normalize(),
        ],
        device="cpu",
    )
    out = root / "tiny-embed"
    model.save(str(out))
    return out


@pytest.fixture(scope="module")
def tiny_cross_encoder(tmp_path_factory) -> Path:
    import torch

    torch.manual_seed(1)
    root = tmp_path_factory.mktemp("tiny-ce")
    out = root / "tiny-ce"
    out.mkdir()
    _tokenizer(out)
    cfg = transformers.BertConfig(
        vocab_size=len(VOCAB),
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=64,
        max_position_embeddings=64,
        num_labels=1,
    )
    transformers.BertForSequenceClassification(cfg).save_pretrained(str(out))
    return out


async def _embedding_contract(emb, expected_dim: int | None = None) -> None:
    docs = ["Adjusted EBITDA increased to EUR 98 million", "restructuring savings", ""]
    vectors = await emb.embed_documents(docs)
    assert len(vectors) == 3 and all(len(v) == emb.dimension for v in vectors)
    if expected_dim:
        assert emb.dimension == expected_dim
    for v in vectors[:2]:
        assert math.isclose(math.sqrt(sum(x * x for x in v)), 1.0, rel_tol=1e-4)
    q = await emb.embed_query(docs[0])
    assert q == pytest.approx(vectors[0], abs=1e-5), "query and document paths agree"
    assert await emb.embed_documents([]) == []
    # deterministic across calls and independent of batch composition
    again = await emb.embed_documents([docs[1], docs[0]])
    assert again[1] == pytest.approx(vectors[0], abs=1e-5)
    assert f"-d{emb.dimension}" in emb.fingerprint(), "the width is part of the space's name"
    assert emb.info.locality == "local"


async def test_sentence_transformers_adapter_local_only(tiny_st_model: Path) -> None:
    from memory_service.adapters.models.embeddings import SentenceTransformersEmbedding

    spec = DenseModel(
        id="tiny/tiny-embed", model_path=str(tiny_st_model), dimension=32, batch_size=2
    )
    emb = SentenceTransformersEmbedding(spec, threads=1)
    await _embedding_contract(emb, expected_dim=32)
    assert emb.fingerprint() == "st-tiny-embed-torch-d32"
    assert emb.threads == 1
    assert emb.info.locality == "local"


def test_missing_local_model_is_a_dependency_error(tmp_path: Path) -> None:
    from memory_service.adapters.models.embeddings import SentenceTransformersEmbedding

    spec = DenseModel(id="nope/none", model_path=str(tmp_path / "missing"))
    with pytest.raises(DependencyUnavailable, match="could not be loaded"):
        SentenceTransformersEmbedding(spec)


def test_the_fingerprint_names_the_onnx_graph_when_one_is_frozen() -> None:
    """int8 and fp32 graphs of one model must never share a collection; without a graph the
    name is what it always was, so today's collections keep their names."""
    from memory_service.adapters.models.embeddings import dense_fingerprint

    base = DenseModel()
    assert dense_fingerprint(base) == "st-granite-embedding-small-english-r2-torch-d384"
    int8 = base.model_copy(update={"backend": "onnx", "graph_file": "onnx/model_quint8_avx2.onnx"})
    fp32 = base.model_copy(update={"backend": "onnx", "graph_file": "onnx/model.onnx"})
    assert dense_fingerprint(int8) != dense_fingerprint(fp32)
    assert (
        dense_fingerprint(int8)
        == "st-granite-embedding-small-english-r2-onnx-model_quint8_avx2-d384"
    )


async def test_cross_encoder_adapter_contract(tiny_cross_encoder: Path) -> None:
    from memory_service.adapters.models.rerankers import CrossEncoderReranker

    rr = CrossEncoderReranker(
        CrossEncoderModel(id="tiny/tiny-ce", model_path=str(tiny_cross_encoder))
    )
    docs = ["adjusted ebitda increased", "restructuring savings", "litigation settlement", "page"]
    out = await rr.rerank("why did adjusted ebitda increase", docs, top_k=3)
    assert len(out) == 3 and len({r.index for r in out}) == 3
    scores = [r.score for r in out]
    assert scores == sorted(scores, reverse=True), "ordered by descending score"
    assert all(0 <= r.index < len(docs) for r in out)
    # stable: same input -> same order and scores
    again = await rr.rerank("why did adjusted ebitda increase", docs, top_k=3)
    assert [(r.index, round(r.score, 6)) for r in again] == [
        (r.index, round(r.score, 6)) for r in out
    ]
    assert await rr.rerank("q", [], top_k=3) == []
    assert rr.fingerprint() == "ce-tiny-ce"


def _weights(name: str = "granite-embedding-small-english-r2") -> Path:
    root = os.environ.get("BENCH_MODELS_DIR") or ("models" if Path("models").is_dir() else "")
    if not root:
        pytest.skip(
            "no ./models — run `make model-test`, which mounts ./models into the runtime image (torch and onnxruntime ship no macOS x86_64 wheels, so these cannot run natively on an Intel Mac)"
        )
    path = Path(root) / name
    if not path.exists():
        pytest.skip(f"{path} not present")
    return path


@pytest.mark.models
async def test_granite_real_weights_contract() -> None:
    """Runs only when real weights are present (./models or BENCH_MODELS_DIR)."""
    path = _weights()
    from memory_service.adapters.models.embeddings import SentenceTransformersEmbedding

    emb = SentenceTransformersEmbedding(DenseModel(model_path=str(path)))
    await _embedding_contract(emb, expected_dim=384)
    a, b, c = await emb.embed_documents(
        [
            "Adjusted EBITDA increased despite lower revenue",
            "EBITDA rose even though sales fell",
            "The data centre migration finished on schedule",
        ]
    )
    sim = lambda x, y: sum(p * q for p, q in zip(x, y, strict=True))  # noqa: E731
    assert sim(a, b) > sim(a, c), "semantic neighbours closer than unrelated text"


@pytest.mark.models
async def test_the_onnx_runner_returns_the_same_vectors_as_the_torch_one() -> None:
    """The gate on the ONNX encoder: the graph is a re-expression of the checkpoint, not a
    different model. fp32 must agree with sentence-transformers to within rounding — below
    0.99 the two runners are different vector spaces and the collection would have to be
    rebuilt to switch between them, which is the one thing the fingerprint exists to
    prevent. int8 is a separate, lower bar and is measured, not asserted, here.
    """
    path = _weights()
    if not (path / "onnx" / "model.onnx").is_file():
        pytest.skip(
            f"no {path}/onnx/model.onnx — write it with `python -m "
            f"memory_service.tools.download_models --export-onnx {path}`"
        )
    from memory_service.adapters.models.embeddings import (
        OnnxEmbedding,
        SentenceTransformersEmbedding,
    )

    texts = [
        "Adjusted EBITDA increased to EUR 98 million",
        "the data centre migration finished on schedule",
        "What did Caroline say about the painting class?",
        "",
    ]
    onnx = OnnxEmbedding(DenseModel(runtime="onnx", model_path=str(path)))
    await _embedding_contract(onnx, expected_dim=384)
    assert onnx.fingerprint().startswith("onnx-")
    assert "-model-" in onnx.fingerprint(), "the graph file is part of the vector space"

    torch_side = SentenceTransformersEmbedding(DenseModel(model_path=str(path)))
    ours = await onnx.embed_documents(texts)
    theirs = await torch_side.embed_documents(texts)
    cosines = [
        sum(a * b for a, b in zip(u, v, strict=True)) for u, v in zip(ours, theirs, strict=True)
    ]
    assert min(cosines) >= 0.99, f"onnx fp32 disagrees with torch: {cosines}"
