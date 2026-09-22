"""The embedding / reranker benchmark modules: candidate matrices, verdict selection on
synthetic rows, and one end-to-end run of the embedding benchmark with the hash stand-in."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from benchmark import embedding, reranker

from tests.conftest import PG_AVAILABLE


def _row(recall: float, egr: float, **extra: object) -> dict[str, object]:
    return {
        "quality": {"critical_recall_at_k": recall, "critical_evidence_group_recall": egr},
        **extra,
    }


def _emb_row(recall: float, egr: float, *, q_p95: float, dim: int) -> dict[str, object]:
    return _row(recall, egr, embed_ms={"query_p95": q_p95}, dimension=dim)


def _rr_row(recall: float, egr: float, *, k: int, p95: float) -> dict[str, object]:
    return _row(recall, egr, candidate_k=k, rerank_ms={"p95": p95})


@pytest.mark.unit
class TestVerdicts:
    def test_embedding_quality_first_then_speed(self) -> None:
        rows = {
            "large/torch": _emb_row(1.0, 1.0, q_p95=30.0, dim=768),
            "small/torch": _emb_row(1.0, 1.0, q_p95=12.0, dim=384),
            "small/onnx": _emb_row(0.9, 1.0, q_p95=4.0, dim=384),
            "large/openvino": {"skipped": "DependencyUnavailable: no openvino"},
        }
        verdict = embedding.pick_default(rows, embedding.embedding_cost)
        assert verdict["default"] == "small/torch"
        assert verdict["eligible"] == ["small/torch", "large/torch"]
        assert verdict["best"] == {
            "critical_recall_at_k": 1.0,
            "critical_evidence_group_recall": 1.0,
        }

    def test_egr_breaks_recall_ties_before_speed(self) -> None:
        rows = {
            "fast-incomplete": _emb_row(1.0, 0.8, q_p95=1.0, dim=64),
            "slow-complete": _emb_row(1.0, 1.0, q_p95=50.0, dim=768),
        }
        assert embedding.pick_default(rows, embedding.embedding_cost)["default"] == "slow-complete"

    def test_dimension_breaks_latency_ties(self) -> None:
        rows = {
            "b-768": _emb_row(1.0, 1.0, q_p95=10.0, dim=768),
            "a-384": _emb_row(1.0, 1.0, q_p95=10.0, dim=384),
        }
        assert embedding.pick_default(rows, embedding.embedding_cost)["default"] == "a-384"

    def test_all_skipped_has_no_default(self) -> None:
        rows = {"x": {"skipped": "ImportError: torch"}, "y": {"skipped": "OSError: weights"}}
        verdict = embedding.pick_default(rows, embedding.embedding_cost)
        assert verdict["default"] is None
        assert verdict["eligible"] == []

    def test_reranker_cheapest_k_keeping_best_quality(self) -> None:
        rows = {
            "sentence_transformers@k15": _rr_row(1.0, 0.9, k=15, p95=20.0),
            "sentence_transformers@k20": _rr_row(1.0, 1.0, k=20, p95=25.0),
            "sentence_transformers@k25": _rr_row(1.0, 1.0, k=25, p95=30.0),
            "onnx@k20": _rr_row(1.0, 1.0, k=20, p95=15.0),
            "lexical@k15": _rr_row(0.8, 0.8, k=15, p95=0.5),
            "disabled@k15": _rr_row(0.7, 0.7, k=15, p95=0.0),
        }
        verdict = embedding.pick_default(rows, reranker.reranker_cost)
        assert verdict["default"] == "onnx@k20"
        assert verdict["eligible"] == [
            "onnx@k20",
            "sentence_transformers@k20",
            "sentence_transformers@k25",
        ]


@pytest.mark.unit
class TestCandidates:
    def test_embedding_matrix(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.setenv("BENCH_MODELS_DIR", str(tmp_path))
        full = embedding.candidates()
        # derived from the catalogue, so adding a candidate model does not break the test
        assert len(full) == len(embedding.MODELS) * len(embedding.BACKENDS)
        specs = [c for c in full.values() if c is not None]
        assert len(specs) == len(full)
        assert {c.backend for c in specs} == set(embedding.BACKENDS)
        assert {c.dimension for c in specs} == {d for _, d in embedding.MODELS.values()}
        large = full["granite-embedding-english-r2/onnx"]
        assert large is not None
        assert large.id == "ibm-granite/granite-embedding-english-r2"
        assert large.model_path == str(tmp_path / "granite-embedding-english-r2")

        quick = embedding.candidates(quick=True)
        assert len(quick) == len(embedding.MODELS)
        assert {c.backend for c in quick.values() if c is not None} == {"torch"}

        stand_in = embedding.candidates(stand_in=True)
        assert list(stand_in) == [embedding.STAND_IN]
        assert stand_in[embedding.STAND_IN] is None

    def test_reranker_matrix(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.setenv("BENCH_MODELS_DIR", str(tmp_path))
        full = reranker.candidates()
        assert len(full) == 12
        assert sorted({c.candidate_k for c in full.values()}) == [15, 20, 25]
        ce = full["onnx@k25"]
        assert ce.provider == "onnx"
        assert ce.candidate_k == 25
        assert ce.model is not None and ce.model.backend == "onnx"
        assert ce.model_path == str(tmp_path / "ms-marco-MiniLM-L6-v2")
        assert full["lexical@k15"].model_path is None
        assert {c.provider for c in reranker.candidates(stand_in=True).values()} == {
            "lexical",
            "disabled",
        }

    def test_sample_documents_cycles_to_batch_size(self) -> None:
        from benchmark.evaluation.golden import GoldenSet

        docs = embedding.sample_documents(GoldenSet.load(embedding.GOLDEN))
        assert len(docs) == embedding.BATCH_DOCUMENTS
        assert all(len(d) > 40 for d in docs)

    def test_candidate_overrides_swap_only_the_encoder(self) -> None:
        """A candidate replaces the frozen encoder for one container and nothing else: the
        benchmark stand-ins for the stores stay exactly what ``bench_overrides()`` says."""
        from benchmark.env import bench_overrides

        from memory_service.config.constants import DenseModel

        spec = DenseModel(id="m", model_path="/w/m", backend="onnx", dimension=768)
        with_spec = embedding.candidate_overrides(spec)
        assert with_spec.dense_model == spec and with_spec.embedding is None
        stand_in = embedding.candidate_overrides(None)
        assert stand_in.embedding == "hash" and stand_in.dense_model is None
        assert stand_in.embedding_dimension == embedding.STAND_IN_DIMENSION
        base = bench_overrides()
        for name in ("cache", "tasks", "search", "authorization", "blob", "nli"):
            assert getattr(with_spec, name) == getattr(base, name) == getattr(stand_in, name)


@pytest.mark.integration
@pytest.mark.skipif(not PG_AVAILABLE, reason="PostgreSQL not reachable at MEMORY_TEST_DATABASE_URL")
def test_embedding_benchmark_stand_in_end_to_end(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("BENCH_SEARCH", "memory")
    out = tmp_path / "embedding.json"
    embedding.main(["--quick", "--stand-in", "--copies", "1", "--batches", "1", "--out", str(out)])

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["mode"] == {"quick": True, "stand_in": True}
    assert payload["providers"]["representative"] is False
    row = payload["candidates"][embedding.STAND_IN]
    assert row["fingerprint"] == "hash-v1-d64"
    assert row["dimension"] == 64
    assert row["indexed_points"] > 0
    assert row["embed_ms"]["batch_size"] == embedding.BATCH_DOCUMENTS
    assert row["embed_ms"]["batches"] == 1
    assert 0.0 <= row["quality"]["critical_recall_at_k"] <= 1.0
    assert row["quality_per_ms"] >= 0.0
    assert row["representative"] is False
    assert payload["verdict"]["default"] == embedding.STAND_IN
    assert "git_commit" in payload["provenance"]

    printed = capsys.readouterr().out
    assert f"wrote {out}" in printed
    assert embedding.STAND_IN in printed
    assert f"default -> {embedding.STAND_IN}" in printed
