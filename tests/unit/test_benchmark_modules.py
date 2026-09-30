"""The embedding benchmark module: candidate matrices, verdict selection on
synthetic rows, and one end-to-end run of the embedding benchmark with the hash stand-in."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from benchmark import embedding

from tests.conftest import DB_URL, PG_AVAILABLE


def _row(recall: float, egr: float, **extra: object) -> dict[str, object]:
    return {
        "quality": {"critical_recall_at_k": recall, "critical_evidence_group_recall": egr},
        **extra,
    }


def _emb_row(recall: float, egr: float, *, q_p95: float, dim: int) -> dict[str, object]:
    return _row(recall, egr, embed_ms={"query_p95": q_p95}, dimension=dim)


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


@pytest.mark.unit
class TestBenchEnvEmbeddingDefault:
    """The host default follows the weights: a ``make gates`` on a checkout without
    ``make models`` must run the labelled stand-in, not load the frozen encoder or die.
    The container path pins ``BENCH_EMBEDDING=frozen`` in the Makefile and is not a default."""

    @staticmethod
    def _roots(monkeypatch: pytest.MonkeyPatch, root: Path, *, weights: bool) -> None:
        from memory_service.config import constants

        if weights:
            # both dense spaces: the shipped stack encodes every record twice, so "the
            # weights are on disk" is only true when the multilingual set is there too
            for model in (constants.FROZEN_MODELS.dense, constants.FROZEN_MODELS.dense_ml):
                (root / model.local_dir).mkdir(parents=True)
        monkeypatch.setattr(constants, "MODEL_ROOTS", (root,))
        monkeypatch.delenv("BENCH_EMBEDDING", raising=False)

    def test_frozen_when_the_dense_weights_resolve(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        from benchmark.env import BenchEnv, default_embedding

        self._roots(monkeypatch, tmp_path, weights=True)
        assert default_embedding() == "frozen"
        env = BenchEnv.from_environ()
        assert env.embedding == "frozen" and BenchEnv().embedding == "frozen"
        overrides = env.overrides()
        assert overrides.embedding is None and overrides.document_parser is None

    def test_hash_stand_in_when_no_root_holds_them(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        from benchmark.env import BenchEnv, default_embedding

        self._roots(monkeypatch, tmp_path, weights=False)
        assert default_embedding() == "hash"
        env = BenchEnv.from_environ()
        assert env.embedding == "hash" and BenchEnv().embedding == "hash"
        overrides = env.overrides()
        assert overrides.embedding == "hash" and overrides.embedding_dimension == 64
        assert overrides.document_parser == "builtin"

    def test_the_variable_wins_over_the_weights(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        from benchmark.env import BenchEnv

        self._roots(monkeypatch, tmp_path, weights=True)
        monkeypatch.setenv("BENCH_EMBEDDING", "hash")
        assert BenchEnv.from_environ().embedding == "hash"
        self._roots(monkeypatch, tmp_path / "empty", weights=False)
        monkeypatch.setenv("BENCH_EMBEDDING", "frozen")
        assert BenchEnv.from_environ().embedding == "frozen"


@pytest.mark.integration
@pytest.mark.skipif(not PG_AVAILABLE, reason="PostgreSQL not reachable at MEMORY_TEST_DATABASE_URL")
def test_embedding_benchmark_stand_in_end_to_end(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("BENCH_SEARCH", "memory")
    # The harness resets the whole schema, and ``benchmark.retrieval._settings`` takes its
    # database from ``MEMORY__DATABASE__URL`` - which the checked-in ``.env`` points at the
    # shared ``memory`` database, loaded into the environment before this module is imported.
    # So this test used to TRUNCATE the dev store on every `make unit`. Point it at the
    # suite's own database, which the suite already migrates and owns.
    monkeypatch.setenv("MEMORY__DATABASE__URL", DB_URL)
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
