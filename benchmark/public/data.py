"""Dataset access for the public benchmarks.

Every set is downloaded once into the git-ignored cache (``.bench_data/``, override with
``BENCH_DATA_DIR``) and parsed into one of two plain shapes: an IR collection (BEIR) or a
list of conversations with questions (LongMemEval, LoCoMo). The parsers also read the tiny
synthetic fixtures under ``tests/fixtures/public`` — the real files never enter the repo.
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parents[2]
CACHE = Path(os.environ.get("BENCH_DATA_DIR", str(ROOT / ".bench_data")))

BEIR_DATASETS = ("nfcorpus", "scifact", "fiqa")
BEIR_HF_PREFIX = "BeIR"
LONGMEMEVAL_REPO = "xiaowu0162/longmemeval-cleaned"
LONGMEMEVAL_FILE = "longmemeval_s_cleaned.json"
LOCOMO_URL = "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"
BEAM_REPO = "Mohammadta/BEAM"
BEAM_STATUS = (
    "not run: BEAM (ICLR 2026, github.com/mohammadtavakoli78/BEAM) is public on the Hub as "
    f"{BEAM_REPO} (parquet splits 100K/500K/1M), but its official grader "
    "(src/evaluation/compute_metrics.py) scores through per-question-type rubric prompts "
    "plus sentence-transformers fact alignment and is not reproduced here; a loader without "
    "the benchmark's own grader would not give comparable numbers"
)

LOCOMO_DATE = "%I:%M %p on %d %B, %Y"
LONGMEMEVAL_DATE = "%Y/%m/%d (%a) %H:%M"


# ------------------------------------------------------------------------------ BEIR


@dataclass
class BeirDataset:
    name: str
    corpus: dict[str, tuple[str, str]]
    queries: dict[str, str]
    qrels: dict[str, dict[str, int]]
    source: str = ""

    @property
    def scored_queries(self) -> list[str]:
        """Queries with at least one positive judgement, in a stable order."""
        return sorted(q for q, rels in self.qrels.items() if any(v > 0 for v in rels.values()))

    def subset(
        self, *, max_docs: int | None = None, max_queries: int | None = None, seed: int = 7
    ) -> BeirDataset:
        """Smoke-run cap: keep the sampled queries' positive documents first, then fill the
        document budget with random distractors, so every kept query stays scorable."""
        rng = random.Random(seed)
        queries = self.scored_queries
        if max_queries is not None and len(queries) > max_queries:
            queries = sorted(rng.sample(queries, max_queries))
        positives: list[str] = []
        seen: set[str] = set()
        for q in queries:
            for doc, rel in sorted(self.qrels[q].items()):
                if rel > 0 and doc in self.corpus and doc not in seen:
                    positives.append(doc)
                    seen.add(doc)
        if max_docs is not None and len(positives) > max_docs:
            positives = positives[:max_docs]
            seen = set(positives)
        keep = list(positives)
        if max_docs is not None and len(keep) < max_docs:
            pool = sorted(d for d in self.corpus if d not in seen)
            keep.extend(rng.sample(pool, min(max_docs - len(keep), len(pool))))
        elif max_docs is None:
            keep = sorted(self.corpus)
        kept = set(keep)
        qrels = {
            q: {d: r for d, r in self.qrels[q].items() if d in kept}
            for q in queries
            if q in self.queries
        }
        qrels = {q: rels for q, rels in qrels.items() if any(v > 0 for v in rels.values())}
        return BeirDataset(
            name=self.name,
            corpus={d: self.corpus[d] for d in keep},
            queries={q: self.queries[q] for q in qrels},
            qrels=qrels,
            source=self.source,
        )


def read_beir_dir(path: Path, *, split: str = "test") -> BeirDataset:
    """BEIR's native layout: ``corpus.jsonl``, ``queries.jsonl``, ``qrels/<split>.tsv``."""
    corpus: dict[str, tuple[str, str]] = {}
    with (path / "corpus.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                row = json.loads(line)
                corpus[str(row["_id"])] = (str(row.get("title") or ""), str(row.get("text") or ""))
    queries: dict[str, str] = {}
    with (path / "queries.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                row = json.loads(line)
                queries[str(row["_id"])] = str(row["text"])
    qrels: dict[str, dict[str, int]] = {}
    with (path / "qrels" / f"{split}.tsv").open(encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3 or (i == 0 and parts[0] in ("query-id", "query_id")):
                continue
            qrels.setdefault(parts[0], {})[parts[1]] = int(float(parts[2]))
    return BeirDataset(name=path.name, corpus=corpus, queries=queries, qrels=qrels, source=str(path))


def _download_beir(name: str, target: Path) -> None:
    import datasets  # eval extra

    datasets.disable_progress_bars()
    hf_cache = str(CACHE / "hf")
    repo = f"{BEIR_HF_PREFIX}/{name}"
    corpus = datasets.load_dataset(repo, "corpus", split="corpus", cache_dir=hf_cache)
    queries = datasets.load_dataset(repo, "queries", split="queries", cache_dir=hf_cache)
    qrels = datasets.load_dataset(f"{repo}-qrels", split="test", cache_dir=hf_cache)
    target.mkdir(parents=True, exist_ok=True)
    (target / "qrels").mkdir(exist_ok=True)
    with (target / "corpus.jsonl").open("w", encoding="utf-8") as fh:
        for row in corpus:
            fh.write(
                json.dumps(
                    {"_id": row["_id"], "title": row.get("title") or "", "text": row["text"]}
                )
                + "\n"
            )
    with (target / "queries.jsonl").open("w", encoding="utf-8") as fh:
        for row in queries:
            fh.write(json.dumps({"_id": row["_id"], "text": row["text"]}) + "\n")
    with (target / "qrels" / "test.tsv").open("w", encoding="utf-8") as fh:
        fh.write("query-id\tcorpus-id\tscore\n")
        for row in qrels:
            fh.write(f"{row['query-id']}\t{row['corpus-id']}\t{row['score']}\n")


def load_beir(name: str, cache: Path = CACHE) -> BeirDataset:
    if name not in BEIR_DATASETS:
        raise ValueError(f"unknown BEIR subset {name!r}; expected one of {BEIR_DATASETS}")
    target = cache / "beir" / name
    if not (target / "qrels" / "test.tsv").is_file():
        _download_beir(name, target)
    data = read_beir_dir(target)
    data.source = f"hf://datasets/{BEIR_HF_PREFIX}/{name} (+ -qrels, split=test)"
    return data


# ------------------------------------------------------------------- conversations


@dataclass
class Turn:
    role: str
    content: str
    speaker: str | None = None


@dataclass
class Session:
    session_id: str
    date: datetime | None
    turns: list[Turn] = field(default_factory=list)


@dataclass
class Question:
    question_id: str
    text: str
    answer: str
    category: str
    date: str | None = None
    abstention: bool = False
    evidence: list[str] = field(default_factory=list)


@dataclass
class Conversation:
    conversation_id: str
    sessions: list[Session]
    questions: list[Question]


@dataclass
class MemoryDataset:
    name: str
    conversations: list[Conversation]
    source: str = ""
    size_bytes: int = 0

    @property
    def questions(self) -> list[tuple[Conversation, Question]]:
        return [(c, q) for c in self.conversations for q in c.questions]

    def limited(
        self, *, max_conversations: int | None = None, max_questions: int | None = None, seed: int = 7
    ) -> MemoryDataset:
        convs = list(self.conversations)
        if max_conversations is not None:
            convs = convs[:max_conversations]
        if max_questions is not None:
            total = sum(len(c.questions) for c in convs)
            if total > max_questions:
                rng = random.Random(seed)
                keep = set(
                    rng.sample(
                        [(c.conversation_id, q.question_id) for c in convs for q in c.questions],
                        max_questions,
                    )
                )
                convs = [
                    Conversation(
                        c.conversation_id,
                        c.sessions,
                        [q for q in c.questions if (c.conversation_id, q.question_id) in keep],
                    )
                    for c in convs
                ]
                convs = [c for c in convs if c.questions]
        return MemoryDataset(self.name, convs, self.source, self.size_bytes)


def parse_date(value: Any, fmt: str) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        return datetime.strptime(value.strip(), fmt).replace(tzinfo=UTC)
    except ValueError:
        return None


def parse_longmemeval(entries: list[dict[str, Any]]) -> list[Conversation]:
    """One conversation per question: its haystack sessions (dated) and the single question."""
    out: list[Conversation] = []
    for entry in entries:
        qid = str(entry["question_id"])
        ids = entry.get("haystack_session_ids") or []
        dates = entry.get("haystack_dates") or []
        sessions: list[Session] = []
        for i, turns in enumerate(entry.get("haystack_sessions") or []):
            sid = str(ids[i]) if i < len(ids) else f"{qid}-s{i}"
            date = parse_date(dates[i], LONGMEMEVAL_DATE) if i < len(dates) else None
            sessions.append(
                Session(
                    session_id=sid,
                    date=date,
                    turns=[
                        Turn(role=str(t.get("role") or "user"), content=str(t.get("content") or ""))
                        for t in turns
                        if t.get("content")
                    ],
                )
            )
        question = Question(
            question_id=qid,
            text=str(entry["question"]),
            answer=str(entry.get("answer") or ""),
            category=str(entry.get("question_type") or "unknown"),
            date=entry.get("question_date"),
            abstention="_abs" in qid,
            evidence=[str(x) for x in entry.get("answer_session_ids") or []],
        )
        out.append(Conversation(qid, sessions, [question]))
    return out


def parse_locomo(samples: list[dict[str, Any]]) -> list[Conversation]:
    """One conversation per sample: dated sessions between two named speakers, the QA list
    (categories 1-5; adversarial questions carry ``adversarial_answer``)."""
    out: list[Conversation] = []
    for sample in samples:
        conv = sample["conversation"]
        sessions: list[Session] = []
        n = 1
        while f"session_{n}" in conv:
            turns = []
            for t in conv[f"session_{n}"]:
                text = str(t.get("text") or "")
                caption = t.get("blip_caption")
                if caption:
                    text = f"{text} [shared an image: {caption}]".strip()
                turns.append(Turn(role="user", content=text, speaker=t.get("speaker")))
            sessions.append(
                Session(
                    session_id=f"session_{n}",
                    date=parse_date(conv.get(f"session_{n}_date_time"), LOCOMO_DATE),
                    turns=turns,
                )
            )
            n += 1
        questions = []
        for i, qa in enumerate(sample.get("qa") or []):
            category = str(qa.get("category"))
            answer = qa.get("answer")
            if answer is None:
                answer = qa.get("adversarial_answer", "")
            questions.append(
                Question(
                    question_id=f"{sample['sample_id']}-q{i}",
                    text=str(qa["question"]),
                    answer=str(answer),
                    category=category,
                    abstention=category == "5",
                    evidence=[str(x) for x in qa.get("evidence") or []],
                )
            )
        out.append(Conversation(str(sample["sample_id"]), sessions, questions))
    return out


def load_longmemeval(path: Path | None = None, cache: Path = CACHE) -> MemoryDataset:
    if path is None:
        from huggingface_hub import hf_hub_download

        target = cache / "longmemeval"
        target.mkdir(parents=True, exist_ok=True)
        path = Path(
            hf_hub_download(
                LONGMEMEVAL_REPO, LONGMEMEVAL_FILE, repo_type="dataset", local_dir=str(target)
            )
        )
        source = f"hf://datasets/{LONGMEMEVAL_REPO}/{LONGMEMEVAL_FILE}"
    else:
        source = str(path)
    entries = json.loads(path.read_text(encoding="utf-8"))
    return MemoryDataset("longmemeval", parse_longmemeval(entries), source, path.stat().st_size)


def load_locomo(path: Path | None = None, cache: Path = CACHE) -> MemoryDataset:
    if path is None:
        path = cache / "locomo" / "locomo10.json"
        if not path.is_file():
            path.parent.mkdir(parents=True, exist_ok=True)
            resp = httpx.get(LOCOMO_URL, timeout=120, follow_redirects=True)
            resp.raise_for_status()
            path.write_bytes(resp.content)
        source = LOCOMO_URL
    else:
        source = str(path)
    samples = json.loads(path.read_text(encoding="utf-8"))
    return MemoryDataset("locomo", parse_locomo(samples), source, path.stat().st_size)
