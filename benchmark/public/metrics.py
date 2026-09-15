"""Metric implementations for the public benchmarks.

IR metrics follow trec_eval / pytrec_eval conventions (linear gain nDCG with a log2 discount,
recall over positive judgements, queries without a positive judgement are not scored).
LoCoMo scoring reproduces ``task_eval/evaluation.py`` from the LoCoMo repository (SQuAD-style
normalisation, Porter-stemmed token F1, comma-split multi-answer F1 for multi-hop questions,
abstention check for adversarial questions). Judge variance control: every LLM-judged metric
is a list of repeated runs summarised as mean ± sample standard deviation.
"""

from __future__ import annotations

import functools
import math
import re
import statistics
import string
from collections import Counter
from collections.abc import Iterable, Sequence
from typing import Any

# ------------------------------------------------------------------------- IR metrics


def ndcg_at_k(ranked: Sequence[str], rels: dict[str, int], k: int) -> float:
    dcg = 0.0
    for i, doc in enumerate(ranked[:k]):
        rel = rels.get(doc, 0)
        if rel > 0:
            dcg += rel / math.log2(i + 2)
    ideal = sorted((r for r in rels.values() if r > 0), reverse=True)[:k]
    idcg = sum(r / math.log2(i + 2) for i, r in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def recall_at_k(ranked: Sequence[str], rels: dict[str, int], k: int) -> float:
    positives = {d for d, r in rels.items() if r > 0}
    if not positives:
        return 0.0
    return len(positives.intersection(ranked[:k])) / len(positives)


def evaluate_run(
    run: dict[str, Sequence[str]],
    qrels: dict[str, dict[str, int]],
    *,
    ndcg_ks: Iterable[int] = (10,),
    recall_ks: Iterable[int] = (20, 100),
) -> dict[str, float]:
    """Mean metrics over the scorable queries (those with a positive judgement); a query
    missing from ``run`` counts as an empty ranking, like trec_eval."""
    scored = [q for q, rels in qrels.items() if any(r > 0 for r in rels.values())]
    out: dict[str, float] = {}
    if not scored:
        return out
    for k in ndcg_ks:
        out[f"ndcg@{k}"] = round(
            sum(ndcg_at_k(run.get(q, ()), qrels[q], k) for q in scored) / len(scored), 4
        )
    for k in recall_ks:
        out[f"recall@{k}"] = round(
            sum(recall_at_k(run.get(q, ()), qrels[q], k) for q in scored) / len(scored), 4
        )
    out["queries"] = len(scored)
    return out


def doc_ranking(chunk_doc_ids: Iterable[str | None]) -> list[str]:
    """Chunk hits in rank order -> document ranking (a document ranks at its best chunk)."""
    seen: set[str] = set()
    out: list[str] = []
    for doc in chunk_doc_ids:
        if doc is None or doc in seen:
            continue
        seen.add(doc)
        out.append(doc)
    return out


# ------------------------------------------------------------------------ LoCoMo QA

_ARTICLES = re.compile(r"\b(a|an|the|and)\b")


@functools.cache
def _stemmer() -> Any:
    try:
        from nltk.stem import PorterStemmer  # eval extra
    except ImportError:  # pragma: no cover - the eval extra installs nltk
        return None
    return PorterStemmer()


def _stem(word: str) -> str:
    stemmer = _stemmer()
    return word if stemmer is None else str(stemmer.stem(word))


def stemmer_name() -> str:
    try:
        import nltk  # noqa: F401  # eval extra
    except ImportError:  # pragma: no cover
        return "none (nltk missing: F1 is unstemmed)"
    return "nltk PorterStemmer"


def normalize_answer(text: str) -> str:
    text = text.replace(",", "")
    text = text.lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = _ARTICLES.sub(" ", text)
    return " ".join(text.split())


def _tokens(text: str) -> list[str]:
    return [_stem(w) for w in normalize_answer(text).split()]


def locomo_f1_score(prediction: str, ground_truth: str) -> float:
    pred, truth = _tokens(prediction), _tokens(ground_truth)
    common = Counter(pred) & Counter(truth)
    same = sum(common.values())
    if same == 0:
        return 0.0
    precision = same / len(pred)
    recall = same / len(truth)
    return 2 * precision * recall / (precision + recall)


def locomo_multi_f1(prediction: str, ground_truth: str) -> float:
    """Multi-hop questions: comma-separated sub-answers, best F1 per gold part, averaged."""
    preds = [p.strip() for p in prediction.split(",")]
    truths = [g.strip() for g in ground_truth.split(",")]
    return statistics.fmean(max(locomo_f1_score(p, g) for p in preds) for g in truths)


def locomo_abstained(prediction: str) -> bool:
    low = prediction.lower()
    return "no information available" in low or "not mentioned" in low


def bleu1(prediction: str, ground_truth: str) -> float:
    """Unigram BLEU with brevity penalty on the normalised, stemmed tokens (smoothed by
    adding one to the clipped count and the denominator, so short answers do not zero out)."""
    pred, truth = _tokens(prediction), _tokens(ground_truth)
    if not pred or not truth:
        return 0.0
    clipped = sum((Counter(pred) & Counter(truth)).values())
    precision = (clipped + 1) / (len(pred) + 1)
    brevity = 1.0 if len(pred) >= len(truth) else math.exp(1 - len(truth) / len(pred))
    return brevity * precision


def locomo_score(prediction: str, answer: str, category: str) -> dict[str, float]:
    """The benchmark's own rule per category: 1 multi-hop (split F1), 2 temporal, 3 open-domain
    (first ';'-alternative), 4 single-hop (token F1), 5 adversarial (abstention)."""
    answer = str(answer)
    if category == "3":
        answer = answer.split(";")[0].strip()
    if category == "5":
        score = 1.0 if locomo_abstained(prediction) else 0.0
        return {"f1": score, "bleu1": score}
    if category == "1":
        f1 = locomo_multi_f1(prediction, answer)
    else:
        f1 = locomo_f1_score(prediction, answer)
    return {"f1": f1, "bleu1": bleu1(prediction, answer)}


# ------------------------------------------------------------------- judge variance


def judge_summary(runs: Sequence[Sequence[float]]) -> dict[str, Any]:
    """Per-run accuracy of repeated judgements -> mean ± sample sd across runs."""
    per_run = [round(statistics.fmean(r), 4) if r else 0.0 for r in runs]
    mean = round(statistics.fmean(per_run), 4) if per_run else 0.0
    sd = round(statistics.stdev(per_run), 4) if len(per_run) > 1 else 0.0
    return {"mean": mean, "sd": sd, "runs": len(per_run), "per_run": per_run}


def grouped_judge_summary(
    labels: Sequence[Sequence[bool]], groups: Sequence[str]
) -> dict[str, dict[str, Any]]:
    """``labels[run][question]`` -> overall and per-group mean ± sd across runs."""
    out = {"overall": judge_summary([[1.0 if x else 0.0 for x in run] for run in labels])}
    for group in sorted(set(groups)):
        idx = [i for i, g in enumerate(groups) if g == group]
        out[group] = judge_summary([[1.0 if run[i] else 0.0 for i in idx] for run in labels])
        out[group]["questions"] = len(idx)
    out["overall"]["questions"] = len(groups)
    return out
