from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import log2


@dataclass(frozen=True, slots=True)
class RetrievalMetrics:
    mrr_at_10: float
    recall_at_5: float
    ndcg_at_10: float


def _grade(relevance: Mapping[str, int], document_id: str) -> int:
    grade = relevance.get(document_id, 0)
    return max(0, int(grade))


def reciprocal_rank(
    ranked_document_ids: Sequence[str],
    relevance: Mapping[str, int],
    *,
    k: int = 10,
) -> float:
    if k <= 0:
        return 0.0
    for position, document_id in enumerate(ranked_document_ids[:k], start=1):
        if _grade(relevance, document_id) > 0:
            return 1.0 / position
    return 0.0


def recall_at_k(
    ranked_document_ids: Sequence[str],
    relevance: Mapping[str, int],
    *,
    k: int = 5,
) -> float:
    relevant = {document_id for document_id, grade in relevance.items() if int(grade) > 0}
    if not relevant or k <= 0:
        return 0.0
    retrieved = set(ranked_document_ids[:k])
    return len(relevant & retrieved) / len(relevant)


def dcg_at_k(
    ranked_document_ids: Sequence[str],
    relevance: Mapping[str, int],
    *,
    k: int = 10,
) -> float:
    if k <= 0:
        return 0.0
    score = 0.0
    for position, document_id in enumerate(ranked_document_ids[:k], start=1):
        grade = _grade(relevance, document_id)
        if grade <= 0:
            continue
        score += (2**grade - 1) / log2(position + 1)
    return score


def ndcg_at_k(
    ranked_document_ids: Sequence[str],
    relevance: Mapping[str, int],
    *,
    k: int = 10,
) -> float:
    if k <= 0:
        return 0.0
    ideal_grades = sorted((max(0, int(grade)) for grade in relevance.values()), reverse=True)
    ideal_score = 0.0
    for position, grade in enumerate(ideal_grades[:k], start=1):
        if grade <= 0:
            continue
        ideal_score += (2**grade - 1) / log2(position + 1)
    if ideal_score == 0.0:
        return 0.0
    return dcg_at_k(ranked_document_ids, relevance, k=k) / ideal_score


def evaluate_ranking(
    ranked_document_ids: Sequence[str],
    relevance: Mapping[str, int],
) -> RetrievalMetrics:
    return RetrievalMetrics(
        mrr_at_10=reciprocal_rank(ranked_document_ids, relevance, k=10),
        recall_at_5=recall_at_k(ranked_document_ids, relevance, k=5),
        ndcg_at_10=ndcg_at_k(ranked_document_ids, relevance, k=10),
    )


def macro_average(metrics: Sequence[RetrievalMetrics]) -> RetrievalMetrics:
    if not metrics:
        raise ValueError("at least one query metric is required")
    count = len(metrics)
    return RetrievalMetrics(
        mrr_at_10=sum(metric.mrr_at_10 for metric in metrics) / count,
        recall_at_5=sum(metric.recall_at_5 for metric in metrics) / count,
        ndcg_at_10=sum(metric.ndcg_at_10 for metric in metrics) / count,
    )
