import math

import pytest

from supracrawl.evaluation import (
    RetrievalMetrics,
    dcg_at_k,
    evaluate_ranking,
    macro_average,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
)


def test_reciprocal_rank_uses_first_relevant_result() -> None:
    ranked = ["noise", "secondary", "target"]
    relevance = {"target": 3, "secondary": 1}
    assert reciprocal_rank(ranked, relevance, k=10) == 0.5


def test_recall_at_k_counts_unique_relevant_documents() -> None:
    ranked = ["a", "a", "b", "noise"]
    relevance = {"a": 3, "b": 2, "c": 1}
    assert recall_at_k(ranked, relevance, k=3) == pytest.approx(2 / 3)


def test_dcg_and_ndcg_use_graded_relevance() -> None:
    ranked = ["medium", "best", "noise"]
    relevance = {"best": 3, "medium": 1}

    expected_dcg = (2**1 - 1) / math.log2(2) + (2**3 - 1) / math.log2(3)
    ideal_dcg = (2**3 - 1) / math.log2(2) + (2**1 - 1) / math.log2(3)

    assert dcg_at_k(ranked, relevance, k=10) == pytest.approx(expected_dcg)
    assert ndcg_at_k(ranked, relevance, k=10) == pytest.approx(expected_dcg / ideal_dcg)


def test_empty_or_non_positive_relevance_is_zero() -> None:
    ranked = ["a", "b"]
    relevance = {"a": 0, "b": -1}
    assert reciprocal_rank(ranked, relevance) == 0.0
    assert recall_at_k(ranked, relevance) == 0.0
    assert ndcg_at_k(ranked, relevance) == 0.0


def test_evaluate_ranking_uses_phase3_cutoffs() -> None:
    metric = evaluate_ranking(["noise", "target"], {"target": 3})
    assert metric == RetrievalMetrics(
        mrr_at_10=0.5,
        recall_at_5=1.0,
        ndcg_at_10=pytest.approx(1 / math.log2(3)),
    )


def test_macro_average_requires_queries_and_averages_each_metric() -> None:
    metrics = [
        RetrievalMetrics(mrr_at_10=1.0, recall_at_5=1.0, ndcg_at_10=1.0),
        RetrievalMetrics(mrr_at_10=0.5, recall_at_5=0.0, ndcg_at_10=0.25),
    ]
    aggregate = macro_average(metrics)
    assert aggregate.mrr_at_10 == pytest.approx(0.75)
    assert aggregate.recall_at_5 == pytest.approx(0.5)
    assert aggregate.ndcg_at_10 == pytest.approx(0.625)

    with pytest.raises(ValueError, match="at least one query"):
        macro_average([])
