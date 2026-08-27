from __future__ import annotations

import pytest

from supracrawl.fusion import reciprocal_rank_fusion


def test_rrf_rewards_documents_seen_by_multiple_rankers() -> None:
    ranking = reciprocal_rank_fusion(
        [["lexical", "shared", "only-bm25"], ["dense", "shared", "only-dense"]],
        k=60,
        limit=5,
    )

    assert ranking[0] == "shared"
    assert set(ranking) == {"lexical", "shared", "only-bm25", "dense", "only-dense"}


def test_rrf_is_deterministic_for_equal_scores() -> None:
    ranking = reciprocal_rank_fusion([["b"], ["a"]], k=60, limit=10)

    assert ranking == ["a", "b"]


def test_rrf_deduplicates_repeated_document_within_one_ranking() -> None:
    ranking = reciprocal_rank_fusion([["a", "a", "b"], ["b"]], k=60, limit=10)

    assert ranking == ["b", "a"]


def test_rrf_allows_empty_rankings() -> None:
    assert reciprocal_rank_fusion([[], []], k=60, limit=10) == []


@pytest.mark.parametrize("k", [0, -1])
def test_rrf_rejects_nonpositive_k(k: int) -> None:
    with pytest.raises(ValueError, match="RRF k must be positive"):
        reciprocal_rank_fusion([["a"]], k=k)


@pytest.mark.parametrize("limit", [0, -1])
def test_rrf_rejects_nonpositive_limit(limit: int) -> None:
    with pytest.raises(ValueError, match="RRF limit must be positive"):
        reciprocal_rank_fusion([["a"]], limit=limit)
