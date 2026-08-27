from __future__ import annotations

from collections.abc import Sequence


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[str]],
    *,
    k: int = 60,
    limit: int = 10,
) -> list[str]:
    """Fuse ranked document-id lists using reciprocal rank fusion.

    Ties are resolved deterministically by the best individual rank and then
    document id. Empty rankings are allowed; invalid k/limit values are not.
    """
    if k <= 0:
        raise ValueError("RRF k must be positive")
    if limit <= 0:
        raise ValueError("RRF limit must be positive")

    scores: dict[str, float] = {}
    best_rank: dict[str, int] = {}
    for ranking in rankings:
        seen_in_ranking: set[str] = set()
        for rank, document_id in enumerate(ranking, start=1):
            if not document_id or document_id in seen_in_ranking:
                continue
            seen_in_ranking.add(document_id)
            scores[document_id] = scores.get(document_id, 0.0) + 1.0 / (k + rank)
            best_rank[document_id] = min(best_rank.get(document_id, rank), rank)

    ordered = sorted(
        scores,
        key=lambda document_id: (
            -scores[document_id],
            best_rank[document_id],
            document_id,
        ),
    )
    return ordered[:limit]
