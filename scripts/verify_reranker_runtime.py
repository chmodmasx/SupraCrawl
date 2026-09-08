from __future__ import annotations

import asyncio
import math

from supracrawl.reranking import (
    RERANKER_MODEL_REPO,
    RERANKER_REVISION,
    RERANKER_STRATEGY,
    LocalCrossEncoderReranker,
)


def _candidate(index: int) -> dict[str, object]:
    return {
        "title": f"SupraCrawl retrieval candidate {index}",
        "url": f"https://example.com/candidate-{index}",
        "description": (
            "Hybrid web retrieval, indexing, ranking, and agent search result "
            f"candidate number {index}."
        ),
        "position": index,
        "score": 1.0 / index,
        "metadata": {"document_id": f"doc-{index}"},
    }


async def _verify() -> None:
    runtime = LocalCrossEncoderReranker()
    await runtime.warmup()

    original = [_candidate(index) for index in range(1, 11)]
    original_scores = {
        item["metadata"]["document_id"]: item["score"]  # type: ignore[index]
        for item in original
    }
    reranked = await runtime.rerank(
        "hybrid web retrieval for autonomous agents",
        original,
    )

    assert len(reranked) == 10
    ids = [item["metadata"]["document_id"] for item in reranked]
    assert set(ids[:5]) == {f"doc-{index}" for index in range(1, 6)}
    assert set(ids[5:]) == {f"doc-{index}" for index in range(6, 11)}
    assert [item["position"] for item in reranked] == list(range(1, 11))

    first_stage_positions: set[int] = set()
    for item in reranked:
        metadata = item["metadata"]
        document_id = metadata["document_id"]
        assert item["score"] == original_scores[document_id]
        assert metadata["reranker_model"] == RERANKER_MODEL_REPO
        assert metadata["reranker_revision"] == RERANKER_REVISION
        assert metadata["reranker_strategy"] == RERANKER_STRATEGY
        assert math.isfinite(float(metadata["reranker_score"]))
        first_stage_positions.add(int(metadata["first_stage_position"]))

    assert first_stage_positions == set(range(1, 11))
    print("Frozen reranker runtime load and inference verified")


if __name__ == "__main__":
    asyncio.run(_verify())
