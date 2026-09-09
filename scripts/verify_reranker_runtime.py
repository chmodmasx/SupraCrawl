from __future__ import annotations

import asyncio
import math

from supracrawl.config import Settings
from supracrawl.embeddings import DenseEmbedder
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


async def _verify_dense_runtime(settings: Settings) -> None:
    embedder = DenseEmbedder(
        model_name=settings.dense_model_name,
        dimension=settings.dense_dimension,
        query_prefix=settings.dense_query_prefix,
        passage_prefix=settings.dense_passage_prefix,
    )
    query_vector = await embedder.embed_query("hybrid retrieval for autonomous agents")
    passage_vectors = await embedder.embed_passages(
        [
            "passage: hybrid retrieval combines lexical and semantic evidence",
            "passage: local embeddings keep agent search self-hosted",
        ]
    )

    if len(query_vector) != settings.dense_dimension:
        raise RuntimeError("dense query vector dimension changed")
    if len(passage_vectors) != 2:
        raise RuntimeError("dense passage runtime returned the wrong vector count")
    for vector in [query_vector, *passage_vectors]:
        if len(vector) != settings.dense_dimension:
            raise RuntimeError("dense passage vector dimension changed")
        if not all(math.isfinite(value) for value in vector):
            raise RuntimeError("dense runtime returned non-finite values")
    print(f"Dense runtime verified: {settings.dense_model_name}")


async def _verify_reranker_runtime() -> None:
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

    if len(reranked) != 10:
        raise RuntimeError("reranker returned the wrong candidate count")
    ids = [item["metadata"]["document_id"] for item in reranked]
    if set(ids[:5]) != {f"doc-{index}" for index in range(1, 6)}:
        raise RuntimeError("reranker changed protected top-5 membership")
    if set(ids[5:]) != {f"doc-{index}" for index in range(6, 11)}:
        raise RuntimeError("reranker changed frozen tail membership")
    if [item["position"] for item in reranked] != list(range(1, 11)):
        raise RuntimeError("reranker returned non-contiguous positions")

    first_stage_positions: set[int] = set()
    for item in reranked:
        metadata = item["metadata"]
        document_id = metadata["document_id"]
        if item["score"] != original_scores[document_id]:
            raise RuntimeError("reranker changed the certified first-stage score")
        if metadata["reranker_model"] != RERANKER_MODEL_REPO:
            raise RuntimeError("reranker model provenance changed")
        if metadata["reranker_revision"] != RERANKER_REVISION:
            raise RuntimeError("reranker revision provenance changed")
        if metadata["reranker_strategy"] != RERANKER_STRATEGY:
            raise RuntimeError("reranker strategy provenance changed")
        if not math.isfinite(float(metadata["reranker_score"])):
            raise RuntimeError("reranker returned a non-finite score")
        first_stage_positions.add(int(metadata["first_stage_position"]))

    if first_stage_positions != set(range(1, 11)):
        raise RuntimeError("reranker first-stage provenance is incomplete")
    print("Frozen reranker runtime load and inference verified")


async def _verify() -> None:
    settings = Settings()
    await _verify_dense_runtime(settings)
    await _verify_reranker_runtime()


if __name__ == "__main__":
    asyncio.run(_verify())
