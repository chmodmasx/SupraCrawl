from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any

import httpx
from verify_real_vector_retrieval import _clear_indices, _seed_corpus_with_vectors
from verify_retrieval_baseline import _load_jsonl

from supracrawl.config import Settings
from supracrawl.embeddings import DenseEmbedder
from supracrawl.search import OpenSearchStore

ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = ROOT / "evaluation" / "phase3e_policy.json"
EXACT_CORPUS_PATH = ROOT / "evaluation" / "phase3c_exact_corpus.jsonl"
EXACT_QUERIES_PATH = ROOT / "evaluation" / "phase3c_exact_queries.jsonl"


def _policy() -> dict[str, Any]:
    payload = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError("Phase 3E policy must be a JSON object")
    return payload


async def _set_read_block(
    client: httpx.AsyncClient,
    index_name: str,
    blocked: bool,
) -> None:
    response = await client.put(
        f"/{index_name}/_settings",
        json={"index.blocks.read": blocked},
    )
    if response.status_code >= 400:
        raise RuntimeError(
            f"unable to set read block={blocked} on {index_name}: "
            f"HTTP {response.status_code} {response.text[:200]}"
        )


async def _default_search(client: httpx.AsyncClient, query: str) -> httpx.Response:
    return await client.post(
        "/v1/search",
        json={"query": query, "limit": 5},
    )


def _assert_degraded(response: httpx.Response, reason_fragment: str) -> None:
    if response.status_code != 200:
        raise RuntimeError(
            f"expected degraded HTTP 200, got {response.status_code}: {response.text[:300]}"
        )
    body = response.json()
    if not isinstance(body, dict):
        raise RuntimeError("degraded response is not an object")
    if body.get("mode_requested") != "hybrid" or body.get("mode_used") != "bm25":
        raise RuntimeError("default hybrid failure did not degrade to BM25")
    if body.get("degraded") is not True:
        raise RuntimeError("default hybrid failure did not report degraded=true")
    reason = body.get("degradation_reason")
    if not isinstance(reason, str) or reason_fragment not in reason:
        raise RuntimeError(
            f"degradation reason {reason!r} does not contain {reason_fragment!r}"
        )
    results = body.get("results")
    if not isinstance(results, list) or not results:
        raise RuntimeError("default hybrid degradation returned no lexical results")


def _assert_healthy_hybrid(response: httpx.Response) -> None:
    if response.status_code != 200:
        raise RuntimeError(
            f"healthy default search returned HTTP {response.status_code}: {response.text[:300]}"
        )
    body = response.json()
    if not isinstance(body, dict):
        raise RuntimeError("healthy default response is not an object")
    if body.get("mode_requested") != "hybrid" or body.get("mode_used") != "hybrid":
        raise RuntimeError("healthy omitted-mode request did not use hybrid retrieval")
    if body.get("degraded") is not False:
        raise RuntimeError("healthy omitted-mode request unexpectedly degraded")
    results = body.get("results")
    if not isinstance(results, list) or not results:
        raise RuntimeError("healthy omitted-mode request returned no results")


async def _seed_fault_fixture(
    store: OpenSearchStore,
    embedder: DenseEmbedder,
) -> str:
    corpus = _load_jsonl(EXACT_CORPUS_PATH)
    queries = _load_jsonl(EXACT_QUERIES_PATH)
    target_query = next(item for item in queries if item["id"] == "q21-current-sha")
    query = target_query["query"]
    target_id = next(iter(target_query["relevance"]))
    target = next(document for document in corpus if document["id"] == target_id)

    await _clear_indices(store, store.settings)
    vector_chunks = await _seed_corpus_with_vectors(store, embedder, [target])
    if vector_chunks <= 0:
        raise RuntimeError("Phase 3E fault fixture wrote no vectors")
    return query


async def _verify_hash_validation_failure(
    api: httpx.AsyncClient,
    opensearch: httpx.AsyncClient,
    settings: Settings,
    query: str,
) -> None:
    index_name = settings.opensearch_documents_index
    await _set_read_block(opensearch, index_name, True)
    try:
        response = await _default_search(api, query)
        _assert_degraded(response, "current-content validation")
    finally:
        await _set_read_block(opensearch, index_name, False)
    print("phase3e_default_hash_validation_failure_degrades=PASS")


async def _verify_lexical_failure(
    api: httpx.AsyncClient,
    opensearch: httpx.AsyncClient,
    settings: Settings,
    query: str,
) -> None:
    index_name = settings.opensearch_chunks_index
    await _set_read_block(opensearch, index_name, True)
    try:
        response = await _default_search(api, query)
        if response.status_code != 503:
            raise RuntimeError(
                f"default lexical failure returned HTTP {response.status_code}, expected 503"
            )
        body = response.json()
        if not isinstance(body, dict) or not isinstance(body.get("detail"), str):
            raise RuntimeError("default lexical 503 response is missing detail")
    finally:
        await _set_read_block(opensearch, index_name, False)
    print("phase3e_default_lexical_failure_503=PASS")


async def _verify_vector_failure(
    api: httpx.AsyncClient,
    opensearch: httpx.AsyncClient,
    settings: Settings,
    query: str,
) -> None:
    index_name = settings.opensearch_vector_chunks_index
    response = await opensearch.delete(f"/{index_name}")
    if response.status_code not in {200, 404}:
        raise RuntimeError(
            f"unable to delete vector index: HTTP {response.status_code} {response.text[:200]}"
        )
    response = await opensearch.put(
        f"/{index_name}",
        json={
            "settings": {"number_of_shards": 1, "number_of_replicas": 0},
            "mappings": {
                "dynamic": "strict",
                "properties": {"document_id": {"type": "keyword"}},
            },
        },
    )
    if response.status_code not in {200, 201}:
        raise RuntimeError(
            f"unable to create incompatible vector index: "
            f"HTTP {response.status_code} {response.text[:200]}"
        )

    response = await _default_search(api, query)
    _assert_degraded(response, "vector path unavailable")
    print("phase3e_default_vector_failure_degrades=PASS")


async def _verify_dense_disabled(api: httpx.AsyncClient, query: str) -> None:
    response = await _default_search(api, query)
    _assert_degraded(response, "disabled")
    print("phase3e_default_dense_disabled_degrades=PASS")


async def _verify_embedding_mismatch(api: httpx.AsyncClient, query: str) -> None:
    response = await _default_search(api, query)
    _assert_degraded(response, "dimension mismatch")
    print("phase3e_default_embedding_mismatch_degrades=PASS")


async def _run() -> None:
    policy = _policy()
    expected = policy["production_default_after_if_certified"]
    if expected != {"search_mode": "hybrid", "dense_enabled": True}:
        raise RuntimeError("Phase 3E default policy changed unexpectedly")

    api_url = os.environ.get("SUPRACRAWL_API_URL", "http://127.0.0.1:8080")
    opensearch_url = os.environ.get(
        "SUPRACRAWL_OPENSEARCH_URL",
        "http://127.0.0.1:9200",
    )
    dense_disabled_only = os.environ.get("PHASE3E_DENSE_DISABLED_ONLY") == "1"
    embedding_mismatch_only = os.environ.get("PHASE3E_EMBEDDING_MISMATCH_ONLY") == "1"

    settings = Settings(opensearch_url=opensearch_url)
    exact_queries = _load_jsonl(EXACT_QUERIES_PATH)
    query = next(item["query"] for item in exact_queries if item["id"] == "q21-current-sha")

    async with (
        httpx.AsyncClient(base_url=api_url, timeout=180.0) as api,
        httpx.AsyncClient(base_url=opensearch_url, timeout=30.0) as opensearch,
    ):
        if dense_disabled_only:
            await _verify_dense_disabled(api, query)
            return
        if embedding_mismatch_only:
            await _verify_embedding_mismatch(api, query)
            return

        store = OpenSearchStore(settings)
        embedder = DenseEmbedder(
            model_name=settings.dense_model_name,
            dimension=settings.dense_dimension,
            query_prefix=settings.dense_query_prefix,
            passage_prefix=settings.dense_passage_prefix,
        )
        try:
            query = await _seed_fault_fixture(store, embedder)
            healthy = await _default_search(api, query)
            _assert_healthy_hybrid(healthy)
            print("phase3e_default_fault_fixture_healthy=PASS")
            await _verify_hash_validation_failure(api, opensearch, settings, query)
            await _verify_lexical_failure(api, opensearch, settings, query)
            await _verify_vector_failure(api, opensearch, settings, query)
        finally:
            await store.close()

    print("Phase 3E default fault matrix: PASS")


if __name__ == "__main__":
    asyncio.run(_run())
