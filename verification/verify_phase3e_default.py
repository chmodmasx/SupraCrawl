from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any

import httpx
from verify_real_vector_retrieval import _clear_indices, _seed_corpus_with_vectors
from verify_retrieval_baseline import _load_jsonl, _percentile, _validate_fixture

from supracrawl.config import Settings
from supracrawl.embeddings import DenseEmbedder
from supracrawl.evaluation import RetrievalMetrics, evaluate_ranking, macro_average
from supracrawl.search import OpenSearchStore

ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = ROOT / "evaluation" / "phase3e_policy.json"
CORPUS_PATH = ROOT / "evaluation" / "corpus.jsonl"
QUERIES_PATH = ROOT / "evaluation" / "queries.jsonl"
EXACT_CORPUS_PATH = ROOT / "evaluation" / "phase3c_exact_corpus.jsonl"
EXACT_QUERIES_PATH = ROOT / "evaluation" / "phase3c_exact_queries.jsonl"
EXAMPLE_URL = "https://example.com/"


def _load_policy() -> dict[str, Any]:
    payload = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError("Phase 3E policy must be a JSON object")
    return payload


def _top_relevant_id(query: dict[str, Any]) -> str:
    relevance = query.get("relevance")
    if not isinstance(relevance, dict) or not relevance:
        raise RuntimeError(f"query {query.get('id')!r} has no relevance judgments")
    return min(relevance, key=lambda document_id: (-int(relevance[document_id]), document_id))


def _assert_hybrid_default(body: dict[str, Any]) -> None:
    if body.get("success") is not True:
        raise RuntimeError("default search response is not successful")
    if body.get("mode_requested") != "hybrid":
        raise RuntimeError("omitted-mode request did not request the configured hybrid default")
    if body.get("mode_used") != "hybrid":
        raise RuntimeError("healthy omitted-mode request did not use hybrid retrieval")
    if body.get("degraded") is not False:
        raise RuntimeError("healthy default hybrid request unexpectedly degraded")

    results = body.get("results")
    if not isinstance(results, list):
        raise RuntimeError("default search response has no results list")
    positions = [item.get("position") for item in results if isinstance(item, dict)]
    if positions != list(range(1, len(results) + 1)):
        raise RuntimeError("default hybrid result positions are not contiguous")
    for item in results:
        if not isinstance(item, dict):
            raise RuntimeError("default hybrid response contains an invalid result")
        metadata = item.get("metadata")
        if not isinstance(metadata, dict) or metadata.get("retrieval_mode") != "hybrid":
            raise RuntimeError("default hybrid result is missing retrieval provenance")


def _ranked_fixture_ids(
    body: dict[str, Any],
    url_to_id: dict[str, str],
) -> list[str]:
    results = body.get("results")
    if not isinstance(results, list):
        raise RuntimeError("search response has no results list")
    ranked: list[str] = []
    for result in results:
        if not isinstance(result, dict):
            raise RuntimeError("search response contains an invalid result")
        url = result.get("url")
        if not isinstance(url, str) or not url:
            raise RuntimeError("search result has no URL")
        ranked.append(url_to_id.get(url, url))
    return ranked


async def _post_search(
    client: httpx.AsyncClient,
    *,
    query: str,
    limit: int = 10,
    mode: str | None = None,
) -> tuple[dict[str, Any], float]:
    payload: dict[str, Any] = {"query": query, "limit": limit}
    if mode is not None:
        payload["mode"] = mode
    started = time.perf_counter()
    response = await client.post("/v1/search", json=payload)
    latency_ms = (time.perf_counter() - started) * 1000.0
    if response.status_code != 200:
        raise RuntimeError(
            f"search API returned HTTP {response.status_code}: {response.text[:300]}"
        )
    body = response.json()
    if not isinstance(body, dict):
        raise RuntimeError("search API returned non-object JSON")
    return body, latency_ms


async def _benchmark_default_api(
    client: httpx.AsyncClient,
    policy: dict[str, Any],
    queries: list[dict[str, Any]],
    url_to_id: dict[str, str],
    exact_query_ids: set[str],
) -> None:
    quality = policy["default_api_quality_gate"]
    warmup_query = next(query for query in queries if query["id"] == "q13-cross-language-rrf")
    warmup, _ = await _post_search(client, query=warmup_query["query"])
    _assert_hybrid_default(warmup)
    print("phase3e_default_hybrid_warmup=PASS")

    metrics: list[RetrievalMetrics] = []
    latencies: list[float] = []
    exact_top1_passes = 0
    semantic_ranks: dict[str, int] = {}

    for query in queries:
        body, latency_ms = await _post_search(client, query=query["query"])
        _assert_hybrid_default(body)
        ranked = _ranked_fixture_ids(body, url_to_id)
        metric = evaluate_ranking(ranked, query["relevance"])
        metrics.append(metric)
        latencies.append(latency_ms)

        query_id = query["id"]
        target = _top_relevant_id(query)
        try:
            target_rank = ranked.index(target) + 1
        except ValueError:
            target_rank = 0

        if query_id in exact_query_ids and target_rank == 1:
            exact_top1_passes += 1
        if query_id in set(quality["semantic_queries_top5"]):
            semantic_ranks[query_id] = target_rank

        print(
            "phase3e_query "
            f"id={query_id} rr={metric.mrr_at_10:.6f} "
            f"recall5={metric.recall_at_5:.6f} ndcg10={metric.ndcg_at_10:.6f} "
            f"target_rank={target_rank} latency_ms={latency_ms:.3f}"
        )

    aggregate = macro_average(metrics)
    p95 = _percentile(latencies, 0.95)
    exact_rate = exact_top1_passes / len(exact_query_ids)
    print(
        "phase3e_default_aggregate "
        f"mrr_at_10={aggregate.mrr_at_10:.6f} "
        f"recall_at_5={aggregate.recall_at_5:.6f} "
        f"ndcg_at_10={aggregate.ndcg_at_10:.6f} "
        f"exact_identifier_top1_rate={exact_rate:.6f} "
        f"p95_latency_ms={p95:.3f}"
    )

    if aggregate.mrr_at_10 < float(quality["mrr_at_10_min"]):
        raise RuntimeError("default hybrid MRR@10 failed the Phase 3E promotion floor")
    if aggregate.recall_at_5 < float(quality["recall_at_5_min"]):
        raise RuntimeError("default hybrid Recall@5 failed the Phase 3E promotion floor")
    if aggregate.ndcg_at_10 < float(quality["ndcg_at_10_min"]):
        raise RuntimeError("default hybrid nDCG@10 failed the Phase 3E promotion floor")
    if exact_rate < float(quality["exact_identifier_top1_rate_min"]):
        raise RuntimeError("default hybrid exact-identifier top-1 gate failed")
    for query_id in quality["semantic_queries_top5"]:
        rank = semantic_ranks.get(query_id, 0)
        if rank <= 0 or rank > 5:
            raise RuntimeError(f"semantic query {query_id} target rank {rank} is outside top 5")
    maximum_p95 = float(quality["warm_api_p95_ms_max"])
    if p95 > maximum_p95:
        raise RuntimeError(
            f"default hybrid API p95 {p95:.3f} ms exceeds {maximum_p95:.3f} ms"
        )
    print("phase3e_default_30_query_quality_gate=PASS")


async def _verify_explicit_bm25_opt_out(
    client: httpx.AsyncClient,
    query: str,
) -> None:
    body, _ = await _post_search(client, query=query, limit=5, mode="bm25")
    if body.get("mode_requested") != "bm25" or body.get("mode_used") != "bm25":
        raise RuntimeError("explicit BM25 opt-out did not use BM25")
    if body.get("degraded") is not False:
        raise RuntimeError("explicit BM25 opt-out unexpectedly reported degradation")
    results = body.get("results")
    if not isinstance(results, list) or not results:
        raise RuntimeError("explicit BM25 opt-out returned no results")
    for result in results:
        metadata = result.get("metadata") if isinstance(result, dict) else None
        if not isinstance(metadata, dict) or metadata.get("retrieval_mode") != "bm25":
            raise RuntimeError("explicit BM25 result has incorrect retrieval provenance")
    print("phase3e_explicit_bm25_opt_out=PASS")


async def _verify_lexical_only_upgrade(
    client: httpx.AsyncClient,
    store: OpenSearchStore,
    query: str,
) -> None:
    response = await store._request(
        "DELETE",
        f"/{store.settings.opensearch_vector_chunks_index}",
    )
    if response.status_code not in {200, 404}:
        raise RuntimeError(
            f"unable to remove vector index for upgrade test: HTTP {response.status_code}"
        )
    store._vector_index_ready = False

    body, _ = await _post_search(client, query=query, limit=5)
    if body.get("mode_requested") != "hybrid" or body.get("mode_used") != "bm25":
        raise RuntimeError("lexical-only upgrade did not degrade hybrid default to BM25")
    if body.get("degraded") is not True:
        raise RuntimeError("lexical-only upgrade did not report degradation")
    reason = body.get("degradation_reason")
    if not isinstance(reason, str) or not reason:
        raise RuntimeError("lexical-only upgrade did not report a degradation reason")
    results = body.get("results")
    if not isinstance(results, list) or not results:
        raise RuntimeError("lexical-only upgrade returned no BM25 results")
    print("phase3e_lexical_only_upgrade_degrades_to_bm25=PASS")


async def _verify_default_indexing(client: httpx.AsyncClient) -> None:
    response = await client.post("/v1/index", json={"urls": [EXAMPLE_URL]})
    if response.status_code != 200:
        raise RuntimeError(
            f"default index endpoint returned HTTP {response.status_code}: {response.text[:300]}"
        )
    body = response.json()
    items = body.get("items") if isinstance(body, dict) else None
    if not isinstance(items, list) or len(items) != 1 or not isinstance(items[0], dict):
        raise RuntimeError("default index endpoint returned an invalid item list")
    item = items[0]
    if item.get("indexed") is not True:
        raise RuntimeError(f"default lexical indexing failed: {item}")
    if item.get("vector_indexed") is not True:
        raise RuntimeError(f"default indexing did not write vectors: {item}")
    vector_chunks = item.get("vector_chunks_indexed")
    if not isinstance(vector_chunks, int) or vector_chunks <= 0:
        raise RuntimeError("default indexing reported no vector chunks")
    print("phase3e_default_indexing_attempts_vector_write=PASS")


async def _run() -> None:
    policy = _load_policy()
    api_url = os.environ.get("SUPRACRAWL_API_URL", "http://127.0.0.1:8080")
    opensearch_url = os.environ.get(
        "SUPRACRAWL_OPENSEARCH_URL",
        "http://127.0.0.1:9200",
    )

    defaults = Settings(_env_file=None)
    expected_defaults = policy["production_default_after_if_certified"]
    if defaults.search_mode != expected_defaults["search_mode"]:
        raise RuntimeError("application search_mode default does not match Phase 3E policy")
    if defaults.dense_enabled is not expected_defaults["dense_enabled"]:
        raise RuntimeError("application dense_enabled default does not match Phase 3E policy")

    frozen = policy["retrieval_configuration_must_not_change"]
    if defaults.dense_model_name != frozen["model"]:
        raise RuntimeError("Phase 3E changed the certified dense model")
    if defaults.dense_dimension != frozen["dimension"]:
        raise RuntimeError("Phase 3E changed the certified dense dimension")
    if defaults.hybrid_rrf_k != frozen["rrf_k"]:
        raise RuntimeError("Phase 3E changed the certified RRF k")

    settings = Settings(opensearch_url=opensearch_url)
    corpus = _load_jsonl(CORPUS_PATH) + _load_jsonl(EXACT_CORPUS_PATH)
    queries = _load_jsonl(QUERIES_PATH) + _load_jsonl(EXACT_QUERIES_PATH)
    exact_queries = _load_jsonl(EXACT_QUERIES_PATH)
    id_to_url = _validate_fixture(corpus, queries, minimum_queries=30)
    url_to_id = {url: document_id for document_id, url in id_to_url.items()}
    exact_query_ids = {query["id"] for query in exact_queries}

    store = OpenSearchStore(settings)
    embedder = DenseEmbedder(
        model_name=settings.dense_model_name,
        dimension=settings.dense_dimension,
        query_prefix=settings.dense_query_prefix,
        passage_prefix=settings.dense_passage_prefix,
    )

    try:
        await _clear_indices(store, settings)
        vector_chunks = await _seed_corpus_with_vectors(store, embedder, corpus)
        if vector_chunks < len(corpus):
            raise RuntimeError("Phase 3E corpus seeding wrote fewer vectors than documents")
        print(f"phase3e_seeded_vector_chunks={vector_chunks}")

        timeout = httpx.Timeout(180.0, connect=10.0)
        async with httpx.AsyncClient(base_url=api_url, timeout=timeout) as client:
            await _benchmark_default_api(
                client,
                policy,
                queries,
                url_to_id,
                exact_query_ids,
            )
            await _verify_explicit_bm25_opt_out(client, queries[0]["query"])
            await _verify_lexical_only_upgrade(client, store, queries[0]["query"])
            await _verify_default_indexing(client)
    finally:
        await _clear_indices(store, settings)
        await store.close()

    print("Phase 3E hybrid default promotion verification: PASS")


if __name__ == "__main__":
    asyncio.run(_run())
