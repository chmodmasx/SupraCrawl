from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path
from statistics import mean
from typing import Any

import httpx
from verify_real_vector_retrieval import _clear_indices, _seed_corpus_with_vectors
from verify_retrieval_baseline import _load_jsonl, _percentile, _validate_fixture

from supracrawl.config import Settings
from supracrawl.embeddings import DenseEmbedder
from supracrawl.evaluation import RetrievalMetrics, evaluate_ranking, macro_average, recall_at_k
from supracrawl.search import OpenSearchStore

ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = ROOT / "evaluation" / "phase3f_policy.json"
CORPUS_PATH = ROOT / "evaluation" / "corpus.jsonl"
QUERIES_PATH = ROOT / "evaluation" / "queries.jsonl"
EXACT_CORPUS_PATH = ROOT / "evaluation" / "phase3c_exact_corpus.jsonl"
EXACT_QUERIES_PATH = ROOT / "evaluation" / "phase3c_exact_queries.jsonl"


def _load_policy() -> dict[str, Any]:
    payload = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError("Phase 3F policy must be a JSON object")
    return payload


def _result_ranking(body: dict[str, Any], url_to_id: dict[str, str]) -> list[str]:
    results = body.get("results")
    if not isinstance(results, list) or not results:
        raise RuntimeError("search response has no non-empty results list")
    ranking: list[str] = []
    for result in results:
        if not isinstance(result, dict):
            raise RuntimeError("search response contains an invalid result")
        url = result.get("url")
        if not isinstance(url, str) or not url:
            raise RuntimeError("search result has no URL")
        ranking.append(url_to_id.get(url, url))
    return ranking


def _assert_mode(body: dict[str, Any], expected: str) -> None:
    if body.get("success") is not True:
        raise RuntimeError(f"{expected} response is not successful")
    if body.get("mode_requested") != expected or body.get("mode_used") != expected:
        raise RuntimeError(f"search response did not use requested {expected} mode")
    if body.get("degraded") is not False:
        raise RuntimeError(f"{expected} response unexpectedly reported degradation")


async def _post_search(
    client: httpx.AsyncClient,
    *,
    query: str,
    mode: str | None,
    limit: int,
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


def _metric_dict(metrics: RetrievalMetrics) -> dict[str, float]:
    return {
        "mrr_at_10": round(metrics.mrr_at_10, 6),
        "recall_at_5": round(metrics.recall_at_5, 6),
        "ndcg_at_10": round(metrics.ndcg_at_10, 6),
    }


async def _run() -> None:
    policy = _load_policy()
    baseline_policy = policy["baseline"]
    api_url = os.environ.get("SUPRACRAWL_API_URL", "http://127.0.0.1:8080")
    opensearch_url = os.environ.get(
        "SUPRACRAWL_OPENSEARCH_URL",
        "http://127.0.0.1:9200",
    )
    output_path = os.environ.get("PHASE3F_BASELINE_OUTPUT")

    defaults = Settings()
    if defaults.search_mode != "hybrid" or defaults.dense_enabled is not True:
        raise RuntimeError("Phase 3F requires the certified Phase 3E hybrid+dense defaults")

    candidate_pool_size = int(baseline_policy["candidate_pool_size"])
    if candidate_pool_size < 10 or candidate_pool_size > 20:
        raise RuntimeError("Phase 3F candidate pool must stay within the certified hybrid window")

    corpus = _load_jsonl(CORPUS_PATH) + _load_jsonl(EXACT_CORPUS_PATH)
    queries = _load_jsonl(QUERIES_PATH) + _load_jsonl(EXACT_QUERIES_PATH)
    id_to_url = _validate_fixture(
        corpus,
        queries,
        minimum_queries=int(baseline_policy["minimum_queries"]),
    )
    url_to_id = {url: document_id for document_id, url in id_to_url.items()}

    settings = Settings(opensearch_url=opensearch_url)
    store = OpenSearchStore(settings)
    embedder = DenseEmbedder(
        model_name=settings.dense_model_name,
        dimension=settings.dense_dimension,
        query_prefix=settings.dense_query_prefix,
        passage_prefix=settings.dense_passage_prefix,
    )

    bm25_metrics: list[RetrievalMetrics] = []
    hybrid_metrics: list[RetrievalMetrics] = []
    candidate_recall_at_10: list[float] = []
    bm25_latencies: list[float] = []
    hybrid_latencies: list[float] = []
    query_details: list[dict[str, Any]] = []

    try:
        await _clear_indices(store, settings)
        indexed_vector_chunks = await _seed_corpus_with_vectors(store, embedder, corpus)
        if indexed_vector_chunks < len(corpus):
            raise RuntimeError("Phase 3F vector seeding wrote fewer chunks than documents")

        timeout = httpx.Timeout(180.0, connect=10.0)
        async with httpx.AsyncClient(base_url=api_url, timeout=timeout) as client:
            warm_query = queries[0]["query"]
            warm_body, _ = await _post_search(
                client,
                query=warm_query,
                mode="hybrid",
                limit=candidate_pool_size,
            )
            _assert_mode(warm_body, "hybrid")

            omitted_body, _ = await _post_search(
                client,
                query=warm_query,
                mode=None,
                limit=candidate_pool_size,
            )
            _assert_mode(omitted_body, "hybrid")
            if _result_ranking(omitted_body, url_to_id) != _result_ranking(
                warm_body,
                url_to_id,
            ):
                raise RuntimeError("omitted-mode default differs from explicit hybrid ranking")

            for query in queries:
                bm25_body, bm25_latency_ms = await _post_search(
                    client,
                    query=query["query"],
                    mode="bm25",
                    limit=candidate_pool_size,
                )
                _assert_mode(bm25_body, "bm25")
                hybrid_body, hybrid_latency_ms = await _post_search(
                    client,
                    query=query["query"],
                    mode="hybrid",
                    limit=candidate_pool_size,
                )
                _assert_mode(hybrid_body, "hybrid")

                bm25_ranking = _result_ranking(bm25_body, url_to_id)
                hybrid_ranking = _result_ranking(hybrid_body, url_to_id)
                relevance = query["relevance"]
                bm25_metric = evaluate_ranking(bm25_ranking, relevance)
                hybrid_metric = evaluate_ranking(hybrid_ranking, relevance)
                recall_10 = recall_at_k(hybrid_ranking, relevance, k=10)

                bm25_metrics.append(bm25_metric)
                hybrid_metrics.append(hybrid_metric)
                candidate_recall_at_10.append(recall_10)
                bm25_latencies.append(bm25_latency_ms)
                hybrid_latencies.append(hybrid_latency_ms)
                query_details.append(
                    {
                        "id": query["id"],
                        "language": query.get("language"),
                        "bm25_top10": bm25_ranking[:10],
                        "hybrid_top10": hybrid_ranking[:10],
                        "hybrid_ndcg_at_10": round(hybrid_metric.ndcg_at_10, 6),
                        "candidate_recall_at_10": round(recall_10, 6),
                        "hybrid_latency_ms": round(hybrid_latency_ms, 3),
                    }
                )
    finally:
        await store.close()

    bm25 = macro_average(bm25_metrics)
    hybrid = macro_average(hybrid_metrics)
    candidate_recall = mean(candidate_recall_at_10)
    hybrid_p95 = _percentile(hybrid_latencies, 0.95)

    checks = {
        "hybrid_mrr_gte_bm25": (
            not baseline_policy["require_hybrid_mrr_gte_bm25"]
            or hybrid.mrr_at_10 >= bm25.mrr_at_10
        ),
        "hybrid_recall_at_5_gte_bm25": (
            not baseline_policy["require_hybrid_recall_at_5_gte_bm25"]
            or hybrid.recall_at_5 >= bm25.recall_at_5
        ),
        "hybrid_ndcg_gte_bm25": (
            not baseline_policy["require_hybrid_ndcg_gte_bm25"]
            or hybrid.ndcg_at_10 >= bm25.ndcg_at_10
        ),
        "candidate_recall_at_10": (
            candidate_recall >= float(baseline_policy["minimum_candidate_recall_at_10"])
        ),
        "warm_hybrid_api_p95": (
            hybrid_p95 <= float(baseline_policy["warm_hybrid_api_p95_ms_max"])
        ),
    }
    passed = all(checks.values())
    report = {
        "schema_version": 1,
        "phase": "3F",
        "benchmark": policy["benchmark"],
        "base_main_sha": policy["base_main_sha"],
        "documents": len(corpus),
        "queries": len(queries),
        "candidate_pool_size": candidate_pool_size,
        "aggregate": {
            "bm25": _metric_dict(bm25),
            "hybrid": _metric_dict(hybrid),
            "hybrid_ndcg_delta_vs_bm25": round(hybrid.ndcg_at_10 - bm25.ndcg_at_10, 6),
            "hybrid_candidate_recall_at_10": round(candidate_recall, 6),
        },
        "latency_ms": {
            "bm25_p50": round(_percentile(bm25_latencies, 0.50), 3),
            "bm25_p95": round(_percentile(bm25_latencies, 0.95), 3),
            "hybrid_p50": round(_percentile(hybrid_latencies, 0.50), 3),
            "hybrid_p95": round(hybrid_p95, 3),
            "hybrid_mean": round(mean(hybrid_latencies), 3),
        },
        "baseline": {"passed": passed, "checks": checks},
        "queries_detail": query_details,
    }

    rendered = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    print(rendered)
    print("PHASE3F_BASELINE_JSON=" + json.dumps(report, ensure_ascii=False, sort_keys=True))
    if output_path:
        Path(output_path).write_text(rendered + "\n", encoding="utf-8")
    if not passed:
        raise RuntimeError("Phase 3F frozen baseline policy failed")
    print("Phase 3F frozen hybrid baseline: PASS")


if __name__ == "__main__":
    asyncio.run(_run())
